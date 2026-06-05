"""
models/yolo_wrapper.py — AranduBackbone para YOLO.

Backbone denso basado en ConvNeXt V2 Tiny pre-entrenado con AranduSSL (MoCo v3).
Extrae features espaciales en 4 escalas: P2 (stride 4), P3 (stride 8),
P4 (stride 16), P5 (stride 32).

Características clave:
  - Coordinate Attention en P2 y P3 para sensibilidad a micro-lesiones.
  - GroupNorm(32) en lugar de BatchNorm2d (estabilidad en transferencia de dominio).
  - Context Gate inicializado a 0 → Sigmoid(0)=0.5 (mezcla neutra al inicio).
  - 4-Phase Unfreeze Curriculum: A (solo adapters) → B (+stage3) → C (+stage2) → D (full).
  - Carga de pesos SSL desde checkpoint ModelBase (moco.py).

Canales de ConvNeXt V2 Tiny por stage:
  Stage 0 → P2: 96 ch  (stride 4)
  Stage 1 → P3: 192 ch (stride 8)
  Stage 2 → P4: 384 ch (stride 16)
  Stage 3 → P5: 768 ch (stride 32)
"""

import logging

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Helper: GroupNorm con fallback automático de grupos
# ---------------------------------------------------------------------------

def _gn(channels: int, num_groups: int = 32) -> nn.GroupNorm:
    """GroupNorm con num_groups adaptado si channels < num_groups."""
    g = num_groups
    while channels % g != 0 and g > 1:
        g -= 1
    return nn.GroupNorm(g, channels)


# ---------------------------------------------------------------------------
# Coordinate Attention — sensibilidad espacial micro-lesión
# ---------------------------------------------------------------------------

class CoordinateAttention(nn.Module):
    """
    Coordinate Attention (Hou et al., CVPR 2021).

    Descompone la atención en dos mapas 1D (H y W) para preservar información
    posicional precisa — crítico para detectar micro-lesiones en imágenes aéreas.
    Se aplica SOLO en P2 y P3 donde la resolución espacial es alta.

    Pipeline:
        x → Pool_H + Pool_W → cat → Conv1×1 + GN + Hardswish
          → Split → Conv_H + Conv_W → Sigmoid → x * attn_H * attn_W
    """
    def __init__(self, channels: int, reduction: int = 32):
        super().__init__()
        mid = max(8, channels // reduction)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # [B, C, H, 1]
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # [B, C, 1, W]

        self.conv_hw = nn.Conv2d(channels, mid, kernel_size=1, bias=False)
        self.norm    = _gn(mid)
        self.act     = nn.Hardswish(inplace=True)

        self.conv_h  = nn.Conv2d(mid, channels, kernel_size=1, bias=False)
        self.conv_w  = nn.Conv2d(mid, channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.pool_h(x)                          # [B, C, H, 1]
        w = self.pool_w(x).permute(0, 1, 3, 2)     # [B, C, W, 1]
        hw = torch.cat([h, w], dim=2)               # [B, C, H+W, 1]
        hw = self.act(self.norm(self.conv_hw(hw)))
        h_feat, w_feat = torch.split(hw, [H, W], dim=2)
        w_feat = w_feat.permute(0, 1, 3, 2)        # [B, C, 1, W]
        attn_h = torch.sigmoid(self.conv_h(h_feat))
        attn_w = torch.sigmoid(self.conv_w(w_feat))
        return x * attn_h * attn_w


# ---------------------------------------------------------------------------
# SpatialFeatureAdapter — Traductor SSL → YOLO
# ---------------------------------------------------------------------------

class SpatialFeatureAdapter(nn.Module):
    """
    Adaptador espacial para traducir features del encoder SSL al espacio de YOLO.

    Pipeline:
        x (in_channels)
          → Conv1×1 + GroupNorm + SiLU          [compresión a out_channels]
          → DW-Conv3×3 + GroupNorm + SiLU + Dropout(0.05)  [localidad]
          → [CoordinateAttention]               [opcional, P2 y P3]
          → Context Gate: α*shortcut + (1-α)*local_feat   [combinación convexa]

    Context Gate inicializado a 0 → Sigmoid(0)=0.5 al inicio.
    Garantiza que el adaptador no destruya las representaciones SSL durante
    las primeras iteraciones del fine-tuning.
    """

    def __init__(self, in_channels: int, out_channels: int, use_coord_attn: bool = False):
        super().__init__()

        # 1. Compresión de canales
        self.compress = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            _gn(out_channels),
            nn.SiLU(inplace=True)
        )

        # 2. Extracción de contexto local (textura, bordes, micro-lesiones)
        self.local_context = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1,
                      bias=False, groups=out_channels),
            _gn(out_channels),
            nn.SiLU(inplace=True),
            nn.Dropout(p=0.05)  # Regularización ligera — no destruye canales enteros
        )

        # 3. Atención espacial posicional (P2 y P3)
        self.coord_attn = CoordinateAttention(out_channels) if use_coord_attn else nn.Identity()

        # 4. Shortcut para preservar señal cruda del encoder
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                _gn(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

        # 5. Residual Gate — Parameter escalar inicializado en 0 (Y = X + beta * T(X))
        self.beta = nn.Parameter(torch.zeros(1))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        # Beta se inicializa explícitamente en 0 en la declaración del Parameter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut   = self.shortcut(x)
        local_feat = self.coord_attn(self.local_context(self.compress(x)))
        return shortcut + self.beta * local_feat


# ---------------------------------------------------------------------------
# AranduBackbone — Backbone Denso ConvNeXt V2 para YOLO
# ---------------------------------------------------------------------------

class AranduBackbone(nn.Module):
    """
    Backbone AranduSSL para YOLO basado en ConvNeXt V2 Tiny.

    Extrae P2/P3/P4/P5 usando timm (features_only=True) y aplica adaptadores
    espaciales con GroupNorm y Coordinate Attention en P2/P3.

    Carga pesos del encoder SSL desde un checkpoint de ModelBase (moco.py).
    El mapa de pesos es directo: ModelBase.encoder == features_only backbone.

    Curriculum de Descongelado (4 fases):
      Fase 1 (A): Solo adaptadores. Backbone 100% congelado.
      Fase 2 (B): + Stage 3 (P5, 768ch).
      Fase 3 (C): + Stage 2 (P4, 384ch).
      Fase 4 (D): Full fine-tuning. Backbone completo.
    """

    # Canales nativos de ConvNeXt V2 Tiny por stage
    STAGE_CHANNELS = (96, 192, 384, 768)  # P2, P3, P4, P5

    def __init__(
        self,
        moco_checkpoint_path: str = None,
        freeze_phase: int = 1,
        yolo_channels: tuple = (128, 256, 512, 1024),  # P2, P3, P4, P5
        use_coord_attn: bool = True,
    ):
        super().__init__()
        self.logger = logging.getLogger("AranduSSL")
        self.phase  = freeze_phase

        # ----------------------------------------------------------------
        # Backbone: ConvNeXt V2 Tiny con extracción de features espaciales
        # features_only=True devuelve [P2, P3, P4, P5] en forward()
        # ----------------------------------------------------------------
        pretrained_imagenet = (moco_checkpoint_path is None)
        self.backbone = timm.create_model(
            "convnextv2_tiny",
            pretrained=pretrained_imagenet,
            features_only=True,
            out_indices=(0, 1, 2, 3)
        )
        self.logger.info(
            f"[AranduBackbone] ConvNeXt V2 Tiny inicializado "
            f"(pretrained_imagenet={pretrained_imagenet})"
        )

        # Cargar pesos SSL si se provee checkpoint
        if moco_checkpoint_path:
            self._load_ssl_weights(moco_checkpoint_path)

        # ----------------------------------------------------------------
        # Adaptadores espaciales — uno por nivel de feature
        # P2 y P3: CoordAtt activado (alta resolución, micro-lesiones)
        # P4 y P5: sin CoordAtt (objetos más grandes, contexto semántico)
        # ----------------------------------------------------------------
        self.adapter_p2 = SpatialFeatureAdapter(
            self.STAGE_CHANNELS[0], yolo_channels[0], use_coord_attn=use_coord_attn
        )
        self.adapter_p3 = SpatialFeatureAdapter(
            self.STAGE_CHANNELS[1], yolo_channels[1], use_coord_attn=use_coord_attn
        )
        self.adapter_p4 = SpatialFeatureAdapter(
            self.STAGE_CHANNELS[2], yolo_channels[2], use_coord_attn=False
        )
        self.adapter_p5 = SpatialFeatureAdapter(
            self.STAGE_CHANNELS[3], yolo_channels[3], use_coord_attn=False
        )

        self.set_training_phase(freeze_phase)

    # ----------------------------------------------------------------
    # Carga de pesos SSL
    # ----------------------------------------------------------------

    def _load_ssl_weights(self, checkpoint_path: str) -> None:
        """
        Carga pesos del encoder SSL (ModelBase.encoder) al backbone features_only.

        ModelBase.encoder y el backbone features_only son el mismo ConvNeXt V2 Tiny
        instanciado con opciones diferentes (num_classes=0 vs features_only=True).
        Los pesos del tronco convolucional son idénticos — solo difieren las cabezas.

        Formato de checkpoint soportado:
          - {'model_q': {...}} → checkpoint de entrenamiento SSL
          - {encoder.*: ...}   → pesos planos del encoder
          - DDP/compile: prefijos 'module.' y '_orig_mod.' eliminados automáticamente
        """
        self.logger.info(f"[AranduBackbone] Cargando pesos SSL desde: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

        # Extraer el state_dict del encoder (query o key)
        source = ckpt.get("model_q", ckpt.get("model_k", ckpt))

        encoder_sd = {}
        for k, v in source.items():
            # Limpiar prefijos DDP y torch.compile
            k = k.replace("_orig_mod.", "").replace("module.", "")
            # El encoder SSL está bajo el prefijo 'encoder.'
            if k.startswith("encoder."):
                encoder_sd[k[len("encoder."):]] = v

        if not encoder_sd:
            self.logger.warning(
                "[AranduBackbone] No se encontró prefijo 'encoder.' en el checkpoint. "
                "Intentando cargar el state_dict completo como backbone."
            )
            encoder_sd = {k.replace("_orig_mod.", "").replace("module.", ""): v
                          for k, v in source.items()
                          if not any(k.startswith(p) for p in ("projector.", "predictor.", "queue"))}

        missing, unexpected = self.backbone.load_state_dict(encoder_sd, strict=False)
        self.logger.info(
            f"[AranduBackbone] Pesos cargados. "
            f"Missing: {len(missing)}, Unexpected: {len(unexpected)}"
        )
        if missing:
            self.logger.debug(f"  Missing keys (muestra): {missing[:5]}")

    # ----------------------------------------------------------------
    # Curriculum de Descongelado
    # ----------------------------------------------------------------

    def set_training_phase(self, phase: int) -> None:
        """
        Configura el curriculum de descongelado progresivo.

        Fase 1 (A) — Solo Adaptadores:
          Backbone 100% congelado. Solo los adaptadores y la cabeza YOLO aprenden.
          Objetivo: alinear el espacio SSL con el detector sin destruir los priors.

        Fase 2 (B) — + Stage 3 (P5):
          Libera stage 3 de ConvNeXt (768ch). Los adaptadores y P5 se actualizan.
          Objetivo: adaptar semántica de alto nivel a detección.

        Fase 3 (C) — + Stage 2 (P4):
          Libera stage 2 (384ch). Gradientes alcanzan características intermedias.
          Objetivo: refinar representaciones de objetos medianos.

        Fase 4 (D) — Full Fine-Tuning:
          Todo el sistema actualiza gradientes.
          Objetivo: convergencia final ajustada al dominio aéreo.
        """
        if phase not in (1, 2, 3, 4):
            raise ValueError(f"Fase inválida: {phase}. Usar 1, 2, 3 o 4.")

        self.phase = phase

        # Paso 1: Congelar TODO el backbone
        for p in self.backbone.parameters():
            p.requires_grad = False

        # Paso 2: Adaptadores siempre entrenables
        for adapter in (self.adapter_p2, self.adapter_p3, self.adapter_p4, self.adapter_p5):
            for p in adapter.parameters():
                p.requires_grad = True

        # Paso 3: Descongelar stages según la fase
        if phase >= 2:
            if hasattr(self.backbone, 'stages_3'):
                for p in self.backbone.stages_3.parameters(): p.requires_grad = True
            elif hasattr(self.backbone, 'stages'):
                for p in self.backbone.stages[3].parameters(): p.requires_grad = True

        if phase >= 3:
            if hasattr(self.backbone, 'stages_2'):
                for p in self.backbone.stages_2.parameters(): p.requires_grad = True
            elif hasattr(self.backbone, 'stages'):
                for p in self.backbone.stages[2].parameters(): p.requires_grad = True

        if phase >= 4:
            for p in self.backbone.parameters():
                p.requires_grad = True  # Todo

        phase_desc = {
            1: "A — Solo Adaptadores (backbone 100% congelado)",
            2: "B — + Stage3/P5 liberado",
            3: "C — + Stage2/P4 liberado",
            4: "D — Full Fine-Tuning",
        }
        self.logger.info(f"[AranduBackbone] Fase {phase_desc[phase]}")

        # Aplicar modo train/eval correctamente
        self.train(mode=self.training)

    def train(self, mode: bool = True):
        """
        Sobrescribe train() para proteger la normalización del backbone
        durante las fases de congelamiento parcial.

        En ConvNeXt V2, LayerNorm interno depende del estado train/eval.
        Mantener en eval() las capas congeladas evita que las estadísticas
        se degraden con batches pequeños durante el fine-tuning.
        """
        super().train(mode)
        if mode and self.phase < 4:
            # Congelado parcial: backbone en eval, stages liberados en train
            self.backbone.eval()
            if self.phase >= 2 and hasattr(self.backbone, 'stages'):
                self.backbone.stages[3].train()
            if self.phase >= 3 and hasattr(self.backbone, 'stages'):
                self.backbone.stages[2].train()
        return self

    # ----------------------------------------------------------------
    # Forward
    # ----------------------------------------------------------------

    def forward(self, x: torch.Tensor):
        """
        Extrae P2, P3, P4, P5 y aplica adaptadores.

        Args:
            x: Tensor [B, 3, H, W] — imagen de entrada (H=W=512 para entrenamiento,
               640 para inferencia en drones).

        Returns:
            [p2, p3, p4, p5]: lista de feature maps en orden ascendente de stride.
              p2: [B, yolo_channels[0], H/4,  W/4]   — micro-lesiones
              p3: [B, yolo_channels[1], H/8,  W/8]   — lesiones pequeñas
              p4: [B, yolo_channels[2], H/16, W/16]  — objetos medios
              p5: [B, yolo_channels[3], H/32, W/32]  — contexto semántico
        """
        # Extraer los 4 niveles de feature maps del backbone ConvNeXt V2
        p2_raw, p3_raw, p4_raw, p5_raw = self.backbone(x)

        # Adaptar cada nivel al espacio de YOLO
        p2 = self.adapter_p2(p2_raw)
        p3 = self.adapter_p3(p3_raw)
        p4 = self.adapter_p4(p4_raw)
        p5 = self.adapter_p5(p5_raw)

        return [p2, p3, p4, p5]


# ---------------------------------------------------------------------------
# Alias para registro en Ultralytics
# ---------------------------------------------------------------------------

# ablation_runner.py registra este alias antes de instanciar el modelo YOLO:
#   from ultralytics.nn.modules import __dict__ as unn
#   unn['AranduYOLOWrapper'] = AranduBackbone
AranduYOLOWrapper = AranduBackbone
