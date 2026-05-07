import torch
import torch.nn as nn
from models.moco import ModelBase

class SpatialFeatureAdapter(nn.Module):
    """
    Capas de traducción de MoCo a YOLO.
    [AJUSTE 1 y 4]: Implementa conexión residual y Dropout para evitar 
    sobre-filtrado y prevenir que el adaptador se vuelva un 'parche universal'.
    """
    def __init__(self, in_channels, out_channels, use_residual=True, use_context_gate=True):
        super().__init__()
        self.use_residual = use_residual
        self.use_context_gate = use_context_gate
        
        # 1. Compresión de canales (Alineación de Distribución)
        self.compress = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )
        
        # 2. Inyección de localidad (Búsqueda de bordes y texturas)
        self.local_context = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False, groups=out_channels),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            # Regularización quirúrgica: Dropout estándar y muy ligero en lugar de Dropout2d 
            # para no destruir canales enteros con info de micro-lesiones (Ajuste 3)
            nn.Dropout(p=0.05)
        )
        
        # 3. Shortcut (Preservación de señal cruda para micro-lesiones)
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

        # [AJUSTE 1 EXPERTO]: Residual dependiente del contenido (Adaptación Contextual)
        if self.use_context_gate:
            self.context_gate = nn.Sequential(
                # Bias en True para poder forzar el 0 inicial
                nn.Conv2d(out_channels, 1, kernel_size=1, bias=True),
                nn.Sigmoid()
            )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
        # [EVITAR SATURACIÓN DEL GATE]: Inicializamos la convolución del gate a 0.
        # Esto fuerza que Sigmoid(0) = 0.5 en toda la imagen al inicio.
        # Evita la saturación prematura y garantiza una exploración equilibrada.
        if hasattr(self, 'context_gate'):
            nn.init.constant_(self.context_gate[0].weight, 0)
            nn.init.constant_(self.context_gate[0].bias, 0)

    def forward(self, x):
        compressed = self.compress(x)
        local_feat = self.local_context(compressed)
        
        if self.use_residual:
            # [BALANCE EXPLÍCITO]: Combinación convexa (alpha vs 1-alpha)
            shortcut_feat = self.shortcut(x)
            
            if self.use_context_gate:
                alpha = self.context_gate(shortcut_feat)
            else:
                # Ablation Study: Modelo 2 (Sin Gate). Alpha fijo en 0.5.
                alpha = 0.5
                
            return (alpha * shortcut_feat) + ((1.0 - alpha) * local_feat)
        return local_feat


class AranduBackbone(nn.Module):
    """
    Backbone Denso de ARANDU-AI para YOLO.
    Expone P3, P4, P5 con protección estricta de estadísticas BN.
    """
    def __init__(self, moco_checkpoint_path=None, freeze_phase=1, yolo_channels=(256, 512, 1024), use_context_gate=True):
        super().__init__()
        
        print("[*] Inicializando AranduBackbone Denso (ResNet-50 MoCo v3)...")
        moco_model = ModelBase()
        self.resnet = moco_model.encoder
        self.phase = freeze_phase
        
        if moco_checkpoint_path:
            print(f"[*] Cargando pesos desde: {moco_checkpoint_path}")
            state_dict = torch.load(moco_checkpoint_path, map_location='cpu', weights_only=True)
            
            encoder_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.encoder.'):
                    encoder_state_dict[k.replace('module.encoder.', '')] = v
                elif k.startswith('encoder.'):
                    encoder_state_dict[k.replace('encoder.', '')] = v
                elif not any(k.startswith(prefix) for prefix in ['projector.', 'predictor.', 'queue']):
                    encoder_state_dict[k] = v
            
            self.resnet.load_state_dict(encoder_state_dict, strict=False)
            
        if hasattr(self.resnet, 'avgpool'): del self.resnet.avgpool
        if hasattr(self.resnet, 'fc'): del self.resnet.fc
            
        # Adaptadores Espaciales (P3 es vital para soja, recibe residual)
        self.adapter_p3 = SpatialFeatureAdapter(512, yolo_channels[0], use_residual=True, use_context_gate=use_context_gate)
        self.adapter_p4 = SpatialFeatureAdapter(1024, yolo_channels[1], use_residual=True, use_context_gate=use_context_gate)
        self.adapter_p5 = SpatialFeatureAdapter(2048, yolo_channels[2], use_residual=True, use_context_gate=use_context_gate)
        
        self.set_training_phase(freeze_phase)

    def train(self, mode=True):
        """
        [AJUSTE 2]: Transición inteligente de BatchNorm.
        Sobrescribe el comportamiento por defecto de PyTorch.
        """
        super().train(mode)
        if mode:
            if self.phase == 1:
                # Fase 1: ResNet 100% en eval para proteger medias móviles.
                self.resnet.eval()
            elif self.phase == 2:
                # Fase 2: Mantenemos el core congelado, pero permitimos que los BNs 
                # de las capas liberadas (layer3 y layer4) pasen a modo train()
                # Esto alivia la "tensión interna" al permitir adaptación progresiva.
                self.resnet.eval()
                self.resnet.layer3.train()
                self.resnet.layer4.train()
        return self

    def set_training_phase(self, phase):
        self.phase = phase
        print(f"[*] Configurando Fase de Entrenamiento: {phase}")
        
        for param in self.resnet.parameters():
            param.requires_grad = False
            
        for adapter in [self.adapter_p3, self.adapter_p4, self.adapter_p5]:
            for param in adapter.parameters():
                param.requires_grad = True

        if phase == 1:
            print("    -> Fase 1: Backbone 100% congelado (incluyendo BN). Entrenando solo Adaptadores, Neck y Head.")
        elif phase == 2:
            for param in self.resnet.layer3.parameters():
                param.requires_grad = True
            for param in self.resnet.layer4.parameters():
                param.requires_grad = True
            print("    -> Fase 2: Ajuste Fino Selectivo. Liberadas layer3 y layer4 (BNs siguen congelados).")
        elif phase == 3:
            for param in self.resnet.parameters():
                param.requires_grad = True
            print("    -> Fase 3: Unfreeze Global. BNs liberados. Todo el sistema actualiza gradientes.")
        else:
            raise ValueError("Fase inválida. Usa 1, 2 o 3.")
            
        # Asegurar que el estado train/eval se aplique correctamente al cambiar de fase
        self.train(self.training)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x) 
        
        p3_raw = self.resnet.layer2(x)
        p3 = self.adapter_p3(p3_raw)
        
        p4_raw = self.resnet.layer3(p3_raw)
        p4 = self.adapter_p4(p4_raw)
        
        p5_raw = self.resnet.layer4(p4_raw)
        p5 = self.adapter_p5(p5_raw)

        return [p3, p4, p5]

