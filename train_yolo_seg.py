"""
train_yolo_seg.py — Entrenamiento YOLO SEGMENTATION con AranduBackbone SSL (MoCo v3)

Currículum de 4 Fases:
  Fase A: Solo adaptadores entrenan. Backbone 100% congelado. (10 épocas)
  Fase B: + Stage 3 (P5, semántica global). LR backbone reducido. (15 épocas)
  Fase C: + Stage 2 (P4, features medias). LR backbone mayor. (15 épocas)
  Fase D: Full fine-tuning. Todo el modelo. (epochs restantes)

Uso:
    python train_yolo_seg.py --data /ruta/data.yaml --encoder /ruta/moco_encoder_ready.pth

Ejemplo Kaggle:
    python train_yolo_seg.py \
        --data /kaggle/input/mi-dataset-seg/data.yaml \
        --encoder /kaggle/input/mi-encoder/moco_encoder_ready.pth \
        --epochs 100 --batch 16 --imgsz 640
"""

import argparse
import os
import sys
import logging

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
local_ultralytics = os.path.join(BASE_DIR, "ultralytics")
if os.path.isdir(os.path.join(local_ultralytics, "ultralytics")):
    sys.path.insert(0, local_ultralytics)
sys.path.insert(0, BASE_DIR)

import torch

try:
    import ultralytics.nn.modules as nn_modules
    from ultralytics import YOLO
except (ImportError, ModuleNotFoundError) as e:
    sys.stderr.write(
        "\n" + "="*75 + "\n"
        "❌ ERROR: El paquete 'ultralytics' no está instalado en este entorno Python.\n"
        "=========================================================================\n"
        "Si estás en Kaggle/Colab, ejecuta esta orden en una celda previa:\n\n"
        "    !pip install -q ultralytics\n\n"
        "O si el proyecto tiene el submódulo ultralytics descargado:\n\n"
        "    !git submodule update --init --recursive\n"
        "=========================================================================\n\n"
    )
    sys.exit(1)

from models.yolo_wrapper import AranduBackbone

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("AranduYOLO_Seg")


# ---------------------------------------------------------------------------
# Registro del wrapper en Ultralytics
# ---------------------------------------------------------------------------

# Variables globales para serialización unpicklable de AranduYOLOWrapper
_GLOBAL_ENCODER_PATH = None
_GLOBAL_FREEZE_PHASE = 1
_GLOBAL_USE_COORD_ATTN = True

class AranduYOLOWrapper(AranduBackbone):
    """Wrapper top-level para que torch.save / pickle puedan serializar el modelo."""
    def __init__(self, *args, **kwargs):
        if _GLOBAL_ENCODER_PATH:
            kwargs['moco_checkpoint_path'] = _GLOBAL_ENCODER_PATH
        kwargs['freeze_phase']         = _GLOBAL_FREEZE_PHASE
        kwargs['use_coord_attn']       = _GLOBAL_USE_COORD_ATTN
        super().__init__(*args, **kwargs)

def register_backbone(encoder_path: str, phase: int = 1, use_coord_attn: bool = True):
    """Registra AranduYOLOWrapper en el espacio de nombres de Ultralytics."""
    global _GLOBAL_ENCODER_PATH, _GLOBAL_FREEZE_PHASE, _GLOBAL_USE_COORD_ATTN
    _GLOBAL_ENCODER_PATH = encoder_path
    _GLOBAL_FREEZE_PHASE = phase
    _GLOBAL_USE_COORD_ATTN = use_coord_attn

    setattr(nn_modules, 'AranduYOLOWrapper', AranduYOLOWrapper)
    setattr(sys.modules['ultralytics.nn.modules'], 'AranduYOLOWrapper', AranduYOLOWrapper)
    
    # Inject into tasks for globals() access during parse_model
    import ultralytics.nn.tasks
    setattr(sys.modules['ultralytics.nn.tasks'], 'AranduYOLOWrapper', AranduYOLOWrapper)
    logger.info(f" AranduYOLOWrapper registrado — fase={phase}, coord_attn={use_coord_attn}")
    return AranduYOLOWrapper


def find_phase_weights(model, project: str, name: str, filename: str = "last.pt") -> str:
    """Resuelve la ruta física del archivo de pesos guardado por Ultralytics."""
    candidates = []
    if hasattr(model, "trainer") and hasattr(model.trainer, "save_dir"):
        candidates.append(os.path.join(str(model.trainer.save_dir), "weights", filename))

    candidates.extend([
        os.path.join("runs", "segment", project, name, "weights", filename),
        os.path.join("runs", "detect", project, name, "weights", filename),
        os.path.join(project, name, "weights", filename),
    ])

    for path in candidates:
        if os.path.isfile(path):
            return path

    if filename == "last.pt":
        return find_phase_weights(model, project, name, filename="best.pt")

    return candidates[0]


# ---------------------------------------------------------------------------
# Utilidad: contar parámetros entrenables
# ---------------------------------------------------------------------------

def count_trainable(model) -> tuple[int, int]:
    total    = sum(p.numel() for p in model.model.parameters())
    trainable = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    return trainable, total


# ---------------------------------------------------------------------------
# Currículum de Descongelado
# ---------------------------------------------------------------------------

def apply_phase(model: YOLO, phase: int, lr: float) -> tuple[float, list[str]]:
    """Aplica la fase de descongelado y retorna (lr_efectivo, freeze_names).

    Returns:
        tuple: (lr_efectivo, freeze_names) donde freeze_names es la lista de
        prefijos a pasar al parámetro `freeze=` de model.train() para que
        Ultralytics NO deshaga nuestro congelamiento progresivo.
    """
    backbone = None
    for module in model.model.modules():
        if isinstance(module, AranduBackbone):
            backbone = module
            break

    if backbone is None:
        logger.warning(" AranduBackbone no encontrado en el modelo. Ignorando fase.")
        return lr, []

    backbone.set_training_phase(phase)
    freeze_names = backbone.get_freeze_names()
    trainable, total = count_trainable(model)

    phase_names = {1: "A — Solo Adaptadores", 2: "B — +Stage3 P5",
                   3: "C — +Stage2 P4",       4: "D — Full Fine-Tuning"}
    lr_factors  = {1: 1.0, 2: 0.5, 3: 0.8, 4: 1.0}

    effective_lr = lr * lr_factors[phase]
    logger.info(f" Fase {phase_names[phase]}")
    logger.info(f"   Parámetros entrenables: {trainable/1e6:.1f}M / {total/1e6:.1f}M total")
    logger.info(f"   LR efectivo: {effective_lr:.2e}")
    logger.info(f"   Freeze Ultralytics: {len(freeze_names)} grupos → {freeze_names}")
    return effective_lr, freeze_names


# ---------------------------------------------------------------------------
# Entrenamiento principal
# ---------------------------------------------------------------------------

def train(args):
    # Validaciones
    if not os.path.isfile(args.data):
        raise FileNotFoundError(f"data.yaml no encontrado: {args.data}")
    if not os.path.isfile(args.encoder):
        raise FileNotFoundError(f"Encoder SSL no encontrado: {args.encoder}")

    logger.info("=" * 60)
    logger.info("🌱 ARANDU-AI YOLO SEGMENTATION — Entrenamiento con Currículum SSL 🌱")
    logger.info(f"   Dataset      : {args.data}")
    logger.info(f"   Encoder      : {args.encoder}")
    logger.info(f"   Épocas       : {args.epochs} | Batch: {args.batch} | Imgsz: {args.imgsz}")
    logger.info(f"   Modo         : {'🚀 FAST (3 Fases)' if args.fast else '🐢 STANDARD (4 Fases)'}")
    logger.info(f"   Cache RAM/Disk: {args.cache} | Workers: {args.workers}")
    logger.info("=" * 60)

    project      = args.project
    base_lr      = args.lr
    imgsz        = args.imgsz
    batch        = args.batch
    device       = int(os.environ["LOCAL_RANK"]) if "LOCAL_RANK" in os.environ else args.device
    workers      = args.workers
    cache_opt    = None if args.cache.lower() in ("none", "false") else args.cache.lower()
    close_mosaic = args.close_mosaic


    model_yaml   = args.cfg if args.cfg else ("arandu_yolo26_slim_seg.yaml" if args.slim else "arandu_yolo26_seg.yaml")
    logger.info(f"   Modelo Spec  : {model_yaml}")

    imgsz_final  = args.imgsz_final if args.imgsz_final > 0 else imgsz

    # ── Augmentaciones optimizadas para detección de manchas foliares ──────
    # Las manchas de frogeye son pequeñas, con colores sutilmente distintos
    # al tejido sano. Estos parámetros aumentan la sensibilidad del modelo
    # a texturas finas, calibran la pérdida de clasificación e imponen
    # resolución de solapamientos entre máscaras.
    aug_params = dict(
        degrees      = 15,      # Hojas son isótropas → rotación libre
        scale        = 0.3,     # Zoom a lesiones pequeñas
        hsv_h        = 0.02,    # Variación de matiz sutil (manchas vs sano)
        hsv_s        = 0.8,     # Saturación amplia para robustez
        hsv_v        = 0.5,     # Brillo moderado
        mosaic       = 1.0,     # Mosaic completo
        cos_lr       = True,    # Cosine annealing (estándar para SSL→downstream)
        cls          = 0.7,     # Ligeramente mayor ponderación de clasificación
        overlap_mask = True,    # Calibración de pérdidas en zonas con solapamiento
    )

    if args.fast:
        # ══════════════════════════════════════════════════════════════════
        # MODO ACELERADO (3 Fases)
        #
        #   Fase 1: Warmup de Adaptadores — backbone 100% congelado.
        #           Solo adapters + neck + head aprenden.
        #
        #   Fase 2: Descongelar Stages 2+3 (P4+P5) — las capas semánticas
        #           del backbone se adaptan al dominio, pero Stages 0+1
        #           (texturas de bajo nivel aprendidas por SSL) se preservan.
        #
        #   Fase 3: Full Fine-Tuning — todo se actualiza con LR reducido
        #           para convergencia final.
        #
        # Esta distribución de 3 fases evita el salto abrupto de phase=1
        # (todo congelado) a phase=4 (todo liberado) que destruía las
        # representaciones de textura del encoder SSL.
        # ══════════════════════════════════════════════════════════════════
        epochs_f1 = max(3, min(5, args.epochs // 8))
        epochs_f2 = max(5, int((args.epochs - epochs_f1) * 0.35))
        epochs_f3 = max(1, args.epochs - epochs_f1 - epochs_f2)
        logger.info(f"📋 Distribución FAST 3-Fases: Warmup={epochs_f1} | P4+P5={epochs_f2} | FullFT={epochs_f3}")

        # ── FASE 1: Warmup solo Adaptadores (Backbone congelado) ─────────
        logger.info("\n" + "─"*50)
        logger.info("⚡ FASE 1/3 — Warmup de Adaptadores (Backbone congelado)")
        logger.info("─"*50)

        register_backbone(args.encoder, phase=1, use_coord_attn=True)
        model = YOLO(model_yaml, task="segment")

        lr_f1, freeze_f1 = apply_phase(model, phase=1, lr=base_lr)

        model.train(
            task          = "segment",
            data          = args.data,
            epochs        = epochs_f1,
            imgsz         = imgsz,
            batch         = batch,
            lr0           = lr_f1,
            lrf           = 0.1,
            weight_decay  = 0.0005,
            warmup_epochs = 1,
            freeze        = freeze_f1,         # ← CRÍTICO: protege el backbone
            optimizer     = "AdamW",
            amp           = True,
            device        = device,
            workers       = workers,
            cache         = cache_opt,
            project       = project,
            name          = "Fast_F1_Warmup",
            seed          = 42,
            verbose       = True,
            save          = True,
            exist_ok      = True,
            **aug_params,
        )

        f1_weights = find_phase_weights(model, project, "Fast_F1_Warmup", "last.pt")

        # ── FASE 2: Descongelar Stages 2+3 (P4+P5 adaptan al dominio) ────
        logger.info("\n" + "─"*50)
        logger.info("⚡ FASE 2/3 — Stages 2+3 liberados (P4+P5 adaptan al dominio)")
        logger.info("─"*50)

        register_backbone(None, phase=3, use_coord_attn=True)
        model = YOLO(f1_weights, task="segment")
        lr_f2, freeze_f2 = apply_phase(model, phase=3, lr=base_lr)

        model.train(
            task          = "segment",
            data          = args.data,
            epochs        = epochs_f2,
            imgsz         = imgsz,
            batch         = batch,
            lr0           = lr_f2,
            lrf           = 0.1,
            weight_decay  = 0.0005,
            warmup_epochs = 1,              # ← Warmup en transición
            freeze        = freeze_f2,       # ← Protege Stages 0+1 + Stem
            optimizer     = "AdamW",
            amp           = True,
            device        = device,
            workers       = workers,
            cache         = cache_opt,
            project       = project,
            name          = "Fast_F2_P4P5",
            seed          = 42,
            verbose       = True,
            save          = True,
            exist_ok      = True,
            **aug_params,
        )

        f2_weights = find_phase_weights(model, project, "Fast_F2_P4P5", "last.pt")

        # ── FASE 3: Full Fine-Tuning (LR reducido, convergencia final) ────
        logger.info("\n" + "─"*50)
        logger.info("⚡ FASE 3/3 — Full Fine-Tuning (convergencia final)")
        logger.info("─"*50)

        register_backbone(None, phase=4, use_coord_attn=True)
        model = YOLO(f2_weights, task="segment")
        lr_f3, freeze_f3 = apply_phase(model, phase=4, lr=base_lr * 0.3)

        results = model.train(
            task          = "segment",
            data          = args.data,
            epochs        = epochs_f3,
            imgsz         = imgsz_final,        # ← Escalado a mayor resolución para Full FT
            batch         = batch,
            lr0           = lr_f3,
            lrf           = 0.05,
            weight_decay  = 0.0005,
            warmup_epochs = 1,              # ← Warmup en transición
            close_mosaic  = close_mosaic,
            freeze        = freeze_f3,       # ← [] en phase 4 (sin freeze)
            mixup         = 0.1,
            copy_paste    = 0.1,
            optimizer     = "AdamW",
            amp           = True,
            device        = device,
            workers       = workers,
            cache         = cache_opt,
            project       = project,
            name          = "Fast_F3_FullFT",
            seed          = 42,
            verbose       = True,
            save          = True,
            exist_ok      = True,
            **aug_params,
        )

        best_model_path = find_phase_weights(model, project, "Fast_F3_FullFT", "best.pt")

    else:
        # ══════════════════════════════════════════════════════════════════
        # MODO ESTÁNDAR (4 Fases)
        #
        #   Fase A (phase=1): Solo adaptadores. Backbone 100% congelado.
        #   Fase B (phase=2): + Stage 3 (P5). Semántica de alto nivel.
        #   Fase C (phase=3): + Stage 2 (P4). Features intermedias.
        #   Fase D (phase=4): Full Fine-Tuning. Convergencia final.
        # ══════════════════════════════════════════════════════════════════
        epochs_A = min(10, args.epochs // 4)
        epochs_B = min(15, args.epochs // 4)
        epochs_C = min(15, args.epochs // 4)
        epochs_D = max(1, args.epochs - epochs_A - epochs_B - epochs_C)

        logger.info(f"📋 Distribución de fases: A={epochs_A} | B={epochs_B} | C={epochs_C} | D={epochs_D}")

        # ── FASE A: Solo adaptadores ──────────────────────────────────────
        logger.info("\n" + "─"*50)
        logger.info(" FASE A — Backbone congelado. Solo SpatialFeatureAdapters.")
        logger.info("─"*50)

        register_backbone(args.encoder, phase=1, use_coord_attn=True)
        model = YOLO(model_yaml, task="segment")

        lr_A, freeze_A = apply_phase(model, phase=1, lr=base_lr)

        model.train(
            task          = "segment",
            data          = args.data,
            epochs        = epochs_A,
            imgsz         = imgsz,
            batch         = batch,
            lr0           = lr_A,
            lrf           = 0.1,
            weight_decay  = 0.0005,
            warmup_epochs = 2,
            freeze        = freeze_A,       # ← CRÍTICO
            optimizer     = "AdamW",
            amp           = True,
            device        = device,
            workers       = workers,
            cache         = cache_opt,
            project       = project,
            name          = "FaseA_Adapters",
            seed          = 42,
            verbose       = True,
            save          = True,
            exist_ok      = True,
            **aug_params,
        )

        fase_a_weights = find_phase_weights(model, project, "FaseA_Adapters", "last.pt")

        # ── FASE B: Descongelar P5 ───────────────────────────────────────
        logger.info("\n" + "─"*50)
        logger.info("🔓 FASE B — Descongelando Stage 3 (P5, semántica global).")
        logger.info("─"*50)

        register_backbone(None, phase=2, use_coord_attn=True)
        model = YOLO(fase_a_weights, task="segment")
        lr_B, freeze_B = apply_phase(model, phase=2, lr=base_lr)

        model.train(
            task          = "segment",
            data          = args.data,
            epochs        = epochs_B,
            imgsz         = imgsz,
            batch         = batch,
            lr0           = lr_B,
            lrf           = 0.1,
            weight_decay  = 0.0005,
            warmup_epochs = 1,
            freeze        = freeze_B,       # ← Protege Stem + Stages 0,1,2
            optimizer     = "AdamW",
            amp           = True,
            device        = device,
            workers       = workers,
            cache         = cache_opt,
            project       = project,
            name          = "FaseB_P5",
            seed          = 42,
            verbose       = True,
            save          = True,
            exist_ok      = True,
            **aug_params,
        )

        fase_b_weights = find_phase_weights(model, project, "FaseB_P5", "last.pt")

        # ── FASE C: Descongelar P4 ───────────────────────────────────────
        logger.info("\n" + "─"*50)
        logger.info(" FASE C — Descongelando Stage 2 (P4, features medias).")
        logger.info("─"*50)

        register_backbone(None, phase=3, use_coord_attn=True)
        model = YOLO(fase_b_weights, task="segment")
        lr_C, freeze_C = apply_phase(model, phase=3, lr=base_lr)

        model.train(
            task          = "segment",
            data          = args.data,
            epochs        = epochs_C,
            imgsz         = imgsz,
            batch         = batch,
            lr0           = lr_C,
            lrf           = 0.1,
            weight_decay  = 0.0005,
            warmup_epochs = 1,
            freeze        = freeze_C,       # ← Protege Stem + Stages 0,1
            mixup         = 0.1,
            copy_paste    = 0.1,
            optimizer     = "AdamW",
            amp           = True,
            device        = device,
            workers       = workers,
            cache         = cache_opt,
            project       = project,
            name          = "FaseC_P4P5",
            seed          = 42,
            verbose       = True,
            save          = True,
            exist_ok      = True,
            **aug_params,
        )

        fase_c_weights = find_phase_weights(model, project, "FaseC_P4P5", "last.pt")

        # ── FASE D: Full Fine-Tuning ──────────────────────────────────────
        logger.info("\n" + "─"*50)
        logger.info(" FASE D — Full Fine-Tuning.")
        logger.info("─"*50)

        register_backbone(None, phase=4, use_coord_attn=True)
        model = YOLO(fase_c_weights, task="segment")
        lr_D, freeze_D = apply_phase(model, phase=4, lr=base_lr * 0.3)

        results = model.train(
            task          = "segment",
            data          = args.data,
            epochs        = epochs_D,
            imgsz         = imgsz_final,        # ← Escalado a mayor resolución para Full FT
            batch         = batch,
            lr0           = lr_D,
            lrf           = 0.05,
            weight_decay  = 0.0005,
            warmup_epochs = 1,
            close_mosaic  = close_mosaic,
            freeze        = freeze_D,       # ← [] en phase 4 (sin freeze)
            mixup         = 0.1,
            copy_paste    = 0.1,
            optimizer     = "AdamW",
            amp           = True,
            device        = device,
            workers       = workers,
            cache         = cache_opt,
            project       = project,
            name          = "FaseD_FullFT",
            seed          = 42,
            verbose       = True,
            save          = True,
            exist_ok      = True,
            **aug_params,
        )

        best_model_path = find_phase_weights(model, project, "FaseD_FullFT", "best.pt")

    # ── Resumen final ─────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info(" ENTRENAMIENTO COMPLETO DE SEGMENTACIÓN")
    logger.info(f"   Mejor modelo → {best_model_path}")
    if hasattr(results, 'results_dict'):
        m = results.results_dict
        logger.info(f"   Box mAP50  : {m.get('metrics/mAP50(B)', 'N/A'):.4f}")
        logger.info(f"   Mask mAP50 : {m.get('metrics/mAP50(M)', 'N/A'):.4f}")
    logger.info("=" * 60)

    return best_model_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Entrenamiento AranduYOLO Segmentación con currículum SSL."
    )
    parser.add_argument("--data",         required=True,  help="Ruta al data.yaml del dataset YOLO (segmentación).")
    parser.add_argument("--encoder",      required=True,  help="Ruta al encoder SSL exportado (.pth).")
    parser.add_argument("--epochs",       type=int, default=100, help="Épocas totales (default: 100).")
    parser.add_argument("--batch",        type=int, default=16,  help="Batch size (default: 16).")
    parser.add_argument("--imgsz",        type=int, default=512, help="Tamaño de imagen base (default: 512).")
    parser.add_argument("--imgsz_final",  type=int, default=640, help="Tamaño de imagen para la fase final Full FT (default: 640, 0=mismo que imgsz).")
    parser.add_argument("--lr",           type=float, default=0.001, help="LR base (default: 0.001).")
    parser.add_argument("--device",       type=str, default="0",  help="GPU(s): '0', '0,1', 'cpu'.")
    parser.add_argument("--project",      type=str, default="AranduYOLO_Seg_runs", help="Directorio de salida.")
    parser.add_argument("--fast",         action="store_true", help="Modo 3 fases acelerado con descongelado gradual.")
    parser.add_argument("--slim",         action="store_true", help="Usar arquitectura Slim-YOLO con 50%% menos canales en el neck (arandu_yolo26_slim_seg.yaml).")
    parser.add_argument("--cfg",          type=str, default="", help="Ruta personalizada a especificación YAML del modelo.")
    parser.add_argument("--cache",        type=str, default="ram", choices=["ram", "disk", "none", "false"], help="Caché de imágenes (default: ram).")
    parser.add_argument("--workers",      type=int, default=8,   help="DataLoader worker threads (default: 8).")
    parser.add_argument("--close_mosaic", type=int, default=15,  help="Desactivar mosaic en últimas N épocas (default: 15).")
    return parser.parse_args()



if __name__ == "__main__":
    args = parse_args()
    best_model = train(args)
    print(f"\n Mejor modelo guardado en: {best_model}")

