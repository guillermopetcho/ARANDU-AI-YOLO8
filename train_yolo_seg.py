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

def apply_phase(model: YOLO, phase: int, lr: float) -> float:
    """Aplica la fase de descongelado sobre el modelo YOLO activo."""
    backbone = None
    for module in model.model.modules():
        if isinstance(module, AranduBackbone):
            backbone = module
            break

    if backbone is None:
        logger.warning(" AranduBackbone no encontrado en el modelo. Ignorando fase.")
        return lr

    backbone.set_training_phase(phase)
    trainable, total = count_trainable(model)

    phase_names = {1: "A — Solo Adaptadores", 2: "B — +Stage3 P5",
                   3: "C — +Stage2 P4",       4: "D — Full Fine-Tuning"}
    lr_factors  = {1: 1.0, 2: 0.5, 3: 0.8, 4: 1.0}

    effective_lr = lr * lr_factors[phase]
    logger.info(f" Fase {phase_names[phase]}")
    logger.info(f"   Parámetros entrenables: {trainable/1e6:.1f}M / {total/1e6:.1f}M total")
    logger.info(f"   LR efectivo: {effective_lr:.2e}")
    return effective_lr


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
    logger.info(f"   Modo         : {'🚀 FAST (2 Fases)' if args.fast else '🐢 STANDARD (4 Fases)'}")
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

    if args.fast:
        # ── MODO ACELERADO (2 Fases) ─────────────────────────────────────────
        epochs_warmup = max(3, min(5, args.epochs // 5))
        epochs_ft     = max(1, args.epochs - epochs_warmup)
        logger.info(f"📋 Distribución FAST: Warmup Adaptadores={epochs_warmup} épocas | Fine-Tuning={epochs_ft} épocas")

        # ── FASE 1: Warmup solo Adaptadores (Backbone congelado) ─────────────
        logger.info("\n" + "─"*50)
        logger.info("⚡ FASE 1/2 — Warmup de Adaptadores (Backbone congelado)")
        logger.info("─"*50)

        register_backbone(args.encoder, phase=1, use_coord_attn=True)
        model = YOLO(model_yaml, task="segment")

        lr_w  = apply_phase(model, phase=1, lr=base_lr)

        model.train(
            task          = "segment",
            data          = args.data,
            epochs        = epochs_warmup,
            imgsz         = imgsz,
            batch         = batch,
            lr0           = lr_w,
            lrf           = 0.1,
            weight_decay  = 0.0005,
            warmup_epochs = 1,
            optimizer     = "AdamW",
            amp           = True,
            device        = device,
            workers       = workers,
            cache         = cache_opt,
            project       = project,
            name          = "Fast_Fase1_Warmup",
            seed          = 42,
            verbose       = True,
            save          = True,
            exist_ok      = True,
        )

        fase1_weights = find_phase_weights(model, project, "Fast_Fase1_Warmup", "last.pt")

        # ── FASE 2: Fine-Tuning de Stages 2+3 (Stages 0+1 SSL congelados) ───
        logger.info("\n" + "─"*50)
        logger.info("⚡ FASE 2/2 — Fine-Tuning Adaptativo (Stages 2+3 liberados)")
        logger.info("─"*50)

        register_backbone(None, phase=4, use_coord_attn=True)
        model = YOLO(fase1_weights, task="segment")
        lr_ft = apply_phase(model, phase=4, lr=base_lr)

        results = model.train(
            task          = "segment",
            data          = args.data,
            epochs        = epochs_ft,
            imgsz         = imgsz,
            batch         = batch,
            lr0           = lr_ft,
            lrf           = 0.05,
            weight_decay  = 0.0005,
            warmup_epochs = 0,
            close_mosaic  = close_mosaic,
            mixup         = 0.1,
            copy_paste    = 0.1,
            optimizer     = "AdamW",
            amp           = True,
            device        = device,
            workers       = workers,
            cache         = cache_opt,
            project       = project,
            name          = "Fast_Fase2_FineTuning",
            seed          = 42,
            verbose       = True,
            save          = True,
            exist_ok      = True,
        )

        best_model_path = find_phase_weights(model, project, "Fast_Fase2_FineTuning", "best.pt")

    else:
        # ── MODO ESTÁNDAR (4 Fases) ──────────────────────────────────────────
        epochs_A = min(10, args.epochs // 4)
        epochs_B = min(15, args.epochs // 4)
        epochs_C = min(15, args.epochs // 4)
        epochs_D = max(1, args.epochs - epochs_A - epochs_B - epochs_C)

        logger.info(f"📋 Distribución de fases: A={epochs_A} | B={epochs_B} | C={epochs_C} | D={epochs_D}")

        # ── FASE A: Solo adaptadores ──────────────────────────────────────────
        logger.info("\n" + "─"*50)
        logger.info(" FASE A — Backbone congelado. Solo SpatialFeatureAdapters.")
        logger.info("─"*50)

        register_backbone(args.encoder, phase=1, use_coord_attn=True)
        model = YOLO(model_yaml, task="segment")

        lr_A  = apply_phase(model, phase=1, lr=base_lr)

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
        )

        fase_a_weights = find_phase_weights(model, project, "FaseA_Adapters", "last.pt")

        # ── FASE B: Descongelar P5 ───────────────────────────────────────────
        logger.info("\n" + "─"*50)
        logger.info("🔓 FASE B — Descongelando Stage 3 (P5, semántica global).")
        logger.info("─"*50)

        register_backbone(None, phase=2, use_coord_attn=True)
        model = YOLO(fase_a_weights, task="segment")
        lr_B  = apply_phase(model, phase=2, lr=base_lr)

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
        )

        fase_b_weights = find_phase_weights(model, project, "FaseB_P5", "last.pt")

        # ── FASE C: Descongelar P4 ───────────────────────────────────────────
        logger.info("\n" + "─"*50)
        logger.info(" FASE C — Descongelando Stage 2 (P4, features medias).")
        logger.info("─"*50)

        register_backbone(None, phase=3, use_coord_attn=True)
        model = YOLO(fase_b_weights, task="segment")
        lr_C  = apply_phase(model, phase=3, lr=base_lr)

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
        )

        fase_c_weights = find_phase_weights(model, project, "FaseC_P4P5", "last.pt")

        # ── FASE D: Full Fine-Tuning ──────────────────────────────────────────
        logger.info("\n" + "─"*50)
        logger.info(" FASE D — Full Fine-Tuning.")
        logger.info("─"*50)

        register_backbone(None, phase=4, use_coord_attn=True)
        model = YOLO(fase_c_weights, task="segment")
        lr_D  = apply_phase(model, phase=4, lr=base_lr * 0.3)

        results = model.train(
            task          = "segment",
            data          = args.data,
            epochs        = epochs_D,
            imgsz         = imgsz,
            batch         = batch,
            lr0           = lr_D,
            lrf           = 0.05,
            weight_decay  = 0.0005,
            warmup_epochs = 0,
            close_mosaic  = close_mosaic,
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
    parser.add_argument("--imgsz",        type=int, default=640, help="Tamaño de imagen (default: 640).")
    parser.add_argument("--lr",           type=float, default=0.001, help="LR base (default: 0.001).")
    parser.add_argument("--device",       type=str, default="0",  help="GPU(s): '0', '0,1', 'cpu'.")
    parser.add_argument("--project",      type=str, default="AranduYOLO_Seg_runs", help="Directorio de salida.")
    parser.add_argument("--fast",         action="store_true", help="Modo 2 fases ultrarrápido con caché en RAM.")
    parser.add_argument("--slim",         action="store_true", help="Usar arquitectura Slim-YOLO con 50%% menos canales en el neck (arandu_yolo26_slim_seg.yaml).")
    parser.add_argument("--cfg",          type=str, default="", help="Ruta personalizada a especificación YAML del modelo.")
    parser.add_argument("--cache",        type=str, default="ram", choices=["ram", "disk", "none", "false"], help="Caché de imágenes (default: ram).")
    parser.add_argument("--workers",      type=int, default=8,   help="DataLoader worker threads (default: 8).")
    parser.add_argument("--close_mosaic", type=int, default=10,  help="Desactivar mosaic en últimas N épocas (default: 10).")
    return parser.parse_args()



if __name__ == "__main__":
    args = parse_args()
    best_model = train(args)
    print(f"\n Mejor modelo guardado en: {best_model}")

