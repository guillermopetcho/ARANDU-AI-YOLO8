"""
train_yolo.py — Entrenamiento YOLO con AranduBackbone SSL (MoCo v3)

Currículum de 4 Fases:
  Fase A: Solo adaptadores entrenan. Backbone 100% congelado. (10 épocas)
  Fase B: + Stage 3 (P5, semántica global). LR backbone reducido. (15 épocas)
  Fase C: + Stage 2 (P4, features medias). LR backbone mayor. (15 épocas)
  Fase D: Full fine-tuning. Todo el modelo. (epochs restantes)

Uso:
    python train_yolo.py --data /ruta/data.yaml --encoder /ruta/moco_encoder_ready.pth

Ejemplo Kaggle:
    python train_yolo.py \\
        --data /kaggle/input/mi-dataset/data.yaml \\
        --encoder /kaggle/input/mi-encoder/moco_encoder_ready.pth \\
        --epochs 100 --batch 16 --imgsz 640
"""

import argparse
import os
import sys
import logging

import torch
import ultralytics.nn.modules as nn_modules
from ultralytics import YOLO

from models.yolo_wrapper import AranduBackbone

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("AranduYOLO")


# ---------------------------------------------------------------------------
# Registro del wrapper en Ultralytics
# ---------------------------------------------------------------------------

def register_backbone(encoder_path: str, phase: int = 1, use_coord_attn: bool = True):
    """Registra AranduYOLOWrapper en el espacio de nombres de Ultralytics.

    Ultralytics instancia el backbone por nombre de clase desde el YAML.
    Hay que inyectar la clase ANTES de llamar a YOLO("arandu_yolo26.yaml").
    """
    class AranduYOLOWrapper(AranduBackbone):
        def __init__(self, *args, **kwargs):
            kwargs['moco_checkpoint_path'] = encoder_path
            kwargs['freeze_phase']         = phase
            kwargs['use_coord_attn']       = use_coord_attn
            super().__init__(*args, **kwargs)

    setattr(nn_modules, 'AranduYOLOWrapper', AranduYOLOWrapper)
    setattr(sys.modules['ultralytics.nn.modules'], 'AranduYOLOWrapper', AranduYOLOWrapper)
    
    # FIX: Inject into tasks for globals() access during parse_model
    import ultralytics.nn.tasks
    setattr(sys.modules['ultralytics.nn.tasks'], 'AranduYOLOWrapper', AranduYOLOWrapper)
    logger.info(f" AranduYOLOWrapper registrado — fase={phase}, coord_attn={use_coord_attn}")
    return AranduYOLOWrapper


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
    """Aplica la fase de descongelado sobre el modelo YOLO activo.

    Retorna el LR efectivo recomendado para esa fase.
    """
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
    logger.info("🌱 ARANDU-AI YOLO — Entrenamiento con Currículum SSL 🌱")
    logger.info(f"   Dataset  : {args.data}")
    logger.info(f"   Encoder  : {args.encoder}")
    logger.info(f"   Épocas   : {args.epochs} | Batch: {args.batch} | Imgsz: {args.imgsz}")
    logger.info("=" * 60)

    # ── Distribución de épocas por fase ──────────────────────────────────
    # Fase A: solo adaptadores          → backbone congelado
    # Fase B: backbone parcial (P5)     → semántica global
    # Fase C: backbone parcial (P4+P5)  → features medias
    # Fase D: full fine-tuning          → todo el modelo
    epochs_A = min(10, args.epochs // 4)
    epochs_B = min(15, args.epochs // 4)
    epochs_C = min(15, args.epochs // 4)
    epochs_D = max(1, args.epochs - epochs_A - epochs_B - epochs_C)

    logger.info(f"📋 Distribución de fases: A={epochs_A} | B={epochs_B} | C={epochs_C} | D={epochs_D}")

    project  = args.project
    base_lr  = args.lr
    imgsz    = args.imgsz
    batch    = args.batch
    device   = args.device

    # ── FASE A: Solo adaptadores ──────────────────────────────────────────
    logger.info("\n" + "─"*50)
    logger.info(" FASE A — Backbone congelado. Solo SpatialFeatureAdapters.")
    logger.info("─"*50)

    register_backbone(args.encoder, phase=1, use_coord_attn=True)
    model = YOLO("arandu_yolo26.yaml")
    lr_A  = apply_phase(model, phase=1, lr=base_lr)

    model.train(
        data      = args.data,
        epochs    = epochs_A,
        imgsz     = imgsz,
        batch     = batch,
        lr0       = lr_A,
        lrf       = 0.1,
        weight_decay = 0.0005,
        warmup_epochs = 2,
        optimizer = "AdamW",
        amp       = True,
        device    = device,
        project   = project,
        name      = "FaseA_Adapters",
        seed      = 42,
        verbose   = True,
        save      = True,
        exist_ok  = True,
    )

    fase_a_weights = os.path.join(project, "FaseA_Adapters", "weights", "last.pt")
    logger.info(f" Fase A completada → {fase_a_weights}")

    # ── FASE B: Descongelar P5 (Stage 3) ─────────────────────────────────
    logger.info("\n" + "─"*50)
    logger.info("🔓 FASE B — Descongelando Stage 3 (P5, semántica global).")
    logger.info("─"*50)

    register_backbone(args.encoder, phase=2, use_coord_attn=True)
    model = YOLO(fase_a_weights)
    lr_B  = apply_phase(model, phase=2, lr=base_lr)

    model.train(
        data      = args.data,
        epochs    = epochs_B,
        imgsz     = imgsz,
        batch     = batch,
        lr0       = lr_B,
        lrf       = 0.1,
        weight_decay = 0.0005,
        warmup_epochs = 1,
        optimizer = "AdamW",
        amp       = True,
        device    = device,
        project   = project,
        name      = "FaseB_P5",
        seed      = 42,
        verbose   = True,
        save      = True,
        exist_ok  = True,
    )

    fase_b_weights = os.path.join(project, "FaseB_P5", "weights", "last.pt")
    logger.info(f" Fase B completada → {fase_b_weights}")

    # ── FASE C: Descongelar P4 (Stage 2) ─────────────────────────────────
    logger.info("\n" + "─"*50)
    logger.info(" FASE C — Descongelando Stage 2 (P4, features medias).")
    logger.info("─"*50)

    register_backbone(args.encoder, phase=3, use_coord_attn=True)
    model = YOLO(fase_b_weights)
    lr_C  = apply_phase(model, phase=3, lr=base_lr)

    model.train(
        data      = args.data,
        epochs    = epochs_C,
        imgsz     = imgsz,
        batch     = batch,
        lr0       = lr_C,
        lrf       = 0.1,
        weight_decay = 0.0005,
        warmup_epochs = 1,
        optimizer = "AdamW",
        amp       = True,
        device    = device,
        project   = project,
        name      = "FaseC_P4P5",
        seed      = 42,
        verbose   = True,
        save      = True,
        exist_ok  = True,
    )

    fase_c_weights = os.path.join(project, "FaseC_P4P5", "weights", "last.pt")
    logger.info(f" Fase C completada → {fase_c_weights}")

    # ── FASE D: Full Fine-Tuning ──────────────────────────────────────────
    logger.info("\n" + "─"*50)
    logger.info(" FASE D — Full Fine-Tuning. Todo el modelo entrena.")
    logger.info("─"*50)

    register_backbone(args.encoder, phase=4, use_coord_attn=True)
    model = YOLO(fase_c_weights)
    lr_D  = apply_phase(model, phase=4, lr=base_lr * 0.3)  # LR reducido al 30% en full FT

    results = model.train(
        data      = args.data,
        epochs    = epochs_D,
        imgsz     = imgsz,
        batch     = batch,
        lr0       = lr_D,
        lrf       = 0.05,
        weight_decay = 0.0005,
        warmup_epochs = 0,
        optimizer = "AdamW",
        amp       = True,
        device    = device,
        project   = project,
        name      = "FaseD_FullFT",
        seed      = 42,
        verbose   = True,
        save      = True,
        exist_ok  = True,
    )

    fase_d_best = os.path.join(project, "FaseD_FullFT", "weights", "best.pt")

    # ── Resumen final ─────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info(" ENTRENAMIENTO COMPLETO")
    logger.info(f"   Mejor modelo → {fase_d_best}")
    if hasattr(results, 'results_dict'):
        m = results.results_dict
        logger.info(f"   mAP50      : {m.get('metrics/mAP50(B)', 'N/A'):.4f}")
        logger.info(f"   mAP50-95   : {m.get('metrics/mAP50-95(B)', 'N/A'):.4f}")
        logger.info(f"   Precision  : {m.get('metrics/precision(B)', 'N/A'):.4f}")
        logger.info(f"   Recall     : {m.get('metrics/recall(B)', 'N/A'):.4f}")
    logger.info("=" * 60)

    return fase_d_best


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Entrenamiento AranduYOLO con currículum SSL de 4 fases."
    )
    parser.add_argument("--data",    required=True,  help="Ruta al data.yaml del dataset YOLO.")
    parser.add_argument("--encoder", required=True,  help="Ruta al encoder SSL exportado (.pth).")
    parser.add_argument("--epochs",  type=int, default=100, help="Épocas totales (default: 100).")
    parser.add_argument("--batch",   type=int, default=16,  help="Batch size (default: 16).")
    parser.add_argument("--imgsz",   type=int, default=640, help="Tamaño de imagen (default: 640).")
    parser.add_argument("--lr",      type=float, default=0.001, help="LR base (default: 0.001).")
    parser.add_argument("--device",  type=str, default="0",  help="GPU(s): '0', '0,1', 'cpu'.")
    parser.add_argument("--project", type=str, default="AranduYOLO_runs", help="Directorio de salida.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    best_model = train(args)
    print(f"\n Mejor modelo guardado en: {best_model}")
