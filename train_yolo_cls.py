"""
train_yolo_cls.py — Entrenamiento de Clasificación YOLO con AranduBackbone SSL

Entrena YOLO26 en modo de clasificación pura usando carpetas de imágenes.
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
import ultralytics.nn.modules as nn_modules
import ultralytics.nn.tasks as nn_tasks
from ultralytics import YOLO
from ultralytics.nn.modules.head import Classify

from models.yolo_wrapper import AranduBackbone

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("AranduYOLOCls")

# ---------------------------------------------------------------------------
# Wrapper de Clasificación
# ---------------------------------------------------------------------------
_ENCODER_PATH = None
_FREEZE_PHASE = 3

class AranduYOLOClsWrapper(AranduBackbone):
    def __init__(self, *args, **kwargs):
        super().__init__(
            moco_checkpoint_path=_ENCODER_PATH,
            freeze_phase=_FREEZE_PHASE,
            use_coord_attn=False
        )

    def forward(self, x):
        features = super().forward(x)
        return features[-1] 

class AranduClassify(Classify):
    def __init__(self, c2, *args, **kwargs):
        super().__init__(768, c2, *args, **kwargs)

def register_cls_backbone(encoder_path: str, freeze_phase: int = 3):
    """
    Registra AranduYOLOClsWrapper en Ultralytics de forma dinámica.
    Las clases se definen de manera global para que 'torch.save' pueda serializarlas (pickle).
    """
    global _ENCODER_PATH, _FREEZE_PHASE
    _ENCODER_PATH = encoder_path
    _FREEZE_PHASE = freeze_phase

    setattr(nn_modules, 'AranduYOLOClsWrapper', AranduYOLOClsWrapper)
    setattr(sys.modules['ultralytics.nn.modules'], 'AranduYOLOClsWrapper', AranduYOLOClsWrapper)
    setattr(nn_tasks, 'AranduYOLOClsWrapper', AranduYOLOClsWrapper)
    setattr(sys.modules['ultralytics.nn.tasks'], 'AranduYOLOClsWrapper', AranduYOLOClsWrapper)

    setattr(nn_modules, 'AranduClassify', AranduClassify)
    setattr(sys.modules['ultralytics.nn.modules'], 'AranduClassify', AranduClassify)
    setattr(nn_tasks, 'AranduClassify', AranduClassify)
    setattr(sys.modules['ultralytics.nn.tasks'], 'AranduClassify', AranduClassify)
    
    logger.info(" AranduYOLOClsWrapper y AranduClassify registrados para CLASIFICACIÓN.")

# ---------------------------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------------------------
def train(args):
    if not os.path.isdir(args.data):
        raise FileNotFoundError(f"Carpeta de dataset no encontrada: {args.data}")
    if not os.path.isfile(args.encoder):
        raise FileNotFoundError(f"Encoder SSL no encontrado: {args.encoder}")

    logger.info("=" * 60)
    logger.info("🌱 ARANDU-AI YOLO — CLASIFICACIÓN SSL 🌱")
    logger.info(f"   Dataset (Carpetas): {args.data}")
    logger.info(f"   Encoder SSL       : {args.encoder}")
    logger.info(f"   Estrategia        : {args.stage.upper()}")
    logger.info("=" * 60)

    # Configuración inteligente según la etapa
    if args.stage == "lp":
        freeze_phase = 3
        default_lr = 0.01
        opt_to_use = "SGD"
        run_name = "Arandu_Clasificacion_LP"
        model_init = "arandu_yolo_cls.yaml"
        logger.info("▶ FASE 1: LINEAR PROBING. Backbone congelado. Entrenando solo el cabezal...")
    else:
        freeze_phase = 4
        default_lr = 0.0001
        opt_to_use = "AdamW"
        run_name = "Arandu_Clasificacion_FT"
        if not args.weights:
            raise ValueError("Para Full Fine-Tuning (--stage ft) debes proveer --weights apuntando al best.pt de la fase LP.")
        model_init = args.weights
        logger.info("▶ FASE 2: FULL FINE-TUNING. Todo descongelado. Ajuste fino de pesos...")

    lr_to_use = args.lr if args.lr is not None else default_lr

    # Registramos el modelo en Ultralytics
    register_cls_backbone(args.encoder, freeze_phase)
    
    # Cargamos la arquitectura o los pesos según la fase
    model = YOLO(model_init, task="classify")

        # Entrenamos
    model.train(
        data      = args.data,
        epochs    = args.epochs,
        imgsz     = args.imgsz,
        batch     = args.batch,
        lr0       = lr_to_use,       # LR automático o manual
        lrf       = 0.01,
        optimizer = opt_to_use,      # SGD para LP, AdamW para FT
        amp       = True,
        patience  = 20,
        project   = args.project,
        name      = run_name,
        seed      = 42,
        exist_ok  = True
    )

    best_model = os.path.join(args.project, run_name, "weights", "best.pt")
    logger.info(f"\n Etapa {args.stage.upper()} completada. Mejor modelo en: {best_model}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",    required=True, help="Carpeta raíz del dataset (debe contener train/ y val/).")
    parser.add_argument("--encoder", required=True, help="Ruta al moco_encoder_ready.pth.")
    parser.add_argument("--stage",   type=str, choices=["lp", "ft"], default="lp", help="Etapa: 'lp' (Linear Probing) o 'ft' (Fine-Tuning)")
    parser.add_argument("--weights", type=str, default="", help="Ruta al best.pt de la fase LP (requerido para stage 'ft').")
    parser.add_argument("--epochs",  type=int, default=100, help="Épocas de fine-tuning.")
    parser.add_argument("--batch",   type=int, default=32)
    parser.add_argument("--imgsz",   type=int, default=224, help="224 es el estándar nativo para MoCo y ConvNeXt.")
    parser.add_argument("--lr",      type=float, default=None, help="Si se omite, usa 0.01 en 'lp' y 0.0001 en 'ft'.")
    parser.add_argument("--project", type=str, default="AranduYOLO_runs")
    args = parser.parse_args()
    train(args)
