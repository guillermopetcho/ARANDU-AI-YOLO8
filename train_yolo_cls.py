"""
train_yolo_cls.py — Entrenamiento de Clasificación YOLO con AranduBackbone SSL

Entrena YOLOv8 en modo de clasificación pura usando carpetas de imágenes.
"""

import argparse
import os
import sys
import logging
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

class AranduYOLOClsWrapper(AranduBackbone):
    def __init__(self, *args, **kwargs):
        global _ENCODER_PATH
        super().__init__(
            moco_checkpoint_path=_ENCODER_PATH,
            freeze_phase=3,
            use_coord_attn=False
        )

    def forward(self, x):
        features = super().forward(x)
        return features[-1] 

class AranduClassify(Classify):
    def __init__(self, c2, *args, **kwargs):
        super().__init__(1024, c2, *args, **kwargs)

def register_cls_backbone(encoder_path: str):
    """
    Registra AranduYOLOClsWrapper en Ultralytics de forma dinámica.
    Las clases se definen de manera global para que 'torch.save' pueda serializarlas (pickle).
    """
    global _ENCODER_PATH
    _ENCODER_PATH = encoder_path

    setattr(nn_modules, 'AranduYOLOClsWrapper', AranduYOLOClsWrapper)
    setattr(sys.modules['ultralytics.nn.modules'], 'AranduYOLOClsWrapper', AranduYOLOClsWrapper)
    setattr(nn_tasks, 'AranduYOLOClsWrapper', AranduYOLOClsWrapper)
    setattr(sys.modules['ultralytics.nn.tasks'], 'AranduYOLOClsWrapper', AranduYOLOClsWrapper)

    setattr(nn_modules, 'AranduClassify', AranduClassify)
    setattr(sys.modules['ultralytics.nn.modules'], 'AranduClassify', AranduClassify)
    setattr(nn_tasks, 'AranduClassify', AranduClassify)
    setattr(sys.modules['ultralytics.nn.tasks'], 'AranduClassify', AranduClassify)
    
    logger.info("✅ AranduYOLOClsWrapper y AranduClassify registrados para CLASIFICACIÓN.")

# ---------------------------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------------------------
def train(args):
    if not os.path.isdir(args.data):
        raise FileNotFoundError(f"Carpeta de dataset no encontrada: {args.data}")
    if not os.path.isfile(args.encoder):
        raise FileNotFoundError(f"Encoder SSL no encontrado: {args.encoder}")

    logger.info("=" * 60)
    logger.info("🌱 ARANDU-AI YOLO — CLASIFICACIÓN SSL")
    logger.info(f"   Dataset (Carpetas): {args.data}")
    logger.info(f"   Encoder SSL       : {args.encoder}")
    logger.info("=" * 60)

    # Registramos el modelo en Ultralytics
    register_cls_backbone(args.encoder)
    
    # Cargamos la arquitectura que definimos en yaml y forzamos la tarea
    model = YOLO("arandu_yolo_cls.yaml", task="classify")

    # Entrenamos
    results = model.train(
        data      = args.data,       # Ruta a la carpeta que contiene train/ y val/
        epochs    = args.epochs,
        imgsz     = args.imgsz,
        batch     = args.batch,
        lr0       = args.lr,         # LR más bajo porque el encoder ya sabe mucho
        lrf       = 0.1,
        optimizer = "AdamW",
        amp       = True,
        project   = args.project,
        name      = "Arandu_Clasificacion",
        seed      = 42,
        exist_ok  = True
    )

    best_model = os.path.join(args.project, "Arandu_Clasificacion", "weights", "best.pt")
    logger.info(f"\n✅ Clasificación completada. Mejor modelo en: {best_model}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",    required=True, help="Carpeta raíz del dataset (debe contener train/ y val/).")
    parser.add_argument("--encoder", required=True, help="Ruta al moco_encoder_ready.pth.")
    parser.add_argument("--epochs",  type=int, default=30, help="Épocas de fine-tuning.")
    parser.add_argument("--batch",   type=int, default=32)
    parser.add_argument("--imgsz",   type=int, default=512)
    parser.add_argument("--lr",      type=float, default=0.0001)
    parser.add_argument("--project", type=str, default="AranduYOLO_runs")
    args = parser.parse_args()
    train(args)
