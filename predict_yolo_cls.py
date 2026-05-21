"""
predict_yolo_cls.py — Script de Inferencia en Lote y Exportación a CSV

Carga un modelo YOLO de clasificación entrenado con AranduBackbone,
procesa una carpeta de imágenes sin etiquetas, y guarda los resultados en un CSV.
"""

import argparse
import os
import sys
import csv
import logging
from ultralytics import YOLO
import ultralytics.nn.modules as nn_modules
import ultralytics.nn.tasks as nn_tasks
from ultralytics.nn.modules.head import Classify

from models.yolo_wrapper import AranduBackbone

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("YOLO-Inference")

_ENCODER_PATH = None

class AranduYOLOClsWrapper(AranduBackbone):
    def __init__(self, *args, **kwargs):
        global _ENCODER_PATH
        super().__init__(
            moco_checkpoint_path=_ENCODER_PATH,
            freeze_phase=4,  # Actualizado para coincidir con el entrenamiento
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
    Re-registra la clase del backbone de manera global.
    Es OBLIGATORIO hacerlo antes de cargar 'best.pt' porque Ultralytics
    necesita saber qué forma tiene la red para poder colocar los pesos.
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


def main(args):
    # 1. Registrar la arquitectura personalizada
    register_cls_backbone(args.encoder)
    
    # 2. Cargar el modelo entrenado y forzar la tarea
    logger.info(f"Cargando modelo YOLO: {args.weights}")
    model = YOLO(args.weights, task="classify")

    if args.mode == "val":
        logger.info(f"Ejecutando validación oficial de Ultralytics sobre: {args.source}")
        metrics = model.val(data=args.source, imgsz=args.imgsz)
        logger.info(f"🚀 Precisión Top-1 real medida: {metrics.top1 * 100:.2f}%")
        return

    logger.info(f"Iniciando predicciones sobre la carpeta: {args.source}")
    
    # 3. Lanzar inferencia. stream=True optimiza la memoria para miles de imágenes
    results = model.predict(source=args.source, imgsz=args.imgsz, stream=True)

    csv_path = args.output
    count = 0
    
    # 4. Procesar y guardar en CSV
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Las clases exactas detectadas automáticamente del modelo
        clases_esperadas = list(model.names.values())
        
        # Escribir encabezados exactamente como en la imagen
        encabezados = ['Nombre de imagen', 'Clase'] + clases_esperadas
        writer.writerow(encabezados)

        for r in results:
            # Nombre de la imagen analizada
            filename = os.path.basename(r.path)
            
            # Clase elegida (la de mayor probabilidad)
            top_class_id = r.probs.top1
            top_class_name = r.names[top_class_id]
            
            # Extraer las probabilidades de TODAS las clases y pasarlas a %
            # r.probs.data tiene el tensor con las probs, r.names mapea el índice al nombre
            probs_dict = {r.names[i]: float(p) * 100 for i, p in enumerate(r.probs.data)}
            
            # Armar la fila iterando sobre las clases esperadas
            fila = [filename, top_class_name]
            for cls_name in clases_esperadas:
                prob_val = probs_dict.get(cls_name, 0.0)
                fila.append(f"{prob_val:.2f}%")
            
            # Escribir fila en CSV
            writer.writerow(fila)
            count += 1
            
    logger.info(f"✅ ¡Completado! Se han analizado {count} imágenes.")
    logger.info(f"✅ Resultados guardados en: {csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="Ruta al modelo entrenado (best.pt)")
    parser.add_argument("--source",  required=True, help="Ruta a la carpeta con las imágenes sin clasificar")
    parser.add_argument("--encoder", required=True, help="Ruta original al moco_encoder_ready.pth (necesario para armar la red)")
    parser.add_argument("--output",  default="resultados_clasificacion.csv", help="Nombre del archivo CSV generado")
    parser.add_argument("--imgsz",   type=int, default=224, help="Debe coincidir con el tamaño de entrenamiento (224).")
    parser.add_argument("--mode",    choices=["predict", "val"], default="predict", help="'predict' para CSV, 'val' para evaluar métricas.")
    
    args = parser.parse_args()
    main(args)
