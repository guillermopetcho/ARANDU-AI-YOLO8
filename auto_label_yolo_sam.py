"""
auto_label_yolo_sam.py — Auto-etiquetado de Segmentación Guiado por YOLO Detector + SAM

Este script utiliza un modelo YOLO de detección (para localizar objetos/lesiones)
y un modelo SAM / MobileSAM (para segmentar con precisión a nivel de píxel las cajas detectadas).
Genera automáticamente archivos .json compatibles con AnyLabeling.

Uso:
    PYTHONPATH=ultralytics python3 auto_label_yolo_sam.py \
        --image-dir /ruta/a/imagenes \
        --detector /ruta/a/yolo_detector.pt \
        --sam sam_b.pt \
        --conf 0.25
"""

import os
import sys
import glob
import json
import argparse
import logging
from typing import List, Dict, Any
import numpy as np
import cv2

# Asegurar importación de ultralytics local si está disponible
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ultralytics"))

try:
    from ultralytics import YOLO, SAM
except ImportError:
    from ultralytics import YOLO
    from ultralytics.models.sam import SAM

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("YoloSamAutoLabeler")

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def mask_to_polygon(mask: np.ndarray, epsilon_ratio: float = 0.010, min_area: float = 50.0) -> List[List[float]]:
    """Convierte una máscara binaria (H, W) en un polígono simplificado."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    if not contours:
        return []

    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < min_area:
        return []

    eps = epsilon_ratio * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, eps, True)

    if len(approx) < 3:
        return []

    return [[float(round(pt[0][0], 2)), float(round(pt[0][1], 2))] for pt in approx]


def create_anylabeling_json(
    image_name: str,
    img_w: int,
    img_h: int,
    shapes: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Crea la estructura de diccionario compatible con AnyLabeling v0.2.10."""
    return {
        "version": "0.2.10",
        "flags": {},
        "shapes": shapes,
        "imagePath": os.path.basename(image_name),
        "imageData": None,
        "imageHeight": int(img_h),
        "imageWidth": int(img_w)
    }


def auto_label_yolo_sam(
    image_dir: str,
    detector_path: str,
    sam_path: str = "sam_b.pt",
    conf_threshold: float = 0.25,
    overwrite: bool = False
):
    """
    Pipeline principal:
    1. YOLO Detector encuentra bounding boxes [x1, y1, x2, y2].
    2. SAM toma las cajas como 'Box Prompts' y calcula las máscaras de alta precisión.
    3. Se extraen los polígonos y se guardan en formato AnyLabeling .json.
    """
    logger.info(f"Cargando detector YOLO desde: {detector_path}")
    detector = YOLO(detector_path)

    logger.info(f"Cargando segmentador SAM desde: {sam_path}")
    sam = SAM(sam_path)

    all_files = glob.glob(os.path.join(image_dir, "*"))
    img_paths = [f for f in all_files if os.path.splitext(f)[1].lower() in IMG_EXTS]

    if not img_paths:
        logger.warning(f"No se encontraron imágenes en: {image_dir}")
        return

    logger.info(f"Encontradas {len(img_paths)} imágenes en: {image_dir}")

    for idx, img_path in enumerate(img_paths, 1):
        json_path = os.path.splitext(img_path)[0] + ".json"
        if os.path.exists(json_path) and not overwrite:
            logger.info(f"[{idx}/{len(img_paths)}] Omitiendo (JSON existe): {os.path.basename(img_path)}")
            continue

        logger.info(f"[{idx}/{len(img_paths)}] Procesando: {os.path.basename(img_path)}")

        # 1. Detección con YOLO
        det_results = detector.predict(source=img_path, conf=conf_threshold, verbose=False)
        if not det_results or len(det_results[0].boxes) == 0:
            logger.warning(f"  Sin detecciones en {os.path.basename(img_path)}")
            continue

        det_result = det_results[0]
        img_h, img_w = det_result.orig_shape
        boxes = det_result.boxes.xyxy.cpu().numpy()  # [N, 4] -> x1, y1, x2, y2
        cls_ids = det_result.boxes.cls.cpu().numpy().astype(int)

        # 2. Segmentación con SAM usando las cajas delimitadoras como Prompts
        logger.info(f"  Enviando {len(boxes)} cajas a SAM para segmentación fina...")
        sam_results = sam.predict(source=img_path, bboxes=boxes, verbose=False)

        shapes = []
        if sam_results and sam_results[0].masks is not None:
            masks_data = sam_results[0].masks.data.cpu().numpy()

            for i, mask in enumerate(masks_data):
                cls_id = cls_ids[i]
                cls_name = det_result.names.get(cls_id, f"clase_{cls_id}")

                # Asegurar dimensiones originales
                if mask.shape[:2] != (img_h, img_w):
                    mask = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_LINEAR)

                mask_binary = (mask > 0.5).astype(np.uint8)
                polygon_pts = mask_to_polygon(mask_binary)

                if len(polygon_pts) >= 3:
                    shape = {
                        "label": cls_name,
                        "text": "",
                        "points": polygon_pts,
                        "group_id": None,
                        "shape_type": "polygon",
                        "flags": {},
                        "description": ""
                    }
                    shapes.append(shape)

        json_data = create_anylabeling_json(img_path, img_w, img_h, shapes)

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)

        logger.info(f"  ✓ Guardado AnyLabeling JSON ({len(shapes)} polígonos): {os.path.basename(json_path)}")

    logger.info("🎉 Proceso completado exitosamente.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto-Labeler YOLO Detector + SAM Segmentador para AnyLabeling")
    parser.add_argument("--image-dir", type=str, required=True, help="Carpeta con imágenes a etiquetar")
    parser.add_argument("--detector", type=str, required=True, help="Ruta al modelo YOLO detector (.pt / .onnx)")
    parser.add_argument("--sam", type=str, default="sam_b.pt", help="Ruta o nombre del modelo SAM (.pt / .onnx), ej: sam_b.pt, mobile_sam.pt")
    parser.add_argument("--conf", type=float, default=0.25, help="Umbral de confianza de detección")
    parser.add_argument("--overwrite", action="store_true", help="Sobrescribir JSONs existentes")

    args = parser.parse_args()
    auto_label_yolo_sam(args.image_dir, args.detector, args.sam, args.conf, args.overwrite)
