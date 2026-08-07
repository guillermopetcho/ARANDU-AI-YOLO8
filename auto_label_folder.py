"""
auto_label_folder.py — Generador Automático de Anotaciones AnyLabeling usando Modelos YOLO-Seg Entrenados

Toma un modelo entrenado de segmentación (YOLOv8-Seg / YOLO26-Seg) y auto-etiqueta automáticamente
carpetas completas de imágenes, creando los archivos .json compatibles con AnyLabeling.

Uso:
    python auto_label_folder.py --image-dir /ruta/a/imagenes --model /ruta/a/modelo.pt
"""

import os
import glob
import json
import argparse
import logging
from typing import List, Dict, Any
import numpy as np
import cv2
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger("AutoLabeler")

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def mask_to_polygon(mask: np.ndarray, epsilon_ratio: float = 0.010) -> List[List[float]]:
    """Convierte una máscara binaria en un polígono simplificado con pocos puntos clave."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    if not contours:
        return []

    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 100:
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
    return {
        "version": "0.2.10",
        "flags": {},
        "shapes": shapes,
        "imagePath": os.path.basename(image_name),
        "imageData": None,
        "imageHeight": int(img_h),
        "imageWidth": int(img_w)
    }


def auto_label_folder(
    image_dir: str,
    model_path: str,
    conf_threshold: float = 0.25,
    overwrite: bool = False
):
    logger.info(f"Cargando modelo de segmentación: {model_path}")
    model = YOLO(model_path)

    all_files = glob.glob(os.path.join(image_dir, "*"))
    img_paths = [f for f in all_files if os.path.splitext(f)[1].lower() in IMG_EXTS]

    logger.info(f"Encontradas {len(img_paths)} imágenes en: {image_dir}")

    for idx, img_path in enumerate(img_paths, 1):
        json_path = os.path.splitext(img_path)[0] + ".json"
        if os.path.exists(json_path) and not overwrite:
            logger.info(f"[{idx}/{len(img_paths)}] Omitiendo (JSON ya existe): {os.path.basename(img_path)}")
            continue

        logger.info(f"[{idx}/{len(img_paths)}] Auto-segmentando: {os.path.basename(img_path)}")
        results = model.predict(source=img_path, conf=conf_threshold, verbose=False)

        if not results or not len(results[0]):
            logger.warning(f"  Sin detecciones en {os.path.basename(img_path)}")
            continue

        result = results[0]
        img_h, img_w = result.orig_shape
        shapes = []

        if result.masks is not None:
            masks_data = result.masks.data.cpu().numpy()
            boxes_data = result.boxes

            for i, mask in enumerate(masks_data):
                cls_id = int(boxes_data.cls[i].item())
                cls_name = result.names.get(cls_id, f"clase_{cls_id}")

                # Redimensionar máscara al tamaño original de la imagen
                mask_resized = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
                mask_binary = (mask_resized > 0.5).astype(np.uint8)

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

        logger.info(f"  ✓ Creado AnyLabeling JSON con {len(shapes)} objetos: {os.path.basename(json_path)}")

    logger.info("🎉 Auto-etiquetado completado correctamente.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto-Labeler de carpetas estilo AnyLabeling")
    parser.add_argument("--image-dir", type=str, required=True, help="Carpeta de imágenes a segmentar")
    parser.add_argument("--model", type=str, default="yolov8n-seg.pt", help="Ruta al modelo YOLO-Seg (.pt / .onnx)")
    parser.add_argument("--conf", type=float, default=0.25, help="Umbral de confianza")
    parser.add_argument("--overwrite", action="store_true", help="Sobrescribir JSONs existentes")

    args = parser.parse_args()
    auto_label_folder(args.image_dir, args.model, args.conf, args.overwrite)
