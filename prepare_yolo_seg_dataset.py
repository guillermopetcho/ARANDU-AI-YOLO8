"""
prepare_yolo_seg_dataset.py — Consolidador de Datasets YOLO Segmentación divididos por clase.

Toma un directorio raíz (ej. DATASET-YOLO-SEGM) que contiene subcarpetas por clase:
  - Soybean-Bacterial-Blight-SEGM
  - Soybean-Frogeye-Leaf-Spot-SEGM
  - Soybean-Healty-SEGM
  - Soybean-Mosaic-Virus-SEGM
  - Soybean-Potassium-Deficiency-SEGM
  - (Soporta dinámicamente cualquier subcarpeta adicional)

Re-indexa los IDs de clase en las anotaciones .txt (de 0 a su ID unificado 0..N-1)
y consolida una estructura estándar de YOLO en /kaggle/working/dataset_seg_unified:
  images/train, labels/train, images/val, labels/val
y genera el data.yaml correspondiente.

Uso:
    python prepare_yolo_seg_dataset.py \
      --src /kaggle/input/datasets/joaquinignaciopetcho/dataset-yolo-segm/DATASET-YOLO-SEGM \
      --dest /kaggle/working/dataset_seg_unified
"""

import argparse
import os
import shutil
from pathlib import Path
import yaml
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger("DatasetPreparer")

# Mapeo de nombres conocidos / alias a nombres limpios de clase
KNOWN_ALIASES = {
    "soybean-bacterial-blight": "bacterial_blight",
    "soybean-bacterial-blight-segm": "bacterial_blight",
    "bacterial-blight": "bacterial_blight",
    "soybean-frogeye-leaf-spot": "frog_eye",
    "soybean-frogeye-leaf-spot-segm": "frog_eye",
    "soybean-frogeye": "frog_eye",
    "frogeye": "frog_eye",
    "soybean-healty": "healthy",
    "soybean-healty-segm": "healthy",
    "soybean-healthy": "healthy",
    "soybean-healthy-segm": "healthy",
    "healthy": "healthy",
    "soybean-mosaic-virus": "mosaic",
    "soybean-mosaic-virus-segm": "mosaic",
    "mosaic": "mosaic",
    "soybean-potassium-deficiency": "potassium_deficiency",
    "soybean-potassium-deficiency-segm": "potassium_deficiency",
    "potassium-deficiency": "potassium_deficiency",
    "soybean-septoria-leaf-spot": "septoria_leaf_spot",
    "soybean-brown-spot": "brown_spot",
    "soybean-target-spot": "target_spot",
    "soybean-rust": "rust",
    "soybean-cercospora-leaf-blight": "cercospora_leaf_blight",
    "soybean-sudden-death-syndrome": "sudden_death_syndrome",
    "insecto": "insect_damage",
    "soybean-damege": "soybean_damage",
    "phyllosticta": "phyllosticta",
}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def clean_class_name(folder_name: str) -> str:
    """Convierte el nombre de carpeta a un identificador de clase limpio."""
    key = folder_name.lower().strip()
    if key in KNOWN_ALIASES:
        return KNOWN_ALIASES[key]
    
    # Remover sufijo '-SEGM' o '-segm' si existe
    if key.endswith("-segm"):
        key = key[:-5]
    if key.startswith("soybean-"):
        key = key[8:]
    return key.replace("-", "_").replace(" ", "_")


def process_folder(src_split_dir, dest_img_dir, dest_lbl_dir, target_class_id):
    """Procesa una carpeta de split (train o val) de una clase específica."""
    if not os.path.exists(src_split_dir):
        return 0

    os.makedirs(dest_img_dir, exist_ok=True)
    os.makedirs(dest_lbl_dir, exist_ok=True)

    count = 0
    all_files = list(Path(src_split_dir).rglob("*"))
    img_files = [f for f in all_files if f.suffix.lower() in IMG_EXTS]

    for img_path in img_files:
        stem = img_path.stem
        lbl_path = img_path.parent / f"{stem}.txt"
        if not lbl_path.exists():
            alt_lbl = img_path.parent.parent / "labels" / f"{stem}.txt"
            if alt_lbl.exists():
                lbl_path = alt_lbl

        unique_prefix = f"cls{target_class_id}_"
        dest_img_path = os.path.join(dest_img_dir, unique_prefix + img_path.name)
        dest_lbl_path = os.path.join(dest_lbl_dir, unique_prefix + f"{stem}.txt")

        # Copiar imagen
        shutil.copy2(img_path, dest_img_path)

        # Re-indexar clase en el .txt
        if lbl_path.exists():
            with open(lbl_path, "r", encoding="utf-8") as f_in:
                lines = f_in.readlines()

            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue
                # Reemplazar class_id original por el unificado
                parts[0] = str(target_class_id)
                new_lines.append(" ".join(parts) + "\n")

            with open(dest_lbl_path, "w", encoding="utf-8") as f_out:
                f_out.writelines(new_lines)
        else:
            open(dest_lbl_path, "w").close()

        count += 1

    return count


def main():
    parser = argparse.ArgumentParser(description="Consolidador dinámico de datasets YOLO Segmentación por clases.")
    parser.add_argument("--src",  required=True, help="Ruta al directorio raíz DATASET-YOLO-SEGM.")
    parser.add_argument("--dest", default="/kaggle/working/dataset_seg_unified", help="Directorio de salida unificado.")
    args = parser.parse_args()

    src_root = Path(args.src)
    dest_root = Path(args.dest)

    if not src_root.exists():
        raise FileNotFoundError(f"Directorio origen no existe: {src_root}")

    logger.info("=" * 60)
    logger.info(f"🌱 Consolidando Dataset YOLO Segmentación desde {src_root}")
    logger.info(f"📍 Destino: {dest_root}")
    logger.info("=" * 60)

    # Descubrimiento dinámico de subcarpetas de clase
    subdirs = sorted([d for d in src_root.iterdir() if d.is_dir()])
    valid_subdirs = []

    for d in subdirs:
        # Verificar que la subcarpeta contiene train o val o imágenes directamente
        if (d / "train").exists() or (d / "val").exists() or any(f.suffix.lower() in IMG_EXTS for f in d.rglob("*")):
            valid_subdirs.append(d)

    if not valid_subdirs:
        raise RuntimeError(f"No se encontraron subcarpetas válidas de dataset en {src_root}")

    class_names = {}
    
    for cls_id, folder_path in enumerate(valid_subdirs):
        folder_name = folder_path.name
        cls_name = clean_class_name(folder_name)
        class_names[cls_id] = cls_name

        # Procesar Train (soporta folder/train o folder directamente)
        train_dir = folder_path / "train" if (folder_path / "train").exists() else folder_path
        train_count = process_folder(
            src_split_dir=train_dir,
            dest_img_dir=dest_root / "images" / "train",
            dest_lbl_dir=dest_root / "labels" / "train",
            target_class_id=cls_id
        )

        # Procesar Val
        val_dir = folder_path / "val" if (folder_path / "val").exists() else folder_path
        val_count = process_folder(
            src_split_dir=val_dir,
            dest_img_dir=dest_root / "images" / "val",
            dest_lbl_dir=dest_root / "labels" / "val",
            target_class_id=cls_id
        )

        logger.info(f"  ✅ Clase [{cls_id}] {cls_name:<25} ({folder_name}) → Train: {train_count} | Val: {val_count}")

    # Generar data.yaml
    yaml_data = {
        "path": str(dest_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": len(class_names),
        "names": {i: name for i, name in class_names.items()}
    }

    yaml_path = dest_root / "data.yaml"
    os.makedirs(dest_root, exist_ok=True)
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(yaml_data, f, sort_keys=False, default_flow_style=False)

    logger.info("=" * 60)
    logger.info(f"🎉 Dataset consolidado exitosamente con {len(class_names)} clases!")
    logger.info(f"   Archivo data.yaml creado en: {yaml_path}")
    logger.info(f"   Clases: {yaml_data['names']}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
