import os
import cv2
import numpy as np
import shutil
from pathlib import Path
import random

def corrupt_image(img):
    """
    Aplica corrupciones sintéticas que simulan condiciones adversas en campo.
    """
    # 1. Reducción aleatoria de brillo (simula sombras fuertes o días nublados)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    # Reduce el brillo entre un 30% y un 50%
    factor = random.uniform(0.5, 0.7)
    v = np.clip(v * factor, 0, 255).astype(np.uint8)
    hsv[:, :, 2] = v
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)  # FIX: HSV_BGR era constante incorrecta
    
    # 2. Blur aleatorio (simula fuera de foco por viento o movimiento)
    if random.random() > 0.3:
        ksize = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (ksize, ksize), 0)
        
    # 3. Ruido Gaussiano muy leve (ruido de sensor en baja luz)
    if random.random() > 0.5:
        noise = np.random.normal(0, 10, img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        
    return img

def create_corrupted_dataset(source_yaml, output_dir):
    """
    Lee un dataset de YOLO y crea una copia corrupta de validación/test.
    """
    import yaml
    with open(source_yaml, 'r') as f:
        data = yaml.safe_load(f)
        
    val_path = data.get('val', '')
    if not val_path:
        print("[!] No se encontró ruta 'val' en el yaml.")
        return

    # Asumimos rutas relativas o absolutas. Manejo básico:
    source_dir = Path(val_path)
    if not source_dir.is_absolute():
        source_dir = Path(source_yaml).parent / source_dir
        
    dest_dir = Path(output_dir) / "images"
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    # Copiar labels directamente
    source_labels = source_dir.parent / "labels" / source_dir.name
    dest_labels = Path(output_dir) / "labels"
    if source_labels.exists():
        if dest_labels.exists():
            shutil.rmtree(dest_labels)
        shutil.copytree(source_labels, dest_labels)
        print(f"[*] Labels copiados a {dest_labels}")
        
    # Procesar imágenes
    img_files = list(source_dir.glob('*.jpg')) + list(source_dir.glob('*.png'))
    print(f"[*] Corrompiendo {len(img_files)} imágenes para el Dataset B...")
    
    for i, img_path in enumerate(img_files):
        img = cv2.imread(str(img_path))
        if img is None: continue
            
        corrupted = corrupt_image(img)
        cv2.imwrite(str(dest_dir / img_path.name), corrupted)
        
        if (i+1) % 100 == 0:
            print(f"    Progreso: {i+1}/{len(img_files)}")
            
    # Crear nuevo YAML
    new_yaml = Path(output_dir) / "corrupted_dataset.yaml"
    data['val'] = str(dest_dir.absolute())
    data['path'] = str(Path(output_dir).absolute())
    
    with open(new_yaml, 'w') as f:
        yaml.dump(data, f)
        
    print(f"[+] Dataset corrupto generado en {output_dir}")
    print(f"[+] Nuevo YAML: {new_yaml}")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Genera una copia corrupta del dataset de validación para stress-testing "
                    "de robustez del modelo ante condiciones adversas de campo."
    )
    parser.add_argument(
        "--source-yaml",
        type=str,
        required=True,
        help="Ruta al YAML de dataset YOLO origen (ej: dataset_soja.yaml)."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directorio de salida donde se guardará el dataset corrupto."
    )

    args = parser.parse_args()

    if not os.path.isfile(args.source_yaml):
        print(f"❌ --source-yaml no encontrado: '{args.source_yaml}'")
        print("\nEjemplo de uso:")
        print("  python utils/corrupt_dataset.py \\")
        print("    --source-yaml dataset_soja.yaml \\")
        print("    --output-dir dataset_soja_corrupto")
        raise SystemExit(1)

    create_corrupted_dataset(args.source_yaml, args.output_dir)
