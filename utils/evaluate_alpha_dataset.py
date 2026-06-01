import os
import sys
import cv2
import yaml
import numpy as np
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO

# Garantizar que la raíz del proyecto esté en el path cuando el script
# se ejecuta directamente (ej: python utils/evaluate_alpha_dataset.py).
# Cuando se importa como módulo desde la raíz, sys.path ya incluye el proyecto.
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Importamos la clase extractora que ya definimos
from utils.visualize_alpha import AlphaMapExtractor


def load_ground_truth(label_path, img_width, img_height):
    """Carga GT boxes desde YOLO txt y los pasa a xyxy absoluto"""
    boxes = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    _, x_c, y_c, w, h = map(float, parts[:5])
                    x1 = int((x_c - w / 2) * img_width)
                    y1 = int((y_c - h / 2) * img_height)
                    x2 = int((x_c + w / 2) * img_width)
                    y2 = int((y_c + h / 2) * img_height)
                    boxes.append([x1, y1, x2, y2])
    return boxes

def evaluate_alpha_on_dataset(model_path, data_yaml):
    print("[*] Inicializando Evaluación Cuantitativa de Escala y Distribución...")
    model = YOLO(model_path)
    extractor = AlphaMapExtractor(model)
    
    with open(data_yaml, 'r') as f:
        data_cfg = yaml.safe_load(f)
        
    val_dir = Path(data_cfg.get('val', ''))
    if not val_dir.is_absolute():
        val_dir = Path(data_yaml).parent / val_dir
        
    images = list(val_dir.glob('*.jpg')) + list(val_dir.glob('*.png'))
    labels_dir = val_dir.parent / "labels" / val_dir.name
    
    stats = {
        'P3 (Small)': {'bg': [], 'gt': [], 'pred': [], 'small_gt': [], 'large_gt': [], 'obj_sizes': [], 'obj_alphas': []},
        'P4 (Medium)': {'bg': [], 'gt': [], 'pred': [], 'small_gt': [], 'large_gt': [], 'obj_sizes': [], 'obj_alphas': []},
        'P5 (Large)': {'bg': [], 'gt': [], 'pred': [], 'small_gt': [], 'large_gt': [], 'obj_sizes': [], 'obj_alphas': []}
    }
    
    print(f"[*] Procesando {len(images)} imágenes del dataset de validación...")
    
    # Subsampling step para no saturar memoria RAM con millones de píxeles
    sub_step = 25 
    
    for img_path in tqdm(images, desc="Inferencia"):
        results = model(str(img_path), verbose=False)
        img = cv2.imread(str(img_path))
        if img is None: continue
        h, w = img.shape[:2]
        
        pred_boxes = results[0].boxes.xyxy.cpu().numpy() if len(results[0].boxes) > 0 else []
        gt_boxes = load_ground_truth(labels_dir / f"{img_path.stem}.txt", w, h)
        
        mask_pred = np.zeros((h, w), dtype=bool)
        for box in pred_boxes:
            x1, y1, x2, y2 = map(int, box)
            mask_pred[y1:y2, x1:x2] = True
            
        mask_gt = np.zeros((h, w), dtype=bool)
        mask_gt_small = np.zeros((h, w), dtype=bool)
        mask_gt_large = np.zeros((h, w), dtype=bool)
        
        img_area = w * h
        for box in gt_boxes:
            x1, y1, x2, y2 = map(int, box)
            mask_gt[y1:y2, x1:x2] = True
            
            box_area = (x2 - x1) * (y2 - y1)
            # Definir small vs large (ej: < 2% del area de la imagen es small)
            if box_area / img_area < 0.02:
                mask_gt_small[y1:y2, x1:x2] = True
            else:
                mask_gt_large[y1:y2, x1:x2] = True
                
        # Extraer alphas y guardar muestras
        for scale in stats.keys():
            alpha_tensor = extractor.alphas.get(scale, None)
            if alpha_tensor is not None:
                a_map = cv2.resize(alpha_tensor[0, 0], (w, h))
                
                if (~mask_gt).any(): stats[scale]['bg'].extend(a_map[~mask_gt].tolist()[::sub_step])
                if mask_gt.any(): stats[scale]['gt'].extend(a_map[mask_gt].tolist()[::sub_step])
                if mask_pred.any(): stats[scale]['pred'].extend(a_map[mask_pred].tolist()[::sub_step])
                if mask_gt_small.any(): stats[scale]['small_gt'].extend(a_map[mask_gt_small].tolist()[::sub_step])
                if mask_gt_large.any(): stats[scale]['large_gt'].extend(a_map[mask_gt_large].tolist()[::sub_step])
                
                # [NUEVO] Registrar datos para Correlación Tamaño vs Alpha
                for box in gt_boxes:
                    x1, y1, x2, y2 = map(int, box)
                    box_area = (x2 - x1) * (y2 - y1)
                    if box_area > 0:
                        box_alpha = a_map[y1:y2, x1:x2].mean()
                        stats[scale]['obj_sizes'].append(box_area / img_area)
                        stats[scale]['obj_alphas'].append(box_alpha)

    extractor.remove_hooks()
    
    # Imprimir Reporte Científico
    print("\n" + "="*90)
    print("🔬 REPORTE CIENTÍFICO DE SEPARACIÓN DEL GATE (ESTADÍSTICA POBLACIONAL)")
    print("="*90)
    
    for scale, metrics in stats.items():
        print(f"\n➡️ ESCALA: {scale}")
        print(f"  {'REGIÓN':<10} | {'MEAN':<6} | {'STD':<6} | {'p25':<6} | {'MEDIAN':<6} | {'p75':<6}")
        print("-" * 65)
        for region in ['bg', 'gt', 'pred', 'small_gt', 'large_gt']:
            data = np.array(metrics[region])
            if len(data) > 0:
                print(f"  {region.upper():<10} | {data.mean():.4f} | {data.std():.4f} | {np.percentile(data, 25):.4f} | {np.median(data):.4f} | {np.percentile(data, 75):.4f}")
            else:
                print(f"  {region.upper():<10} | Sin datos en el dataset")
                
        # Calcular Delta Maestro y Cohen's d (Varianza Combinada)
        bg_data = np.array(metrics['bg'])
        gt_data = np.array(metrics['gt'])
        small_gt_data = np.array(metrics['small_gt'])
        
        if len(bg_data) > 0 and len(gt_data) > 0:
            delta = bg_data.mean() - gt_data.mean()
            
            # Cohen's d con Sigma Pooled
            var_bg = bg_data.std() ** 2
            var_gt = gt_data.std() ** 2
            sigma_pooled = np.sqrt((var_bg + var_gt) / 2)
            cohens_d = delta / sigma_pooled if sigma_pooled > 0 else 0.0
            
            print(f"  \u25B6 \u0394\u03B1 (Background - GT) = {delta:.4f}   |   Cohen's d = {cohens_d:.4f}")
            
        if len(small_gt_data) > 0 and len(gt_data) > 0:
            delta_small = gt_data.mean() - small_gt_data.mean()
            print(f"  \u25B6 Sensibilidad a escala (GT general - GT Small) = {delta_small:.4f}")
            
        # [NUEVO] Correlación de Pearson: Tamaño de Objeto vs Alpha
        sizes = np.array(metrics['obj_sizes'])
        alphas = np.array(metrics['obj_alphas'])
        if len(sizes) > 1:
            # np.corrcoef devuelve matriz 2x2, el valor de interes esta en [0,1]
            correlation = np.corrcoef(sizes, alphas)[0, 1]
            print(f"  \u25B6 Correlación de Pearson (Tamaño vs \u03B1) = {correlation:.4f}")

if __name__ == "__main__":
    import argparse
    import os
    import sys

    # Asegurar que la raiz del proyecto esté en el path,
    # independientemente de desde qué directorio se ejecute el script.
    _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)

    parser = argparse.ArgumentParser(
        description="Evaluación cuantitativa del Context Gate (alpha) sobre el dataset de validación."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Ruta al modelo YOLO híbrido (ej: runs/detect/Model3_ContextGate/weights/best.pt)."
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Ruta al YAML de dataset YOLO (ej: dataset_soja.yaml)."
    )

    args = parser.parse_args()

    errors = []
    if not os.path.isfile(args.model):
        errors.append(f"  ❌ --model no encontrado: '{args.model}'")
    if not os.path.isfile(args.data):
        errors.append(f"  ❌ --data no encontrado: '{args.data}'")

    if errors:
        print("\n[!] Errores en los argumentos:\n")
        for e in errors:
            print(e)
        print("\nEjemplo de uso:")
        print("  python utils/evaluate_alpha_dataset.py \\")
        print("    --model runs/detect/Model3_ContextGate/weights/best.pt \\")
        print("    --data dataset_soja.yaml")
        raise SystemExit(1)

    evaluate_alpha_on_dataset(args.model, args.data)
