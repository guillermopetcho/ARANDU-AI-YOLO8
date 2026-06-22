"""
pseudo_label_generator.py — Generador Autodidacta de Dataset para YOLO Segmentación

Este script actúa como el "Maestro". Utiliza el AranduBackbone (MoCo v3) preentrenado
para extraer representaciones texturales profundas de las imágenes. Luego, aplica
K-Means clustering en el espacio latente para separar:
  - Fondo (Background)
  - Tejido Sano (Healthy)
  - Lesiones/Enfermedades (Disease)

Finalmente, extrae los polígonos de las lesiones y genera los archivos .txt 
con formato YOLO Segmentación, listos para entrenar a YOLO26.
"""

import os
import glob
import argparse
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans

from models.yolo_wrapper import AranduBackbone

def extract_polygons(mask, img_w, img_h, class_id=0, min_points=5):
    """
    Extrae contornos de una máscara binaria y los formatea para YOLO.
    Retorna una lista de strings: "class_id x1 y1 x2 y2 ... xn yn"
    """
    # Encontrar contornos
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    yolo_lines = []
    for contour in contours:
        # Simplificar el polígono para no tener miles de puntos (epsilon ajustado)
        epsilon = 0.005 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Ignorar contornos muy pequeños o ruido
        if len(approx) < min_points or cv2.contourArea(approx) < 50:
            continue
            
        # Normalizar coordenadas entre 0.0 y 1.0
        polygon_norm = []
        for point in approx:
            x_norm = point[0][0] / img_w
            y_norm = point[0][1] / img_h
            # Clamp para asegurar que estén exactamente entre 0 y 1
            x_norm = max(0.0, min(1.0, x_norm))
            y_norm = max(0.0, min(1.0, y_norm))
            polygon_norm.extend([f"{x_norm:.5f}", f"{y_norm:.5f}"])
            
        if len(polygon_norm) >= 6: # Al menos un triángulo (3 puntos = 6 coordenadas)
            line = f"{class_id} " + " ".join(polygon_norm)
            yolo_lines.append(line)
            
    return yolo_lines

def generate_pseudo_labels(images_dir, output_labels_dir, encoder_path, num_clusters=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Dispositivo: {device}")
    
    # 1. Cargar el Maestro (Encoder)
    print(f"[*] Cargando AranduBackbone desde {encoder_path}...")
    backbone = AranduBackbone(moco_checkpoint_path=encoder_path).to(device)
    backbone.eval()
    
    os.makedirs(output_labels_dir, exist_ok=True)
    
    # Extensiones válidas
    image_paths = []
    for ext in ('*.jpg', '*.jpeg', '*.png'):
        image_paths.extend(glob.glob(os.path.join(images_dir, ext)))
        
    if not image_paths:
        print(f"[!] No se encontraron imágenes en {images_dir}")
        return

    print(f"[*] Generando pseudo-etiquetas para {len(image_paths)} imágenes...")
    
    for img_path in tqdm(image_paths):
        img_name = os.path.basename(img_path)
        txt_name = os.path.splitext(img_name)[0] + ".txt"
        out_txt_path = os.path.join(output_labels_dir, txt_name)
        
        # Si ya existe, lo saltamos (permite reanudar)
        if os.path.exists(out_txt_path):
            continue

        # 2. Cargar y preparar imagen
        img_bgr = cv2.imread(img_path)
        if img_bgr is None: continue
        
        orig_h, orig_w = img_bgr.shape[:2]
        img_resized = cv2.resize(img_bgr, (512, 512))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        tensor_img = torch.from_numpy(img_rgb).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        tensor_img = tensor_img.permute(2, 0, 1).unsqueeze(0).to(device)
        tensor_img = (tensor_img - mean) / std
        
        # 3. Extraer conocimiento del Encoder
        # AranduBackbone.forward() retorna [P2, P3, P4, P5] en orden de stride ascendente.
        # P3 (stride 8, index 1) ofrece el mejor balance entre detalle espacial y
        # riqueza semántica para clustering de texturas de enfermedad.
        _P3_INDEX = 1
        _P3_EXPECTED_CHANNELS = 256  # yolo_channels[1] default en AranduBackbone
        with torch.no_grad():
            features_list = backbone(tensor_img)
            # HIGH-3 FIX: Constante nombrada + sanity check de canales para detectar
            # cambios en el orden de outputs de AranduBackbone en lugar de fallar
            # silenciosamente con features incorrectas.
            feat = features_list[_P3_INDEX][0]  # [C, H_f, W_f] — primer (único) elemento del batch
            C, H_f, W_f = feat.shape
            assert C == _P3_EXPECTED_CHANNELS, (
                f"P3 tiene {C} canales pero se esperaban {_P3_EXPECTED_CHANNELS}. "
                f"Verificar que AranduBackbone.forward() siga retornando [P2, P3, P4, P5] "
                f"y que yolo_channels[1] == {_P3_EXPECTED_CHANNELS}."
            )
            
            # Normalizar y aplanar para K-Means
            feat_flat = feat.view(C, -1).permute(1, 0).cpu().numpy() # [N, C]
            
        # 4. K-Means en el espacio latente
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=1)
        labels = kmeans.fit_predict(feat_flat)
        cluster_map = labels.reshape(H_f, W_f)
        
        # Upsample de la máscara al tamaño original de la imagen
        cluster_map_up = cv2.resize(cluster_map.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        
        # 5. Heurística de Identificación (¿Cuál cluster es la enfermedad?)
        # Suposición general:
        # - El cluster más grande suele ser el fondo.
        # - El cluster de tamaño medio es la hoja sana.
        # - El cluster más pequeño (o el más texturizado) suele ser la enfermedad.
        cluster_areas = [np.sum(cluster_map_up == i) for i in range(num_clusters)]
        
        # Ordenamos los clusters de menor a mayor área
        sorted_clusters = np.argsort(cluster_areas)
        
        # Asumimos que el cluster más pequeño es la enfermedad (clase 0 temporalmente)
        # NOTA: En un pipeline de producción, podrías cruzar esto con el clasificador
        # lineal para asignar la clase correcta (healthy, mosaic, etc.).
        disease_cluster_idx = sorted_clusters[0] 
        disease_mask = (cluster_map_up == disease_cluster_idx)
        
        # 6. Extraer polígonos y guardar
        # Usamos clase 0 como "Lesión Genérica" para este auto-aprendizaje
        yolo_polygons = extract_polygons(disease_mask, orig_w, orig_h, class_id=0)
        
        with open(out_txt_path, 'w') as f:
            for poly in yolo_polygons:
                f.write(poly + "\n")

    print("\n[+] ¡Pseudo-Etiquetado Finalizado!")
    print(f"[*] Las etiquetas YOLO están listas en: {output_labels_dir}")
    print("[*] Ahora puedes usar estas etiquetas para entrenar YOLO26 con train_yolo_seg.py")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generador de pseudo-etiquetas YOLO Segmentación usando MoCo")
    parser.add_argument("--images", required=True, help="Directorio con imágenes de entrada")
    parser.add_argument("--labels", required=True, help="Directorio de salida para los .txt de YOLO")
    parser.add_argument("--encoder", required=True, help="Ruta al moco_encoder_ready.pth")
    parser.add_argument("--clusters", type=int, default=3, help="Cantidad de texturas a buscar (Fondo, Hoja, Lesión)")
    
    args = parser.parse_args()
    generate_pseudo_labels(args.images, args.labels, args.encoder, args.clusters)
