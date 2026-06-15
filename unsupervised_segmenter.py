import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Importamos tu backbone MoCo pre-entrenado
from models.yolo_wrapper import AranduBackbone

class AutodidactSegmenter(nn.Module):
    """
    Una red pequeña que aprenderá a segmentar basándose en una función de recompensa.
    No usa etiquetas reales, solo intenta maximizar la coherencia de las texturas.
    """
    def __init__(self, in_channels, num_clusters=3):
        super().__init__()
        self.segmenter = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_clusters, kernel_size=1)
        )

    def forward(self, x):
        # Retorna probabilidades para cada cluster (pixeles)
        return F.softmax(self.segmenter(x), dim=1)

def compute_reward_loss(masks, features):
    """
    Función de recompensa (Loss No Supervisada).
    Premia al modelo si agrupa píxeles que tienen características (texturas) similares
    en el espacio latente del MoCo, y lo penaliza si mezcla texturas diferentes.
    """
    B, K, H, W = masks.shape
    C = features.shape[1]
    
    # Aplanar dimensiones espaciales
    masks = masks.view(B, K, -1)        # [B, K, N] donde N = H*W
    features = features.view(B, C, -1)  # [B, C, N]
    
    # Normalizar features para comparar con Similitud Coseno
    features = F.normalize(features, p=2, dim=1)
    
    # 1. Calcular el centroide (textura promedio) de cada cluster
    mask_sum = masks.sum(dim=-1, keepdim=True) + 1e-6
    centroids = torch.bmm(masks, features.transpose(1, 2)) / mask_sum  # [B, K, C]
    centroids = F.normalize(centroids, p=2, dim=2)
    
    # 2. Calcular la similitud de cada pixel con los centroides
    sim_to_centroids = torch.bmm(centroids, features)  # [B, K, N]
    
    # 3. La recompensa es qué tan similares son los píxeles al centroide del cluster al que fueron asignados
    compactness_reward = (sim_to_centroids * masks).sum(dim=-1) / mask_sum.squeeze(-1) # [B, K]
    
    # Convertimos la recompensa en una Loss (minimizamos el negativo de la recompensa)
    loss = -compactness_reward.mean()
    
    return loss

def train_autodidact_segmentation(image_path, encoder_path, iterations=100, num_clusters=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Usando dispositivo: {device}")

    # 1. Cargar el Encoder MoCo (Congelado, ya conoce las texturas)
    print("[*] Cargando AranduBackbone...")
    backbone = AranduBackbone(moco_checkpoint_path=encoder_path).to(device)
    backbone.eval()
    for param in backbone.parameters():
        param.requires_grad = False

    # 2. Preparar la imagen
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen: {image_path}")
    
    # Redimensionar para la red (ej. 512x512)
    img_resized = cv2.resize(img_bgr, (512, 512))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    # Normalización estándar ImageNet (como usa ConvNeXt)
    tensor_img = torch.from_numpy(img_rgb).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    tensor_img = tensor_img.permute(2, 0, 1).unsqueeze(0).to(device)
    tensor_img = (tensor_img - mean) / std

    # Extraer features de las texturas (usamos P3 o P4 del backbone)
    with torch.no_grad():
        features_list = backbone(tensor_img)
        # Usamos features de P3 (resolución decente, buena textura)
        texture_features = features_list[1] # [B, C, H/8, W/8]
        
    # 3. Inicializar el Segmentador Autodidacta
    in_channels = texture_features.shape[1]
    segmenter = AutodidactSegmenter(in_channels, num_clusters).to(device)
    optimizer = torch.optim.Adam(segmenter.parameters(), lr=0.01)

    print(f"[*] Iniciando aprendizaje por recompensa (Maximizando similitud de texturas)...")
    for i in range(iterations):
        optimizer.zero_grad()
        
        # El segmentador predice una máscara basada en las texturas
        masks = segmenter(texture_features)
        
        # Calculamos qué tan buena fue la segmentación (Reward Loss)
        loss = compute_reward_loss(masks, texture_features)
        
        loss.backward()
        optimizer.step()
        
        if (i+1) % 20 == 0:
            print(f"  Iteración {i+1}/{iterations} | Recompensa (Neg Loss): {-loss.item():.4f}")

    # 4. Visualización
    segmenter.eval()
    with torch.no_grad():
        final_masks = segmenter(texture_features) # [1, K, H', W']
        
    # Hacer upsample de la máscara al tamaño original
    final_masks_up = F.interpolate(final_masks, size=(512, 512), mode='bilinear', align_corners=False)
    segmentation_map = torch.argmax(final_masks_up[0], dim=0).cpu().numpy()

    # Visualizar
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.title("Imagen Original")
    plt.imshow(img_rgb)
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title("Mapa de Segmentación Autodidacta")
    plt.imshow(segmentation_map, cmap='viridis')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Superposición")
    overlay = img_rgb.copy()
    # Crear mascara de colores
    colored_mask = plt.get_cmap('viridis')(segmentation_map / (num_clusters - 1))[:, :, :3] * 255
    overlay = cv2.addWeighted(overlay, 0.6, colored_mask.astype(np.uint8), 0.4, 0)
    plt.imshow(overlay)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig("segmentacion_autodidacta.png", dpi=300)
    print("\n[+] Resultado guardado en 'segmentacion_autodidacta.png'")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Ruta a la imagen de soja")
    parser.add_argument("--encoder", required=True, help="Ruta al moco_encoder_ready.pth")
    parser.add_argument("--clusters", type=int, default=3, help="Fondo, Hoja sana, Enfermedad (defecto 3)")
    parser.add_argument("--iters", type=int, default=100, help="Iteraciones de aprendizaje por recompensa")
    args = parser.parse_args()
    
    train_autodidact_segmentation(args.image, args.encoder, args.iters, args.clusters)
