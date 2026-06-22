"""
unsupervised_segmenter.py — Segmentación No Supervisada por Recompensa (AranduSSL)

Segmenta imágenes de soja usando exclusivamente las representaciones texturales
aprendidas por el encoder MoCo, sin etiquetas manuales.

Función de Recompensa (3 términos):
  1. Compactness:  Maximizar similitud intra-cluster (texturas similares juntas).
  2. Separation:   Minimizar similitud inter-cluster (clusters distintos entre sí).
  3. Entropy:      Regularizar el tamaño de los clusters para evitar colapso a uno solo.

Uso:
    python unsupervised_segmenter.py \\
        --image /ruta/a/imagen.jpg \\
        --encoder /ruta/a/moco_encoder_ready.pth \\
        --clusters 3 --iters 150
"""

import argparse
import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt

from models.yolo_wrapper import AranduBackbone

logger = logging.getLogger("AranduSegmenter")


# ---------------------------------------------------------------------------
# Segmentador Autodidacta
# ---------------------------------------------------------------------------

class AutodidactSegmenter(nn.Module):
    """
    Red ligera que aprende a segmentar basándose en una función de recompensa.
    No usa etiquetas reales, solo maximiza la coherencia textural en el espacio
    latente del MoCo.

    Arquitectura: Conv3×3 + BN + ReLU → Conv3×3 + BN + ReLU → Conv1×1
    El doble bloque Conv3×3 le da campo receptivo suficiente para capturar
    contexto local antes de la clasificación por píxel.
    """
    def __init__(self, in_channels: int, num_clusters: int = 3):
        super().__init__()
        mid = max(64, in_channels // 4)
        self.segmenter = nn.Sequential(
            nn.Conv2d(in_channels, mid, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, num_clusters, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Retorna soft-assignments [B, K, H, W] con softmax por píxel."""
        return F.softmax(self.segmenter(x), dim=1)


# ---------------------------------------------------------------------------
# Función de Recompensa (Loss No Supervisada)
# ---------------------------------------------------------------------------

def compute_reward_loss(
    masks: torch.Tensor,
    features: torch.Tensor,
    separation_weight: float = 0.5,
    entropy_weight: float = 0.3,
) -> torch.Tensor:
    """
    Loss de recompensa de 3 términos para segmentación no supervisada.

    Args:
        masks:    [B, K, H, W] — soft-assignments del segmentador.
        features: [B, C, H, W] — feature maps del encoder MoCo (congelado).
        separation_weight: Peso del término de separación inter-cluster.
        entropy_weight:    Peso del término de entropía para balanceo de clusters.

    Returns:
        loss escalar (menor = mejor segmentación).

    Términos:
        1. Compactness (↑): Similitud media de cada píxel con el centroide de su cluster.
           Premia agrupar texturas similares.

        2. Separation (↓): Similitud media entre centroides de clusters distintos.
           Penaliza clusters redundantes (que miren texturas iguales).

        3. Entropy (↑): Entropía de la distribución de tamaños de cluster.
           Previene que un solo cluster absorba todos los píxeles.
    """
    B, K, H, W = masks.shape
    N = H * W

    # Aplanar dimensiones espaciales
    masks_flat = masks.view(B, K, N)             # [B, K, N]
    features_flat = features.view(B, -1, N)      # [B, C, N]
    features_flat = F.normalize(features_flat, p=2, dim=1)

    # --- 1. COMPACTNESS: similitud intra-cluster ---
    mask_sum = masks_flat.sum(dim=-1, keepdim=True) + 1e-6        # [B, K, 1]
    centroids = torch.bmm(masks_flat, features_flat.transpose(1, 2))  # [B, K, C]
    centroids = centroids / mask_sum
    centroids = F.normalize(centroids, p=2, dim=2)

    sim_to_centroids = torch.bmm(centroids, features_flat)        # [B, K, N]
    compactness = (sim_to_centroids * masks_flat).sum(dim=-1)     # [B, K]
    compactness = compactness / mask_sum.squeeze(-1)
    loss_compactness = -compactness.mean()

    # --- 2. SEPARATION: disimilitud inter-cluster ---
    # Similitud coseno entre todos los pares de centroides
    # [B, K, C] × [B, C, K] → [B, K, K]
    centroid_sims = torch.bmm(centroids, centroids.transpose(1, 2))  # [B, K, K]
    # Excluir la diagonal (similitud consigo mismo = 1.0)
    if K > 1:
        eye = torch.eye(K, device=masks.device).unsqueeze(0)        # [1, K, K]
        off_diag_mask = 1.0 - eye
        # Media de similitudes entre pares distintos (queremos minimizarla)
        n_pairs = K * (K - 1)
        loss_separation = (centroid_sims * off_diag_mask).sum() / (B * n_pairs)
    else:
        loss_separation = torch.zeros(1, device=masks.device)

    # --- 3. ENTROPY: regularización de balance de clusters ---
    # Distribución de probabilidad de pertenencia promedio por cluster
    cluster_probs = masks_flat.mean(dim=-1)                # [B, K] — proporción de píxeles por cluster
    cluster_probs = cluster_probs.mean(dim=0)              # [K] — promedio sobre el batch
    cluster_probs = cluster_probs.clamp(min=1e-6)
    cluster_probs = cluster_probs / cluster_probs.sum()    # Normalizar
    # Entropía máxima = log(K) → clusters perfectamente balanceados
    entropy = -(cluster_probs * torch.log(cluster_probs)).sum()
    max_entropy = torch.log(torch.tensor(float(K), device=masks.device))
    # Penalizar baja entropía (clusters desbalanceados)
    loss_entropy = -(entropy / (max_entropy + 1e-6))

    # --- LOSS TOTAL ---
    loss = loss_compactness + separation_weight * loss_separation + entropy_weight * loss_entropy

    return loss


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def train_autodidact_segmentation(
    image_path: str,
    encoder_path: str,
    num_clusters: int = 3,
    iterations: int = 100,
    output_path: str = "segmentacion_autodidacta.png",
):
    """
    Ejecuta la segmentación no supervisada por recompensa sobre una imagen.

    Args:
        image_path:   Ruta a la imagen de soja.
        encoder_path: Ruta al checkpoint SSL (moco_encoder_ready.pth o checkpoint de training).
        num_clusters: Número de clusters (ej. 3 = Fondo, Hoja sana, Enfermedad).
        iterations:   Iteraciones de optimización por recompensa.
        output_path:  Ruta de salida del PNG con la visualización.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Usando dispositivo: {device}")

    # 1. Cargar el Encoder MoCo (Congelado — ya conoce las texturas)
    print("[*] Cargando AranduBackbone...")
    backbone = AranduBackbone(moco_checkpoint_path=encoder_path).to(device)
    backbone.eval()
    for param in backbone.parameters():
        param.requires_grad = False

    # 2. Preparar la imagen
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen: {image_path}")

    img_resized = cv2.resize(img_bgr, (512, 512))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

    # Normalización estándar ImageNet (como usa ConvNeXt)
    tensor_img = torch.from_numpy(img_rgb).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    tensor_img = tensor_img.permute(2, 0, 1).unsqueeze(0).to(device)
    tensor_img = (tensor_img - mean) / std

    # 3. Extraer features multi-escala del backbone
    # AranduBackbone.forward() retorna [P2, P3, P4, P5] en orden de stride ascendente.
    _P3_INDEX = 1
    _P3_EXPECTED_CHANNELS = 256  # yolo_channels[1] default en AranduBackbone
    with torch.no_grad():
        features_list = backbone(tensor_img)  # [P2, P3, P4, P5]
        # Usamos P3 (stride 8): balance entre resolución espacial y semántica textural.
        texture_features = features_list[_P3_INDEX]   # [B, 256, H/8, W/8]
        assert texture_features.shape[1] == _P3_EXPECTED_CHANNELS, (
            f"P3 tiene {texture_features.shape[1]} canales pero se esperaban "
            f"{_P3_EXPECTED_CHANNELS}. Verificar el orden de outputs de AranduBackbone."
        )

    # 4. Inicializar el Segmentador Autodidacta
    in_channels = texture_features.shape[1]
    segmenter = AutodidactSegmenter(in_channels, num_clusters).to(device)
    optimizer = torch.optim.Adam(segmenter.parameters(), lr=0.01)
    # Cosine annealing para convergencia más suave en las últimas iteraciones
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)

    print(f"[*] Iniciando aprendizaje por recompensa ({iterations} iteraciones, {num_clusters} clusters)...")
    for i in range(iterations):
        optimizer.zero_grad()

        masks = segmenter(texture_features)
        loss = compute_reward_loss(masks, texture_features)

        loss.backward()
        optimizer.step()
        scheduler.step()

        if (i + 1) % 20 == 0:
            # Mostrar distribución de clusters para diagnosticar colapso
            with torch.no_grad():
                cluster_sizes = masks.mean(dim=(0, 2, 3))  # [K]
                sizes_str = " | ".join([f"C{j}:{cluster_sizes[j]:.2%}" for j in range(num_clusters)])
            print(f"  Iter {i+1:>4}/{iterations} | Loss: {loss.item():.4f} | {sizes_str}")

    # 5. Generar mapa de segmentación final
    segmenter.eval()
    with torch.no_grad():
        final_masks = segmenter(texture_features)  # [1, K, H', W']

    # Upsample al tamaño original
    final_masks_up = F.interpolate(final_masks, size=(512, 512), mode='bilinear', align_corners=False)
    segmentation_map = torch.argmax(final_masks_up[0], dim=0).cpu().numpy()

    # 6. Visualización
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: Imagen Original
    axes[0].imshow(img_rgb)
    axes[0].set_title("Imagen Original", fontsize=14, fontweight='bold')
    axes[0].axis('off')

    # Panel 2: Mapa de Segmentación
    cmap = plt.get_cmap('Set1', num_clusters)
    im = axes[1].imshow(segmentation_map, cmap=cmap, vmin=0, vmax=num_clusters - 1)
    axes[1].set_title("Segmentación Autodidacta", fontsize=14, fontweight='bold')
    axes[1].axis('off')
    # Leyenda de clusters
    cbar = fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04, ticks=range(num_clusters))
    cbar.set_ticklabels([f"Cluster {i}" for i in range(num_clusters)])

    # Panel 3: Superposición
    # Normalizar segmentation_map a [0, 1] para el colormap (safe para K=1)
    seg_normalized = segmentation_map.astype(np.float32)
    if num_clusters > 1:
        seg_normalized = seg_normalized / (num_clusters - 1)
    colored_mask = (cmap(seg_normalized)[:, :, :3] * 255).astype(np.uint8)
    overlay = cv2.addWeighted(img_rgb, 0.6, colored_mask, 0.4, 0)
    axes[2].imshow(overlay)
    axes[2].set_title("Superposición", fontsize=14, fontweight='bold')
    axes[2].axis('off')

    plt.suptitle(f"Segmentación No Supervisada (K={num_clusters}) — AranduSSL",
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n[+] Resultado guardado en '{output_path}'")
    plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Segmentación no supervisada de enfermedades en soja usando features MoCo."
    )
    parser.add_argument("--image", required=True, help="Ruta a la imagen de soja.")
    parser.add_argument("--encoder", required=True,
                        help="Ruta al checkpoint SSL (moco_encoder_ready.pth o checkpoint de training).")
    parser.add_argument("--clusters", type=int, default=3,
                        help="Número de clusters: Fondo, Hoja sana, Enfermedad (default: 3).")
    parser.add_argument("--iters", type=int, default=150,
                        help="Iteraciones de aprendizaje por recompensa (default: 150).")
    parser.add_argument("--output", type=str, default="segmentacion_autodidacta.png",
                        help="Ruta del PNG de salida (default: segmentacion_autodidacta.png).")
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        raise FileNotFoundError(f"Imagen no encontrada: {args.image}")
    if not os.path.isfile(args.encoder):
        raise FileNotFoundError(f"Encoder no encontrado: {args.encoder}")

    train_autodidact_segmentation(
        image_path=args.image,
        encoder_path=args.encoder,
        num_clusters=args.clusters,
        iterations=args.iters,
        output_path=args.output,
    )
