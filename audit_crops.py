import os
import random
import argparse
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt

def get_transforms(global_size, local_size):
    # Simula exactamente las transformaciones base de MoCo
    global_trans = T.Compose([
        T.RandomResizedCrop(global_size, scale=(0.2, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip()
    ])
    local_trans = T.Compose([
        T.RandomResizedCrop(local_size, scale=(0.10, 0.35)), # Escala local de moco.yaml
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip()
    ])
    return global_trans, local_trans

def plot_moco_views(img_path, global_trans, local_trans, num_local=4, save_path=None):
    try:
        img = Image.open(img_path).convert('RGB')
    except Exception as e:
        print(f"Error loading {img_path}: {e}")
        return

    # Generar vistas
    global_views = [global_trans(img) for _ in range(2)]
    local_views = [local_trans(img) for _ in range(num_local)]

    # Plot
    fig = plt.figure(figsize=(15, 6))
    
    # Original
    ax = fig.add_subplot(2, 3 + num_local//2, 1)
    ax.imshow(img)
    ax.set_title("Original")
    ax.axis('off')
    
    # Globals
    for i, gv in enumerate(global_views):
        ax = fig.add_subplot(2, 3 + num_local//2, 2 + i)
        ax.imshow(gv)
        ax.set_title(f"Global Crop {i+1}")
        ax.axis('off')
        
    # Locals
    for i, lv in enumerate(local_views):
        ax = fig.add_subplot(2, 3 + num_local//2, 4 + i + (1 if num_local > 2 else 0))
        ax.imshow(lv)
        ax.set_title(f"Local Crop {i+1}")
        ax.axis('off')
        
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser("Auditar Crops de MoCo")
    parser.add_argument("--dataset_path", type=str, required=True, help="Directorio del dataset (ej: train/)")
    parser.add_argument("--global_size", type=int, default=512)
    parser.add_argument("--local_size", type=int, default=128)
    parser.add_argument("--samples", type=int, default=10, help="Número de imágenes aleatorias a visualizar")
    parser.add_argument("--out_dir", type=str, default="audit_output", help="Directorio para guardar imágenes")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    
    # Recopilar imágenes válidas
    all_imgs = []
    for ext in ['*.jpg', '*.png', '*.jpeg']:
        all_imgs.extend(list(Path(args.dataset_path).rglob(ext)))
        
    if not all_imgs:
        print(f"No se encontraron imágenes en {args.dataset_path}")
        return
        
    sampled_imgs = random.sample(all_imgs, min(args.samples, len(all_imgs)))
    
    global_t, local_t = get_transforms(args.global_size, args.local_size)
    
    for i, img_path in enumerate(sampled_imgs):
        out_file = os.path.join(args.out_dir, f"audit_sample_{i+1}.png")
        plot_moco_views(str(img_path), global_t, local_t, save_path=out_file)
        print(f"Guardado {out_file} (Original: {img_path.name})")

if __name__ == "__main__":
    main()
