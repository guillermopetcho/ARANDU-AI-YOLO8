import os
import glob
import yaml
import argparse
import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt

# Importar las transformaciones exactamente como las usa tu modelo
from models.moco import get_global_transforms, get_local_transforms

def unnormalize(tensor):
    """Revierte la normalización de ImageNet para poder visualizar la imagen."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = tensor * std + mean
    return torch.clamp(tensor, 0, 1)

def tensor_to_pil(tensor):
    """Convierte un tensor a una imagen PIL."""
    unnorm = unnormalize(tensor)
    return T.ToPILImage()(unnorm)

def main():
    parser = argparse.ArgumentParser(description="Visualizar Aumentos de MoCo")
    parser.add_argument("--input", "-i", type=str, required=True, help="Carpeta con algunas imágenes de prueba")
    parser.add_argument("--output", "-o", type=str, default="debug_augs", help="Carpeta donde guardar los resultados")
    parser.add_argument("--config", "-c", type=str, default="config/moco.yaml", help="Ruta al moco.yaml")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Cargar parámetros de augmentation del config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    cfg = config.get("moco", {})
    global_size = cfg.get('global_crop_size', 640)
    local_size  = cfg.get('local_crop_size', 128)
    ultra_size  = int(local_size * 0.75)
    aug_cfg = cfg.get('augmentation', {})

    print("[*] Cargando configuraciones de augmentation:")
    print(f"    - Global size: {global_size}")
    print(f"    - Local size: {local_size}")
    print(f"    - Parámetros: {aug_cfg}")

    t_q, t_k = get_global_transforms(global_size=global_size, aug_cfg=aug_cfg)
    t_locals = get_local_transforms(local_size=local_size, ultra_size=ultra_size, aug_cfg=aug_cfg)

    # Buscar algunas imágenes
    exts = ('*.jpg', '*.jpeg', '*.png')
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(args.input, '**', ext), recursive=True))
    
    files = files[:10] # Solo las primeras 10 para no saturar

    if not files:
        print(f"No se encontraron imágenes en {args.input}")
        return

    print(f"[*] Procesando {len(files)} imágenes y guardando en {args.output}/")

    for idx, img_path in enumerate(files):
        try:
            img = Image.open(img_path).convert("RGB")
            
            # Generar transformaciones
            q_view = t_q(img)
            k_view = t_k(img)
            local_views = [t(img) for t in t_locals]

            # Crear un plot para visualizar todo junto
            # 1 Original + 2 Globales + N Locales
            n_cols = 3 + len(local_views)
            fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
            
            # Mostrar Original
            axes[0].imshow(img)
            axes[0].set_title("Original")
            axes[0].axis('off')

            # Mostrar Global Q
            axes[1].imshow(tensor_to_pil(q_view))
            axes[1].set_title("Global View 1 (Q)")
            axes[1].axis('off')

            # Mostrar Global K
            axes[2].imshow(tensor_to_pil(k_view))
            axes[2].set_title("Global View 2 (K)")
            axes[2].axis('off')

            # Mostrar Locales
            for i, local_v in enumerate(local_views):
                axes[3+i].imshow(tensor_to_pil(local_v))
                axes[3+i].set_title(f"Local {i+1}")
                axes[3+i].axis('off')

            out_file = os.path.join(args.output, f"sample_{idx}.jpg")
            plt.tight_layout()
            plt.savefig(out_file, bbox_inches='tight', dpi=150)
            plt.close()
            print(f"    Guardado -> {out_file}")

        except Exception as e:
            print(f"Error con {img_path}: {e}")

    print("[*] ¡Listo! Revisa la carpeta de salida.")

if __name__ == "__main__":
    main()
