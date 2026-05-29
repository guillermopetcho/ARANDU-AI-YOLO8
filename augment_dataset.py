import os
import argparse
import glob
from pathlib import Path
from PIL import Image
from torchvision import transforms as T
from tqdm import tqdm

def get_leaf_pipeline(size=640):
    """
    Pipeline de Data Augmentation para HOJAS COMPLETAS.
    Mantiene la morfología general y la biología del color.
    """
    return T.Compose([
        # Rotación libre: las hojas no tienen un "arriba" o "abajo" estricto
        T.RandomRotation(degrees=45),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        # Simulamos diferentes ángulos de la cámara con perspectiva leve
        T.RandomPerspective(distortion_scale=0.2, p=0.3),
        # ColorJitter: Permitimos variación en luz y saturación, 
        # pero el Matiz (HUE) DEBE SER CASI CERO para no cambiar enfermedades (ej. volver amarillo un tejido verde)
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.015),
        # Desenfoque ocasional simulando cámara fuera de foco
        T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.3),
        T.Resize((size, size))
    ])

def get_texture_pipeline(size=384):
    """
    Pipeline de Data Augmentation para TEXTURAS Y MICRO-PATRONES.
    Enfocado en acercamientos (zooms) a lesiones pequeñas (ej. Frog Eye, manchas).
    """
    return T.Compose([
        # Recorte aleatorio severo para forzar zoom en texturas locales
        T.RandomResizedCrop(size, scale=(0.4, 0.9), ratio=(0.8, 1.2)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        # Acentuamos un poco más el contraste para resaltar bordes de las lesiones
        T.ColorJitter(brightness=0.2, contrast=0.4, saturation=0.2, hue=0.01),
        T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.2),
    ])

def process_images(input_dir, output_dir, mode, num_augments, size):
    # Extensiones válidas
    exts = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff'}
    files = [f for f in Path(input_dir).rglob("*") if f.suffix.lower() in exts]
    
    if not files:
        print(f" No se encontraron imágenes en {input_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    pipeline = get_leaf_pipeline(size) if mode == 'leaf' else get_texture_pipeline(size)
    
    print(f"[*] Iniciando Data Augmentation ({mode.upper()})")
    print(f"[*] Imágenes encontradas: {len(files)}")
    print(f"[*] Copias por imagen: {num_augments}")
    print(f"[*] Carpeta de salida: {output_dir}")
    print("-" * 50)

    for img_path in tqdm(files, desc="Aumentando"):
        try:
            img = Image.open(img_path).convert("RGB")
            
            # Guardamos la original redimensionada opcionalmente (o tal cual)
            base_name = img_path.stem
            
            # Generamos N versiones aumentadas
            for i in range(num_augments):
                aug_img = pipeline(img)
                out_name = f"{base_name}_aug_{mode}_{i+1}.jpg"
                out_path = os.path.join(output_dir, out_name)
                aug_img.save(out_path, format="JPEG", quality=95)
                
        except Exception as e:
            print(f"\n Error procesando {img_path.name}: {e}")

    print("\n Proceso completado exitosamente.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Herramienta de Data Augmentation para SojAI (Hojas y Texturas)")
    parser.add_argument("--input", "-i", type=str, required=True, help="Directorio con imágenes de entrada")
    parser.add_argument("--output", "-o", type=str, required=True, help="Directorio donde guardar las imágenes aumentadas")
    parser.add_argument("--mode", "-m", type=str, choices=['leaf', 'texture'], required=True, 
                        help="'leaf' para hojas enteras, 'texture' para micro-lesiones")
    parser.add_argument("--copies", "-c", type=int, default=3, help="Número de imágenes aumentadas a generar por cada original")
    parser.add_argument("--size", "-s", type=int, default=640, help="Resolución de salida (px)")
    
    args = parser.parse_args()
    
    process_images(args.input, args.output, args.mode, args.copies, args.size)
