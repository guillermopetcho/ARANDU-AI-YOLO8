import os
import random
import argparse
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter, ImageChops
from tqdm import tqdm

def generate_fractal_mask(size):
    """
    Genera una máscara orgánica fractal usando ruido redimensionado
    multiplicado por un gradiente circular, para evitar el sesgo de 
    'manchas circulares perfectas'.
    """
    # 1. Ruido blanco en muy baja resolución
    scale = max(4, size // 8)
    noise_arr = np.random.randint(0, 255, (scale, scale), dtype=np.uint8)
    noise_img = Image.fromarray(noise_arr, mode='L')
    
    # 2. Redimensionar con Bicubic genera un patrón de 'nubes' (ruido suave)
    noise_img = noise_img.resize((size, size), Image.Resampling.BICUBIC)
    
    # 3. Máscara circular base difusa para que los bordes exteriores mueran a 0
    base_mask = Image.new('L', (size, size), 0)
    draw = ImageDraw.Draw(base_mask)
    draw.ellipse((0, 0, size, size), fill=255)
    base_mask = base_mask.filter(ImageFilter.GaussianBlur(size // 6))
    
    # 4. Multiplicar el ruido por el círculo
    fractal = ImageChops.multiply(noise_img, base_mask)
    
    # 5. Thresholding suave (aumentar el contraste) para crear bordes orgánicos pero definidos
    fractal = fractal.point(lambda p: min(255, int(max(0, p - 80) * 2.5)))
    
    return fractal

def create_mosaic(image_paths, output_size=512, num_patches=40):
    """
    Crea un 'Océano de Textura' continuo con Máscaras Fractales.
    """
    bg_path = random.choice(image_paths)
    try:
        mosaic = Image.open(bg_path).convert('RGB')
        mosaic = mosaic.resize((output_size, output_size), Image.Resampling.LANCZOS)
    except Exception:
        mosaic = Image.new('RGB', (output_size, output_size), (0, 0, 0))
    
    for _ in range(num_patches):
        img_path = random.choice(image_paths)
        try:
            img = Image.open(img_path).convert('RGB')
            w, h = img.size
            
            patch_size = random.randint(64, 256)
            if w < patch_size or h < patch_size:
                img = img.resize((max(w, patch_size), max(h, patch_size)), Image.Resampling.LANCZOS)
                w, h = img.size
                
            x1 = random.randint(0, w - patch_size)
            y1 = random.randint(0, h - patch_size)
            crop = img.crop((x1, y1, x1 + patch_size, y1 + patch_size))
            
            # Generar máscara fractal orgánica
            mask = generate_fractal_mask(patch_size)
            
            # Rotación aleatoria (0-360)
            angle = random.uniform(0, 360)
            crop = crop.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False)
            mask = mask.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False)
            
            # Posición aleatoria (permitiendo que parte del parche caiga fuera)
            px = random.randint(-patch_size // 2, output_size - patch_size // 2)
            py = random.randint(-patch_size // 2, output_size - patch_size // 2)
            
            # Pegado alfa-blended
            mosaic.paste(crop, (px, py), mask)
        except Exception:
            pass
            
    return mosaic

def process_class(class_dir, output_dir, class_name, num_images=1000, output_size=512, num_patches=40):
    exts = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff'}
    images = [str(f) for f in Path(class_dir).rglob("*") if f.suffix.lower() in exts]
    
    if not images:
        print(f" [!] No se encontraron imágenes válidas en {class_dir}")
        return
        
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"[*] Procesando '{class_name}' | {len(images)} imgs fuente -> {num_images} mosaicos orgánicos")
    
    for i in tqdm(range(num_images), desc=f"Generando {class_name}"):
        mosaic = create_mosaic(images, output_size=output_size, num_patches=num_patches)
        out_path = os.path.join(output_dir, f"{class_name}_ocean_{i:04d}.jpg")
        mosaic.save(out_path, format="JPEG", quality=95)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Scale Disease Ocean Generator")
    parser.add_argument("--input", "-i", type=str, required=True, help="Directorio raíz del dataset (debe contener subcarpetas por enfermedad)")
    parser.add_argument("--output", "-o", type=str, required=True, help="Directorio de salida para los océanos")
    parser.add_argument("--num", "-n", type=int, default=1000, help="Número de océanos a generar por enfermedad")
    parser.add_argument("--size", "-s", type=int, default=512, help="Resolución de salida (px)")
    parser.add_argument("--patches", "-p", type=int, default=40, help="Número de parches por imagen")
    
    args = parser.parse_args()
    
    input_root = Path(args.input)
    if not input_root.exists() or not input_root.is_dir():
        print(f"Error: El directorio de entrada {args.input} no existe.")
        exit(1)
        
    classes = [d for d in input_root.iterdir() if d.is_dir()]
    
    if not classes:
        print("No se encontraron subcarpetas de clases. Verifica la estructura del dataset.")
        exit(1)
        
    print(f"Iniciando síntesis oceánica para {len(classes)} clases...")
    
    for cls_dir in classes:
        cls_name = cls_dir.name
        cls_out = os.path.join(args.output, cls_name)
        process_class(str(cls_dir), cls_out, cls_name, args.num, args.size, args.patches)
        
    print("\n¡Síntesis de océanos de textura completada con éxito!")
