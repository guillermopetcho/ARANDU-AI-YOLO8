import os
import random
import torch
import torch.nn.functional as F
import argparse
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter, ImageChops
from torchvision import transforms
from tqdm import tqdm
import sys
import shutil

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.moco import ModelBase

def load_encoder(checkpoint_path, dim=512, device='cuda'):
    print(f"[*] Cargando Active Teacher desde: {checkpoint_path}")
    model = ModelBase(dim=dim).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict = {}
    for k, v in ckpt.items():
        if isinstance(v, dict):
            state_dict = {key.replace('module.', '').replace('_orig_mod.', ''): val for key, val in v.items()}
            break
    if not state_dict:
        state_dict = {k.replace('module.', '').replace('_orig_mod.', ''): v for k, v in ckpt.items()}
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

@torch.no_grad()
def get_embedding(model, img_tensor):
    feat = model.encoder(img_tensor)
    if isinstance(feat, list):
        feat = feat[-1]
    if feat.ndim == 4:
        feat = F.adaptive_avg_pool2d(feat, (1, 1)).flatten(1)
    elif feat.ndim == 3:
        feat = feat.mean(dim=1)
    return F.normalize(feat, dim=-1)

def generate_fractal_mask(size):
    scale = max(4, size // 8)
    noise_arr = np.random.randint(0, 255, (scale, scale), dtype=np.uint8)
    noise_img = Image.fromarray(noise_arr, mode='L').resize((size, size), Image.Resampling.BICUBIC)
    base_mask = Image.new('L', (size, size), 0)
    draw = ImageDraw.Draw(base_mask)
    draw.ellipse((0, 0, size, size), fill=255)
    base_mask = base_mask.filter(ImageFilter.GaussianBlur(size // 6))
    fractal = ImageChops.multiply(noise_img, base_mask)
    return fractal.point(lambda p: min(255, int(max(0, p - 80) * 2.5)))

def active_ocean_synthesis(real_dir, output_dir, encoder, device, target_per_class=1000, hard_patch_mining=False):
    exts = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff'}
    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    classes = [d for d in Path(real_dir).iterdir() if d.is_dir()]
    
    for cls_dir in classes:
        cls_name = cls_dir.name
        out_cls_dir = os.path.join(output_dir, cls_name)
        os.makedirs(out_cls_dir, exist_ok=True)
        
        print(f"\n[*] Nivel 5: Generación Activa para '{cls_name}'")
        real_images = [str(f) for f in cls_dir.rglob("*") if f.suffix.lower() in exts]
        random.shuffle(real_images)
        real_images = real_images[:800] # Limite por RAM/Tiempo
        
        if not real_images:
            continue
            
        # 1. Extraer embeddings de toda la clase
        embeds = []
        valid_paths = []
        pil_images = []
        
        for p in tqdm(real_images, desc="    1. Mapeo Latente", leave=False):
            try:
                img = Image.open(p).convert('RGB')
                t = transform(img).unsqueeze(0).to(device)
                embeds.append(get_embedding(encoder, t))
                valid_paths.append(p)
                pil_images.append(img)
            except: pass
            
        bank = torch.cat(embeds, dim=0) # (N, D)
        centroid = bank.mean(dim=0, keepdim=True)
        centroid = F.normalize(centroid, dim=-1)
        
        # Similitud par a par para KNN Latente
        sim_matrix = torch.mm(bank, bank.t()) # (N, N)
        
        generated_oceans = []
        # Generamos el doble del target para luego aplicar CoreSet
        num_to_generate = target_per_class * 2 
        
        for i in tqdm(range(num_to_generate), desc="    2. Síntesis de Océanos Híbridos", leave=False):
            # Seleccionar una semilla aleatoria
            seed_idx = random.randint(0, len(valid_paths)-1)
            
            # Encontrar los 5 vecinos top en el espacio latente
            top_k_indices = torch.topk(sim_matrix[seed_idx], k=min(5, len(valid_paths))).indices.tolist()
            neighbor_imgs = [pil_images[idx] for idx in top_k_indices]
            
            # Lienzo base
            mosaic = neighbor_imgs[0].resize((512, 512), Image.Resampling.LANCZOS)
            
            patches_to_blend = []
            
            # Extraer parches de los vecinos semánticos (Jerárquicos: 64 a 256)
            for _ in range(60): # Extraemos 60 parches candidatos
                src_img = random.choice(neighbor_imgs)
                w, h = src_img.size
                patch_size = random.randint(64, 256)
                if w < patch_size or h < patch_size:
                    src_img = src_img.resize((max(w, patch_size), max(h, patch_size)), Image.Resampling.LANCZOS)
                    w, h = src_img.size
                
                x1 = random.randint(0, w - patch_size)
                y1 = random.randint(0, h - patch_size)
                crop = src_img.crop((x1, y1, x1 + patch_size, y1 + patch_size))
                patches_to_blend.append(crop)
                
            # Opcional (Nivel 5 completo): Hard Patch Mining
            if hard_patch_mining:
                patch_tensors = torch.stack([transform(p.resize((384,384))) for p in patches_to_blend]).to(device)
                
                # 1. Filtro de Saliencia/Varianza Cromática
                # Evita que el sistema seleccione "bordes negros" o "fondo liso" como anomalías
                variances = patch_tensors.std(dim=[2, 3]).mean(dim=1) # (60,)
                
                # Nos quedamos con los 50 parches con mayor textura (descartamos 10 muy lisos)
                valid_idx = torch.topk(variances, k=50).indices
                valid_tensors = patch_tensors[valid_idx]
                valid_patches = [patches_to_blend[i] for i in valid_idx.tolist()]
                
                with torch.no_grad():
                    patch_embeds = get_embedding(encoder, valid_tensors) # (50, D)
                
                # 2. Hard Mining (Distancia al Centroide)
                patch_sims = torch.mm(patch_embeds, centroid.t()).squeeze() # (50,)
                
                # Nos quedamos con los 40 parches más alejados del centroide (los más raros)
                hardest_indices = torch.topk(patch_sims, k=40, largest=False).indices.tolist()
                patches_to_blend = [valid_patches[idx] for idx in hardest_indices]
            else:
                patches_to_blend = random.sample(patches_to_blend, 40)
                
            # Pegado Orgánico (Alpha + Rotación + Fractal Mask)
            for crop in patches_to_blend:
                patch_size = crop.size[0]
                mask = generate_fractal_mask(patch_size)
                angle = random.uniform(0, 360)
                crop = crop.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False)
                mask = mask.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False)
                
                px = random.randint(-patch_size // 2, 512 - patch_size // 2)
                py = random.randint(-patch_size // 2, 512 - patch_size // 2)
                mosaic.paste(crop, (px, py), mask)
                
            generated_oceans.append(mosaic)
            
        # 3. CoreSet Selection Final (Diversidad Latente)
        # Extraemos embeddings de los océanos generados
        synth_embeds = []
        for img in tqdm(generated_oceans, desc="    3. Evaluación CoreSet", leave=False):
            t = transform(img).unsqueeze(0).to(device)
            synth_embeds.append(get_embedding(encoder, t))
            
        synth_bank = torch.cat(synth_embeds, dim=0) # (2*Target, D)
        
        # Realismo (Pertenencia a la clase global)
        sims_to_centroid = torch.mm(synth_bank, centroid.t()).squeeze(1) # (N,)
        
        # Clonación (Similitud a la imagen real individual más cercana)
        sims_to_real = torch.mm(synth_bank, bank.t())
        max_sim_to_real, _ = sims_to_real.max(dim=1)
        
        # Novedad (Queremos que sea distinto a cualquier imagen de entrenamiento específica)
        novelty = 1.0 - max_sim_to_real
        
        selected_indices = []
        max_sim_to_selected = torch.full((len(generated_oceans),), -1.0, device=device)
        
        for i in range(min(target_per_class, len(generated_oceans))):
            if i == 0:
                # Primer océano: maximiza Realismo + Novedad
                scores = 0.6 * sims_to_centroid + 0.4 * novelty
                best_idx = scores.argmax().item()
            else:
                # CoreSet Balanceado: Realismo + Novedad - Redundancia
                scores = 0.6 * sims_to_centroid + 0.4 * novelty - max_sim_to_selected
                scores[selected_indices] = -float('inf')
                best_idx = scores.argmax().item()
                
            selected_indices.append(best_idx)
            new_embed = synth_bank[best_idx].unsqueeze(0)
            sims_to_new = torch.mm(synth_bank, new_embed.t()).squeeze(1)
            max_sim_to_selected = torch.max(max_sim_to_selected, sims_to_new)
            
        # Guardar ganadores
        for idx in tqdm(selected_indices, desc="    4. Escribiendo Dataset D", leave=False):
            img = generated_oceans[idx]
            out_path = os.path.join(out_cls_dir, f"{cls_name}_active_ocean_{idx:04d}.jpg")
            img.save(out_path, format="JPEG", quality=95)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nivel 5: Active Teacher-Guided Ocean Synthesis")
    parser.add_argument("--real_dir", type=str, required=True, help="Directorio del dataset REAL 384x384")
    parser.add_argument("--output", type=str, required=True, help="Directorio donde guardar el Dataset D")
    parser.add_argument("--weights", type=str, required=True, help="Ruta al moco_encoder_384_ready.pth")
    parser.add_argument("--dim", type=int, default=512, help="Dimensión del espacio latente")
    parser.add_argument("--target", type=int, default=1000, help="Cuántos océanos generar por clase")
    parser.add_argument("--hard_mining", action='store_true', help="Activar puntuación de parches difíciles (Lento pero muy potente)")
    
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    encoder = load_encoder(args.weights, dim=args.dim, device=device)
    
    active_ocean_synthesis(
        real_dir=args.real_dir,
        output_dir=args.output,
        encoder=encoder,
        device=device,
        target_per_class=args.target,
        hard_patch_mining=args.hard_mining
    )
    
    print("\n[+] Síntesis Guiada por Profesor Activo (Nivel 5) Completada.")
