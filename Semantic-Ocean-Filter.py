import os
import random
import torch
import torch.nn.functional as F
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import sys
import shutil

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.moco import ModelBase

def load_encoder(checkpoint_path, dim=512, device='cuda'):
    print(f"[*] Cargando Encoder Teacher desde: {checkpoint_path}")
    model = ModelBase(dim=dim).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    
    state_dict = {}
    for k, v in ckpt.items():
        if isinstance(v, dict):
            state_dict = {key.replace('module.', '').replace('_orig_mod.', ''): val 
                          for key, val in v.items()}
            break
    if not state_dict:
        state_dict = {k.replace('module.', '').replace('_orig_mod.', ''): v for k, v in ckpt.items()}
        
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

@torch.no_grad()
def get_embedding(model, img, device):
    """
    Extrae la representación del BACKBONE puro, soportando CNNs y ViTs.
    """
    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tensor = transform(img).unsqueeze(0).to(device)
    feat = model.encoder(tensor)
    
    if isinstance(feat, list):
        feat = feat[-1]
    
    # Blindaje arquitectónico
    if feat.ndim == 4:
        feat = F.adaptive_avg_pool2d(feat, (1, 1)).flatten(1)
    elif feat.ndim == 3:
        feat = feat.mean(dim=1)
    elif feat.ndim == 2:
        pass
        
    return F.normalize(feat, dim=-1)

def extract_bank(image_paths, encoder, device, desc="Extrayendo"):
    embeds = []
    valid_paths = []
    for img_path in tqdm(image_paths, desc=desc, leave=False):
        try:
            img = Image.open(img_path).convert('RGB')
            embeds.append(get_embedding(encoder, img, device))
            valid_paths.append(img_path)
        except Exception:
            pass
    if embeds:
        return torch.cat(embeds, dim=0), valid_paths
    return None, []

def select_coreset(real_bank, synth_bank, target=1000):
    """
    Selección guiada por representación (Score = Realismo - Redundancia).
    """
    N = synth_bank.shape[0]
    target = min(target, N)
    
    # 1. Similitud a la biología real (Realismo)
    # Cuán parecido es este sintético a SU IMAGEN REAL MÁS CERCANA
    sims_to_real = torch.mm(synth_bank, real_bank.t()) # (N, M)
    realism_scores, _ = sims_to_real.max(dim=1) # (N,)
    
    # 2. Selección Greedy (O(N) optimizado con tensores)
    selected_indices = []
    
    # Mantenemos un registro de cuán redundante es cada candidato frente al set ya seleccionado
    max_sim_to_selected = torch.full((N,), -1.0, device=synth_bank.device)
    
    for i in tqdm(range(target), desc="    Seleccionando Coreset Semántico", leave=False):
        if i == 0:
            # El primero es simplemente el más realista
            best_idx = realism_scores.argmax().item()
        else:
            # Score dinámico: queremos realismo alto y redundancia baja
            scores = realism_scores - max_sim_to_selected
            scores[selected_indices] = -float('inf') # Ignorar los ya elegidos
            best_idx = scores.argmax().item()
            
        selected_indices.append(best_idx)
        
        # Actualizar redundancia para la próxima iteración (vectorizado)
        new_embed = synth_bank[best_idx].unsqueeze(0) # (1, D)
        sims_to_new = torch.mm(synth_bank, new_embed.t()).squeeze(1) # (N,)
        max_sim_to_selected = torch.max(max_sim_to_selected, sims_to_new)
        
    return selected_indices

def filter_oceans(input_dir, output_dir, encoder, device, real_dir, target_per_class=1000):
    input_path = Path(input_dir)
    classes = [d for d in input_path.iterdir() if d.is_dir()]
    
    exts = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff'}
    
    for cls_dir in classes:
        cls_name = cls_dir.name
        out_cls_dir = os.path.join(output_dir, cls_name)
        os.makedirs(out_cls_dir, exist_ok=True)
        
        real_cls_dir = os.path.join(real_dir, cls_name)
        if not os.path.exists(real_cls_dir):
            print(f"[!] Saltando {cls_name}: No hay dataset real para anclar la semántica.")
            continue
            
        print(f"\n[*] Procesando '{cls_name}'")
        
        # 1. Banco Real
        real_images = [str(f) for f in Path(real_cls_dir).rglob("*") if f.suffix.lower() in exts]
        random.shuffle(real_images)
        real_images = real_images[:800] # Limitar a 800 reales por velocidad
        real_bank, _ = extract_bank(real_images, encoder, device, desc="    Semántica Real")
        
        if real_bank is None:
            continue
            
        # 2. Banco Sintético
        synth_images = [str(f) for f in cls_dir.rglob("*.jpg")]
        random.shuffle(synth_images)
        # Podemos extraer hasta 10000 o 20000 (el script puede con ello, limitamos a 5000 por RAM si es GPU pequeña)
        synth_images = synth_images[:5000] 
        synth_bank, valid_synth_paths = extract_bank(synth_images, encoder, device, desc="    Semántica Sintética")
        
        if synth_bank is None:
            continue
            
        # 3. CoreSet Selection (Realismo - Redundancia)
        best_indices = select_coreset(real_bank, synth_bank, target=target_per_class)
        
        # 4. Guardar los ganadores
        for idx in tqdm(best_indices, desc="    Guardando Océanos", leave=False):
            src_path = valid_synth_paths[idx]
            dst_path = os.path.join(out_cls_dir, os.path.basename(src_path))
            shutil.copy2(src_path, dst_path)
            
        print(f" -> Retenidos {len(best_indices)} mejores océanos.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Representation-Guided Dataset Synthesis")
    parser.add_argument("--input", type=str, required=True, help="Directorio con océanos generados (sin filtrar)")
    parser.add_argument("--output", type=str, required=True, help="Directorio donde guardar el dataset filtrado")
    parser.add_argument("--weights", type=str, required=True, help="Ruta al moco_encoder_384_ready.pth")
    parser.add_argument("--real_dir", type=str, required=True, help="Directorio del dataset REAL 384x384")
    parser.add_argument("--dim", type=int, default=512, help="Dimensión del espacio latente")
    parser.add_argument("--target", type=int, default=1000, help="Cuántos océanos curados queremos por clase")
    
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    encoder = load_encoder(args.weights, dim=args.dim, device=device)
    
    filter_oceans(
        input_dir=args.input,
        output_dir=args.output,
        encoder=encoder,
        device=device,
        real_dir=args.real_dir,
        target_per_class=args.target
    )
    
    print("\n[+] Curación de Océanos Jerárquicos Completada.")
