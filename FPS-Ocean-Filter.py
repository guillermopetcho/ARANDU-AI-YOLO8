import os
import random
import torch
import torch.nn.functional as F
import argparse
from pathlib import Path
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import sys
import shutil

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.moco import ModelBase

def load_encoder(checkpoint_path, dim=512, device='cuda'):
    print(f"[*] Cargando Encoder FPS desde: {checkpoint_path}")
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

class OceanDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths
        self.transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def __len__(self):
        return len(self.paths)
        
    def __getitem__(self, idx):
        path = self.paths[idx]
        try:
            img = Image.open(path).convert('RGB')
            return self.transform(img), path
        except Exception:
            return torch.zeros((3, 384, 384)), "" # Dummy en caso de error

@torch.no_grad()
def extract_embeddings_in_batches(image_paths, encoder, device, batch_size=128, desc="Extrayendo Embeddings"):
    dataset = OceanDataset(image_paths)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    embeds = []
    valid_paths = []
    
    for batch_tensors, batch_paths in tqdm(dataloader, desc=desc, leave=False):
        batch_tensors = batch_tensors.to(device)
        feat = encoder.encoder(batch_tensors)
        
        if isinstance(feat, list):
            feat = feat[-1]
            
        if feat.ndim == 4:
            feat = F.adaptive_avg_pool2d(feat, (1, 1)).flatten(1)
        elif feat.ndim == 3:
            feat = feat.mean(dim=1)
            
        embed = F.normalize(feat, dim=-1)
        
        # Mantener todo en GPU (20.000 vectores de dim 512 pesan ~40 MB, insignificante)
        embeds.append(embed)
        valid_paths.extend([p for p in batch_paths if p != ""])
                
    if embeds:
        # torch.cat concatena directamente en la VRAM sin cuellos de botella CPU/GPU
        return torch.cat(embeds, dim=0), valid_paths
    return None, []

def farthest_point_sampling(embeddings, target=1000):
    """
    Farthest Point Sampling (FPS) en el espacio latente.
    Maximiza explícitamente la cobertura geométrica del manifold de la clase.
    """
    N = embeddings.shape[0]
    target = min(target, N)
    
    selected_indices = []
    
    # max_sim_to_selected guarda la similitud del vecino MÁS CERCANO del set ya seleccionado
    max_sim_to_selected = torch.full((N,), -float('inf'), device=embeddings.device)
    
    # Elegimos el primer punto (el más representativo, cerca del centroide global)
    centroid = embeddings.mean(dim=0, keepdim=True)
    centroid = F.normalize(centroid, dim=-1)
    sims_to_centroid = torch.mm(embeddings, centroid.t()).squeeze()
    first_idx = sims_to_centroid.argmax().item()
    
    for i in tqdm(range(target), desc="    Ejecutando FPS", leave=False):
        if i == 0:
            best_idx = first_idx
        else:
            # Enmascaramos los puntos ya seleccionados con infinito
            max_sim_to_selected[selected_indices] = float('inf')
            
            # FPS: Seleccionamos el punto cuyo vecino más cercano en el set elegido está MÁS LEJOS.
            # Más lejos = Mínima similitud.
            best_idx = max_sim_to_selected.argmin().item()
            
        selected_indices.append(best_idx)
        
        # Actualizar distancias
        new_embed = embeddings[best_idx].unsqueeze(0)
        sims_to_new = torch.mm(embeddings, new_embed.t()).squeeze(1)
        max_sim_to_selected = torch.max(max_sim_to_selected, sims_to_new)
        
    return selected_indices

def run_fps_filtering(input_dir, output_dir, real_dir, encoder, device, target_per_class=1000, batch_size=128, bio_threshold=0.55):
    input_path = Path(input_dir)
    classes = [d for d in input_path.iterdir() if d.is_dir()]
    exts = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff'}
    
    for cls_dir in classes:
        cls_name = cls_dir.name
        out_cls_dir = os.path.join(output_dir, cls_name)
        os.makedirs(out_cls_dir, exist_ok=True)
        
        print(f"\n[*] Procesando Pipeline FPS Masivo para '{cls_name}'")
        
        # 1. Extraer semántica de biológica real (Ancla)
        real_cls_name = cls_name.split('-OCEANS')[0]
        real_cls_dir = os.path.join(real_dir, real_cls_name)
        real_images = [str(f) for f in Path(real_cls_dir).rglob("*") if f.suffix.lower() in exts]
        
        if not real_images:
            print(f"    [!] Sin dataset real para {real_cls_name} (original: {cls_name}). Saltando.")
            continue
            
        # Limitamos a 800 reales para que la matriz sim_to_real quepa holgadamente en VRAM
        real_images = random.sample(real_images, min(800, len(real_images)))
        real_bank, _ = extract_embeddings_in_batches(real_images, encoder, device, batch_size=batch_size, desc="    Embeddings Reales")
        
        # 2. Extraer semántica sintética masiva
        synth_images = [str(f) for f in cls_dir.rglob("*.jpg")]
        if not synth_images:
            continue
            
        print(f"    -> Encontrados {len(synth_images)} océanos generados.")
        synth_bank, valid_paths = extract_embeddings_in_batches(synth_images, encoder, device, batch_size=batch_size, desc="    Embeddings Sintéticos")
        
        if synth_bank is None or real_bank is None:
            continue
            
        # 3. FILTRO BIOLÓGICO SUAVE DINÁMICO (Local Density)
        import numpy as np
        
        # A) Densidad Local de la distribución Real
        sim_real_to_real = torch.mm(real_bank, real_bank.t())
        sim_real_to_real.fill_diagonal_(-1.0) # Ignorar similitud consigo mismo
        
        k_neighbors = min(10, sim_real_to_real.shape[1])
        top_sims_real = torch.topk(sim_real_to_real, k=k_neighbors, dim=1).values
        real_local_density = top_sims_real.mean(dim=1)
        
        # Percentil 5 de la densidad biológica (los bordes del manifold)
        dynamic_threshold = np.percentile(real_local_density.cpu().numpy(), 5)
        # Relajamos levemente (2%) para tolerar texturas compuestas
        dynamic_threshold = dynamic_threshold - 0.02
        
        print(f"    -> Umbral de Densidad Local (Top-{k_neighbors}) para {cls_name}: {dynamic_threshold:.3f}")
        
        # B) Densidad Local de los Sintéticos respecto al mundo Real
        sims_to_real = torch.mm(synth_bank, real_bank.t())
        top_sims_synth = torch.topk(sims_to_real, k=k_neighbors, dim=1).values
        synth_local_density = top_sims_synth.mean(dim=1)
        
        # Máscara biológica: ¿Está inmerso en la distribución biológica?
        valid_mask = synth_local_density >= dynamic_threshold
        num_dropped = (~valid_mask).sum().item()
        
        print(f"    -> Retención Biológica: {valid_mask.sum().item()}/{len(synth_images)} océanos viables ({num_dropped} descartados).")
        
        synth_bank = synth_bank[valid_mask]
        valid_paths = [valid_paths[i] for i in range(len(valid_paths)) if valid_mask[i].item()]
        
        print(f"    -> Filtro Biológico (>{bio_threshold}): {num_dropped} monstruos descartados. {len(valid_paths)} viables.")
        
        if len(valid_paths) == 0:
            print("    [!] Ningún océano superó el umbral biológico.")
            continue
            
        # 4. Selección Farthest Point Sampling (FPS)
        best_indices = farthest_point_sampling(synth_bank, target=target_per_class)
        
        # 5. Guardar en disco
        for idx in tqdm(best_indices, desc="    Guardando Dataset", leave=False):
            src_path = valid_paths[idx]
            dst_path = os.path.join(out_cls_dir, os.path.basename(src_path))
            shutil.copy2(src_path, dst_path)
            
        print(f" -> Guardados {len(best_indices)} océanos de máxima cobertura.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Farthest Point Sampling (FPS) con Filtro Biológico")
    parser.add_argument("--input", type=str, required=True, help="Directorio con océanos generados masivamente")
    parser.add_argument("--output", type=str, required=True, help="Directorio donde guardar el Dataset Final")
    parser.add_argument("--real_dir", type=str, required=True, help="Directorio del dataset REAL 384x384 (Para el filtro biológico)")
    parser.add_argument("--weights", type=str, required=True, help="Ruta al moco_encoder_384_ready.pth")
    parser.add_argument("--dim", type=int, default=512, help="Dimensión del espacio latente")
    parser.add_argument("--target", type=int, default=1000, help="Cuántos océanos curados queremos por clase")
    parser.add_argument("--batch_size", type=int, default=128, help="Tamaño de lote para extracción (T4 puede manejar >128)")
    parser.add_argument("--threshold", type=float, default=0.55, help="Umbral mínimo de similitud biológica (Ej: 0.55)")
    
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    encoder = load_encoder(args.weights, dim=args.dim, device=device)
    
    run_fps_filtering(
        input_dir=args.input,
        output_dir=args.output,
        real_dir=args.real_dir,
        encoder=encoder,
        device=device,
        target_per_class=args.target,
        batch_size=args.batch_size,
        bio_threshold=args.threshold
    )
    
    print("\n[+] Filtrado Biológico + FPS Masivo Completado.")
