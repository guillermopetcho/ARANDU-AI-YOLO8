import os
import json
import random
import argparse
from pathlib import Path
import numpy as np
import scipy.stats
from sklearn.cluster import KMeans

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

from models.moco import ModelBase
import yaml

def get_transforms(global_size, local_size, local_scale=(0.10, 0.35)):
    global_trans = T.Compose([
        T.RandomResizedCrop(global_size, scale=(0.2, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    local_trans = T.Compose([
        T.RandomResizedCrop(local_size, scale=local_scale),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    base_t = T.Compose([
        T.Resize((global_size, global_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return global_trans, local_trans, base_t

def load_encoder(ckpt_path, config_path, device):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    model = ModelBase(
        dim=config["moco"]["dim"],
        predictor_hidden_dim=config["moco"].get("predictor_hidden_dim", 1024)
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    source_dict = ckpt.get("model_q", ckpt)
    clean_dict = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in source_dict.items()}
    model.load_state_dict(clean_dict, strict=False)
    model.eval()
    return model

@torch.no_grad()
def compute_class_prototypes(dataset_path, model, device, base_t, K=3, max_per_class=150):
    classes = [d.name for d in Path(dataset_path).iterdir() if d.is_dir()]
    prototypes = {}
    for cls in classes:
        cls_dir = Path(dataset_path) / cls
        imgs = list(cls_dir.rglob("*.jpg")) + list(cls_dir.rglob("*.png")) + list(cls_dir.rglob("*.jpeg"))
        sampled = random.sample(imgs, min(max_per_class, len(imgs)))
        embs = []
        for p in sampled:
            try:
                img = Image.open(p).convert('RGB')
                e = F.normalize(model(base_t(img).unsqueeze(0).to(device)), dim=-1)
                embs.append(e.cpu().numpy()[0])
            except: pass
        if len(embs) >= K:
            kmeans = KMeans(n_clusters=K, random_state=42, n_init='auto').fit(embs)
            centers = torch.tensor(kmeans.cluster_centers_).to(device)
            prototypes[cls] = F.normalize(centers, dim=-1)
        elif len(embs) > 0:
            center = torch.tensor(np.mean(embs, axis=0)).unsqueeze(0).to(device)
            prototypes[cls] = F.normalize(center, dim=-1)
    return prototypes

def get_proto_affinities(e, prototypes_dict):
    affinities = {}
    for cls, prots in prototypes_dict.items():
        sims = F.cosine_similarity(e, prots)
        affinities[cls] = torch.max(sims).item()
    return affinities

@torch.no_grad()
def audit_images_by_class(img_paths, model, device, global_size, local_size, local_scale, prototypes, delta=0.10, num_locals=10, k_nn=10):
    global_t, local_t, _ = get_transforms(global_size, local_size, local_scale)
    
    class_metrics = {cls: [] for cls in prototypes.keys()}
    
    global_bank = []
    valid_paths = []
    for p in img_paths:
        try:
            img = Image.open(p).convert('RGB')
            e_g = F.normalize(model(global_t(img).unsqueeze(0).to(device)), dim=-1)
            global_bank.append(e_g)
            valid_paths.append(p)
        except: pass
    
    bank_tensor = torch.cat(global_bank, dim=0) if global_bank else None
    
    for i, img_path in enumerate(valid_paths):
        true_class = Path(img_path).parent.name
        if true_class not in class_metrics: continue
            
        img = Image.open(img_path).convert('RGB')
        e_g1 = global_bank[i]
        
        set_g1_10, set_g1_50 = set(), set()
        if bank_tensor is not None:
            sims_g1_bank = F.cosine_similarity(e_g1, bank_tensor)
            _, topk_g1_50 = torch.topk(sims_g1_bank, min(50, bank_tensor.shape[0]))
            set_g1_50 = set(topk_g1_50.tolist())
            set_g1_10 = set(topk_g1_50[:10].tolist())
            
        local_intra_affinities = []
        local_inter_affinities = []
        recalls_10 = []
        recalls_50 = []
        sor_hits = 0
        
        for _ in range(num_locals):
            lc = local_t(img).unsqueeze(0).to(device)
            e_c = F.normalize(model(lc), dim=-1)
            
            # Prototypes
            affs = get_proto_affinities(e_c, prototypes)
            a_in = affs[true_class]
            local_intra_affinities.append(a_in)
            
            a_outs = [v for k, v in affs.items() if k != true_class]
            a_out = max(a_outs) if a_outs else 0.0
            if a_outs:
                local_inter_affinities.append(a_out)
                
            # Semantic Occupancy Ratio (SOR) basado en ProtoMargin
            proto_margin = a_in - a_out
            if proto_margin > delta:
                sor_hits += 1
            
            # Recalls
            if bank_tensor is not None:
                sims_c_bank = F.cosine_similarity(e_c, bank_tensor)
                _, topk_c_50 = torch.topk(sims_c_bank, min(50, bank_tensor.shape[0]))
                set_c_50 = set(topk_c_50.tolist())
                set_c_10 = set(topk_c_50[:10].tolist())
                recalls_10.append(len(set_g1_10.intersection(set_c_10)) / 10.0)
                recalls_50.append(len(set_g1_50.intersection(set_c_50)) / 50.0)
                
        crop_instability = np.var(local_intra_affinities) if local_intra_affinities else 0.0
        sor = sor_hits / num_locals
        
        class_metrics[true_class].append([
            np.mean(local_intra_affinities) if local_intra_affinities else 0.0,
            np.mean(local_inter_affinities) if local_inter_affinities else 0.0,
            crop_instability,
            np.mean(recalls_10) if recalls_10 else 0.0,
            np.mean(recalls_50) if recalls_50 else 0.0,
            sor
        ])

    aggregated = {}
    for cls, metrics in class_metrics.items():
        if not metrics: continue
        arr = np.array(metrics)
        
        p = float(np.mean(arr[:, 5]))
        p_clamped = max(1e-5, min(1.0 - 1e-5, p))
        h = -p_clamped * np.log2(p_clamped) - (1 - p_clamped) * np.log2(1 - p_clamped)
        
        aggregated[cls] = {
            "A_intra": float(np.mean(arr[:, 0])),
            "A_inter": float(np.mean(arr[:, 1])),
            "CropInstability": float(np.mean(arr[:, 2])),
            "Recall_10": float(np.mean(arr[:, 3])),
            "Recall_50": float(np.mean(arr[:, 4])),
            "SOR": p,
            "CaptureEntropy": float(h)
        }
    return aggregated

def main():
    parser = argparse.ArgumentParser("Auditoría P_capture por Enfermedad (SojAI)")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--config_path", type=str, default="config/moco.yaml")
    parser.add_argument("--global_size", type=int, default=512)
    parser.add_argument("--local_sizes", type=int, nargs='+', default=[128, 144, 160, 176, 192, 208, 224, 240, 256])
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--delta", type=float, default=0.10, help="Umbral de ProtoMargin para SOR")
    parser.add_argument("--out", type=str, default="disease_ablation_report.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Cargando encoder maestro ({args.ckpt_path})...")
    model = load_encoder(args.ckpt_path, args.config_path, device)
    _, _, base_t = get_transforms(args.global_size, 128)

    all_imgs = []
    classes_found = set()
    for ext in ['*.jpg', '*.png', '*.jpeg']:
        paths = list(Path(args.dataset_path).rglob(ext))
        all_imgs.extend(paths)
        for p in paths:
            classes_found.add(p.parent.name)
            
    classes_found = list(classes_found)

    print("\n[Fase 1] Extrayendo Prototipos de Clase (K-Means K=3)...")
    prototypes = compute_class_prototypes(args.dataset_path, model, device, base_t, K=3)
    
    report = {"dataset": args.dataset_path, "classes": {cls: {} for cls in classes_found}}
    
    for cls in classes_found:
        report["classes"][cls]["Crop_Ablation"] = {}

    print("\n[Fase 2] Auditoría de Crops y Curvas SOR")
    sampled_imgs = random.sample(all_imgs, min(args.samples, len(all_imgs)))

    for ls in args.local_sizes:
        print(f"Evaluando local_crop_size = {ls}px...")
        scale = (0.10, 0.35)
        metrics_by_class = audit_images_by_class(
            sampled_imgs, model, device, args.global_size, ls, scale, prototypes, delta=args.delta
        )
        
        for cls, metrics in metrics_by_class.items():
            report["classes"][cls]["Crop_Ablation"][f"Local_{ls}"] = metrics

    with open(args.out, 'w') as f:
        json.dump(report, f, indent=4)
        
    print("\n--- Semantic Occupancy Ratio (SOR) ---")
    header = f"{'Clase':<20} | " + " | ".join([f"{s}px" for s in args.local_sizes])
    print(header)
    print("-" * len(header))
    for cls in classes_found:
        row = f"{cls[:18]:<20} | "
        for ls in args.local_sizes:
            val = report["classes"][cls]["Crop_Ablation"].get(f"Local_{ls}", {}).get("SOR", 0.0)
            row += f"{val:<5.3f} | "
        print(row)
        
    print("\n--- Capture Entropy (Transición de Fase) ---")
    print(header)
    print("-" * len(header))
    for cls in classes_found:
        row = f"{cls[:18]:<20} | "
        for ls in args.local_sizes:
            val = report["classes"][cls]["Crop_Ablation"].get(f"Local_{ls}", {}).get("CaptureEntropy", 0.0)
            row += f"{val:<5.3f} | "
        print(row)
        
    print(f"\nReporte completo guardado en {args.out}")

if __name__ == "__main__":
    main()
