import os
import json
import random
import argparse
from pathlib import Path
import numpy as np
import scipy.linalg

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
def compute_class_centroids(dataset_path, model, device, base_t, max_per_class=100):
    classes = [d.name for d in Path(dataset_path).iterdir() if d.is_dir()]
    centroids = {}
    for cls in classes:
        cls_dir = Path(dataset_path) / cls
        imgs = list(cls_dir.rglob("*.jpg")) + list(cls_dir.rglob("*.png")) + list(cls_dir.rglob("*.jpeg"))
        sampled = random.sample(imgs, min(max_per_class, len(imgs)))
        embs = []
        for p in sampled:
            try:
                img = Image.open(p).convert('RGB')
                e = F.normalize(model(base_t(img).unsqueeze(0).to(device)), dim=-1)
                embs.append(e)
            except Exception: pass
        if embs:
            center = torch.cat(embs, dim=0).mean(dim=0, keepdim=True)
            centroids[cls] = F.normalize(center, dim=-1)
    return centroids

@torch.no_grad()
def ablation_curves(img_paths, model, device, centroids, scales=None):
    """Bloque A (R(s)) y Bloque B (B(s))"""
    if scales is None:
        scales = [512, 384, 320, 256, 224, 192, 160, 128, 96]
    base_t = T.Compose([
        T.Resize((512, 512)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    retention_geom = {s: [] for s in scales}
    retention_bio = {s: [] for s in scales}
    
    for img_path in img_paths:
        try:
            img = Image.open(img_path).convert('RGB')
            true_class = Path(img_path).parent.name
            e_base = F.normalize(model(base_t(img).unsqueeze(0).to(device)), dim=-1)
            
            for s in scales:
                t_s = T.Compose([
                    T.Resize((s, s)),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                e_s = F.normalize(model(t_s(img).unsqueeze(0).to(device)), dim=-1)
                
                # Bloque A: Geometric Retention
                retention_geom[s].append(F.cosine_similarity(e_base, e_s).item())
                
                # Bloque B: Biological Retention
                if centroids and true_class in centroids:
                    retention_bio[s].append(F.cosine_similarity(e_s, centroids[true_class]).item())
        except Exception: pass
        
    return {
        "R_s": {s: np.mean(vals) if vals else 0.0 for s, vals in retention_geom.items()},
        "B_s": {s: np.mean(vals) if vals else 0.0 for s, vals in retention_bio.items()}
    }

@torch.no_grad()
def audit_images(img_paths, model, device, global_size, local_size, local_scale, centroids, num_locals=4, num_simulations=5):
    """Bloques C, D y E: Evaluación profunda por imagen."""
    global_t, local_t, _ = get_transforms(global_size, local_size, local_scale)
    
    G_matrix = [] # [A, S, M] (Geometría contrastiva)
    bio_metrics = [] # [BioAcc, BioMargin, GapHealthy, V_bio]
    
    for img_path in img_paths:
        try:
            img = Image.open(img_path).convert('RGB')
            true_class = Path(img_path).parent.name
        except Exception: continue
            
        other_imgs = random.sample([p for p in img_paths if p != img_path], min(5, len(img_paths)-1))
        e_others = []
        for opath in other_imgs:
            try:
                oimg = Image.open(opath).convert('RGB')
                e_others.append(F.normalize(model(global_t(oimg).unsqueeze(0).to(device)), dim=-1))
            except Exception: pass
            
        img_A, img_S, img_M = [], [], []
        img_bio_hits, img_bio_margins, img_gaps_healthy = [], [], []
        
        for _ in range(num_simulations):
            g1 = global_t(img).unsqueeze(0).to(device)
            e_g1 = F.normalize(model(g1), dim=-1)
            
            local_embs = []
            local_bio_sims = [] # Para V_bio
            
            for _ in range(num_locals):
                lc = local_t(img).unsqueeze(0).to(device)
                e_c = F.normalize(model(lc), dim=-1)
                local_embs.append(e_c)
                
                # Bloque C: Geometría
                alignment = F.cosine_similarity(e_c, e_g1).item()
                img_A.append(alignment)
                
                if e_others:
                    sims_neg = [F.cosine_similarity(e_c, e_o).item() for e_o in e_others]
                    img_M.append(alignment - np.mean(sims_neg))
                    
                # Bloque D: Separabilidad Biológica
                if centroids and true_class in centroids:
                    sims_k = {cls: F.cosine_similarity(e_c, center).item() for cls, center in centroids.items()}
                    sim_true = sims_k[true_class]
                    local_bio_sims.append(sim_true)
                    
                    pred_class = max(sims_k, key=sims_k.get)
                    img_bio_hits.append(1 if pred_class == true_class else 0)
                    
                    sims_other = [v for k, v in sims_k.items() if k != true_class]
                    if sims_other:
                        img_bio_margins.append(sim_true - max(sims_other))
                        
                    healthy_key = next((k for k in centroids.keys() if "health" in k.lower() or "sana" in k.lower()), None)
                    if healthy_key and true_class != healthy_key:
                        img_gaps_healthy.append(sim_true - sims_k[healthy_key])
                
            for i in range(len(local_embs)):
                for j in range(i + 1, len(local_embs)):
                    img_S.append(F.cosine_similarity(local_embs[i], local_embs[j]).item())
                    
            # Bloque E: Varianza Biológica (V_bio)
            v_bio = np.var(local_bio_sims) if local_bio_sims else 0.0
            
        G_matrix.append([np.mean(img_A), np.mean(img_S), np.mean(img_M)])
        
        bio_metrics.append([
            np.mean(img_bio_hits) if img_bio_hits else 0.0,
            np.mean(img_bio_margins) if img_bio_margins else 0.0,
            np.mean(img_gaps_healthy) if img_gaps_healthy else 0.0,
            v_bio
        ])

    return np.array(G_matrix), np.array(bio_metrics)

def compute_mahalanobis(candidates_G, baseline_G):
    """Bloque F: Distancia al régimen histórico."""
    mu_base = np.mean(baseline_G, axis=0)
    cov_base = np.cov(baseline_G, rowvar=False)
    cov_reg = cov_base + np.eye(cov_base.shape[0]) * 1e-5
    
    try: inv_cov = scipy.linalg.inv(cov_reg)
    except Exception: inv_cov = np.eye(cov_base.shape[0])
        
    mu_cand = np.mean(candidates_G, axis=0)
    diff = mu_cand - mu_base
    return float(np.sqrt(np.dot(np.dot(diff.T, inv_cov), diff)))

def get_ci(data):
    n = len(data)
    mean = np.mean(data)
    margin = 1.96 * np.std(data) / np.sqrt(n) if n > 0 else 0
    return [float(mean - margin), float(mean + margin)]

def main():
    parser = argparse.ArgumentParser("Auditoría de 6 Bloques: Ablación Latente MoCo")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--config_path", type=str, default="config/moco.yaml")
    parser.add_argument("--global_size", type=int, default=512)
    parser.add_argument("--local_sizes", type=int, nargs='+', default=[96, 128, 192, 224, 256])
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--baseline_json", type=str, default="")
    parser.add_argument("--out", type=str, default="audit_report_6blocks.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Cargando encoder maestro ({args.ckpt_path})...")
    model = load_encoder(args.ckpt_path, args.config_path, device)
    _, _, base_t = get_transforms(args.global_size, 128)

    all_imgs = []
    for ext in ['*.jpg', '*.png', '*.jpeg']:
        all_imgs.extend(list(Path(args.dataset_path).rglob(ext)))
        
    baseline_stats = None
    if args.baseline_json and os.path.exists(args.baseline_json):
        with open(args.baseline_json, "r") as f:
            base_report = json.load(f)
            first_conf = list(base_report["configurations"].values())[0]
            if "samples_G" in first_conf:
                baseline_stats = np.array(first_conf["samples_G"])
                print(f"Bloque F: Baseline cargado ({baseline_stats.shape[0]} imágenes).")

    print("\n[Bloque] Extrayendo Centroides de Clase...")
    centroids = compute_class_centroids(args.dataset_path, model, device, base_t)
    print(f"Centroides hallados: {list(centroids.keys())}")

    report = {"dataset": args.dataset_path, "configurations": {}}

    print("\n[Bloques A y B] Curvas de Retención Semántica (R(s) vs B(s))")
    retention_imgs = random.sample(all_imgs, min(100, len(all_imgs)))
    curves = ablation_curves(retention_imgs, model, device, centroids)
    
    print(f"{'Scale':<6} | {'R(s) Geom':<10} | {'B(s) Biol':<10}")
    print("-" * 35)
    for s in sorted(curves["R_s"].keys(), reverse=True):
        print(f"{s:<6} | {curves['R_s'][s]:<10.3f} | {curves['B_s'][s]:<10.3f}")
    report["retention_curves"] = curves

    sampled_imgs = random.sample(all_imgs, min(args.samples, len(all_imgs)))

    print(f"\n{'Crop':<5} | {'BioAcc':<6} | {'BioMrg':<6} | {'GapHlt':<6} | {'V_bio':<6} | {'d_M':<6}")
    print("-" * 55)

    for ls in args.local_sizes:
        scale = (0.10, 0.35)
        # G_matrix: [A, S, M]
        # bio_metrics: [BioAcc, BioMargin, GapHealthy, V_bio]
        G_matrix, bio_metrics = audit_images(sampled_imgs, model, device, args.global_size, ls, scale, centroids)
        
        mu_G = np.mean(G_matrix, axis=0)
        mu_B = np.mean(bio_metrics, axis=0)
        
        stats = {
            "Geom_A_mean": float(mu_G[0]), "Geom_S_mean": float(mu_G[1]), "Geom_M_mean": float(mu_G[2]),
            "Bio_Acc": float(mu_B[0]),
            "Bio_Margin": float(mu_B[1]),
            "Bio_GapHealthy": float(mu_B[2]),
            "Bio_Variance": float(mu_B[3]),
            "samples_G": G_matrix.tolist()
        }
        
        d_m_str = "-"
        if baseline_stats is not None:
            d_m = compute_mahalanobis(G_matrix, baseline_stats)
            stats["mahalanobis_distance"] = float(d_m)
            d_m_str = f"{d_m:>6.2f}"
            
        b_acc = f"{mu_B[0]:.2f}"
        b_mrg = f"{mu_B[1]:.2f}"
        b_gap = f"{mu_B[2]:.2f}"
        v_bio = f"{mu_B[3]:.3f}"
        
        print(f"{ls:<5} | {b_acc:<6} | {b_mrg:<6} | {b_gap:<6} | {v_bio:<6} | {d_m_str}")
            
        report["configurations"][f"Local_{ls}"] = stats

    with open(args.out, 'w') as f:
        json.dump(report, f, indent=4)
    print(f"\nReporte guardado en {args.out}")

if __name__ == "__main__":
    main()
