import os
import multiprocessing
import multiprocessing.context
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import timm
from torchvision import transforms as T
from torch.utils.data import Dataset
from PIL import Image
import random
import numpy as np
from pathlib import Path
import logging
import threading
import time

from utils.distributed import concat_all_gather

# ---------------------------------------------------------------------------
# Contadores y Helpers
# ---------------------------------------------------------------------------

class _ThreadSafeFallbackCounter:
    def __init__(self):
        self.value = 0
        self._lock = threading.Lock()

    def get_lock(self):
        return self._lock

def _make_shared_counter():
    logger = logging.getLogger("AranduSSL")
    try:
        import torch.multiprocessing as tmp
        start_method = tmp.get_start_method(allow_none=True) or 'fork'
        ctx = multiprocessing.get_context(start_method)
        counter = ctx.Value('i', 0)
        return counter
    except Exception as e:
        logger.warning("MoCoDataset: Fallback local counter.")
        return _ThreadSafeFallbackCounter()

# ---------------------------------------------------------------------------
# Data Augmentation (Multi-Crop Jerárquico)
# ---------------------------------------------------------------------------

def get_global_transforms():
    """ 2 vistas globales de 384x384 para estructura foliar completa. """
    # B2 FIX: crear instancias INDEPENDIENTES de transforms para t_q y t_k.
    # ColorJitter y GaussianBlur son stateful — compartir instancias entre t_q/t_k
    # puede causar que ambas vistas reciban los mismos parámetros aleatorios.
    def _make_global_pipeline():
        return T.Compose([
            T.RandomResizedCrop(384, scale=(0.65, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(p=0.5),
            T.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.06, hue=0.02),
            T.RandomGrayscale(p=0.01),
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    return _make_global_pipeline(), _make_global_pipeline()

def get_local_transforms():
    """ 4 vistas locales (96x96) y 2 vistas ultra-locales (64x64). """
    
    # B2 FIX: cada llamada crea nuevas instancias independientes.
    def _make_local_pipeline(size, scale):
        return T.Compose([
            T.RandomResizedCrop(size, scale=scale),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(p=0.5),
            T.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.06, hue=0.02),
            T.RandomGrayscale(p=0.01),
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    # 4 vistas de 96x96 — capturan estructuras intermedias (bordes, manchas)
    t_96 = [_make_local_pipeline(96, (0.08, 0.25)) for _ in range(4)]
    # 2 vistas de 64x64 — capturan micro-lesiones y texturas ultrafinas
    t_64 = [_make_local_pipeline(64, (0.02, 0.08)) for _ in range(2)]
    
    return t_96 + t_64

# ---------------------------------------------------------------------------
# Dataset e Índices
# ---------------------------------------------------------------------------

class MoCoDataset(Dataset):
    def __init__(self, paths, moco_config=None):
        self.paths = paths
        self.t_q, self.t_k = get_global_transforms()
        self.local_transforms = get_local_transforms()
        self._load_errors = _make_shared_counter()
        self.logger = logging.getLogger("AranduSSL")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        max_retries = min(100, len(self.paths))
        last_err = None
        for attempt in range(max_retries):
            try:
                img = Image.open(self.paths[idx]).convert("RGB")
                v_q = self.t_q(img)
                v_k = self.t_k(img)
                
                # Lista de tensores en lugar de un tensor apilado (torch.stack)
                # PyTorch `default_collate` agrupará esto en una lista de Batches en el DataLoader
                locals_ = [t(img) for t in self.local_transforms]
                
                return v_q, v_k, locals_
            except Exception as e:
                last_err = e
                with self._load_errors.get_lock():
                    self._load_errors.value += 1
                if len(self.paths) == 0:
                    raise RuntimeError("MoCoDataset está vacío.")
                idx = random.randint(0, len(self.paths) - 1)
        raise RuntimeError(f"MoCoDataset falló 100 veces. Último error: {last_err}")

def build_index(root, rank, cache_path):
    is_dist = dist.is_available() and dist.is_initialized()
    
    if rank == 0:
        rebuild = True
        if os.path.exists(cache_path):
            try:
                cached_files = np.load(cache_path, allow_pickle=True).tolist()
                if len(cached_files) > 0 and cached_files[0].startswith(root) and os.path.exists(cached_files[0]):
                    rebuild = False
            except Exception:
                pass
                
        if rebuild:
            _logger = logging.getLogger("AranduSSL")
            _logger.info(f"📂 Escaneando {root}...")
            files = sorted([str(f) for ext in ["*.jpg", "*.png", "*.jpeg", "*.JPG", "*.PNG", "*.JPEG"] for f in Path(root).rglob(ext)])
            if len(files) == 0:
                raise RuntimeError(f"Sin imágenes en {root}")
            np.save(cache_path, files)
            _logger.info(f"📁 Índice creado con {len(files)} imágenes.")
    
    if is_dist:
        dist.barrier(device_ids=[torch.cuda.current_device()])
    
    if not os.path.exists(cache_path):
        raise RuntimeError(f"Cache {cache_path} no existe.")
        
    return np.load(cache_path, allow_pickle=True).tolist()

# ---------------------------------------------------------------------------
# Arquitectura (ConvNeXt V2 Tiny + MoCo v3)
# ---------------------------------------------------------------------------

class ModelBase(nn.Module):
    """
    Backbone ConvNeXt V2 Tiny + Projector MLP + Predictor MLP (MoCo v3).
    """
    def __init__(self, dim=512, predictor_hidden_dim=1024):
        super().__init__()
        
        # Encoder: ConvNeXt V2 Tiny (num_classes=0 remueve el clasificador y aplica Global Average Pooling)
        # Salida garantizada: 768 canales.
        self.encoder = timm.create_model(
            "convnextv2_tiny", 
            pretrained=True, 
            num_classes=0
        )
        self.encoder.set_grad_checkpointing(enable=True)
        
        # Projector: 768 -> 2048 -> 512
        self.projector = nn.Sequential(
            nn.Linear(768, 2048),
            nn.LayerNorm(2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, dim),
            nn.LayerNorm(dim, elementwise_affine=False)
        )
        
        # Predictor MLP (MoCo v3 asimétrico): 512 -> 1024 -> 512
        self.predictor = nn.Sequential(
            nn.Linear(dim, predictor_hidden_dim),
            nn.LayerNorm(predictor_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(predictor_hidden_dim, dim)
        )

    def forward(self, x, use_predictor=False, return_norm=False):
        h = self.encoder(x)  # Shape: [B, 768]
        z = self.projector(h) # Shape: [B, 512]
        
        z_norm = z.norm(dim=1).mean()
        z = F.normalize(z.float(), dim=1).to(z.dtype)
        
        if use_predictor:
            p = self.predictor(z) # Shape: [B, 512]
            p_norm = p.norm(dim=1).mean()
            p = F.normalize(p.float(), dim=1).to(p.dtype)
            if return_norm: return p, p_norm
            return p
        
        if return_norm: return z, z_norm
        return z

class MoCoQueue(nn.Module):
    """
    Cola FIFO para representaciones negativas. K por defecto a 16384.
    """
    def __init__(self, dim=512, K=16384):
        super().__init__()
        self.K = K
        self.register_buffer("queue", F.normalize(torch.randn(dim, K), dim=0))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def enqueue_dequeue(self, keys, step=None):
        keys = concat_all_gather(keys.detach())
        keys = F.normalize(keys, dim=1)
        batch_size = keys.shape[0]
        
        if batch_size > self.K:
            keys = keys[:self.K]
            batch_size = self.K
            
        ptr = int(self.queue_ptr)
        end_ptr = ptr + batch_size
        
        if end_ptr <= self.K:
            self.queue[:, ptr:end_ptr].copy_(keys.T)
        else:
            first_part = self.K - ptr
            self.queue[:, ptr:].copy_(keys[:first_part].T)
            self.queue[:, :batch_size - first_part].copy_(keys[first_part:].T)
            
        self.queue_ptr[0] = (ptr + batch_size) % self.K
        
        if step is not None and step % 500 == 0:
            with torch.no_grad():
                self.queue.copy_(F.normalize(self.queue, dim=0))
                if dist.is_available() and dist.is_initialized():
                    dist.broadcast(self.queue, src=0)