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
    except Exception:
        logger.warning("MoCoDataset: Fallback local counter.")
        return _ThreadSafeFallbackCounter()

# ---------------------------------------------------------------------------
# Data Augmentation (Multi-Crop Jerárquico)
# ---------------------------------------------------------------------------

def get_global_transforms(global_size=640, aug_cfg=None):
    """ 2 vistas globales de `global_size`x`global_size` para contexto foliar completo.

    Fase 1: 384px — texturas y micro-patrones.
    Fase 2: 640px — hojas completas con contexto espacial.

    Parametros de augmentation (aug_cfg, leídos del YAML):
      - hue            : Rotación de matiz. MUY BAJO (≤0.02) — el color es señal diagnóstica
                         (potassium_deficiency=amarillo, mosaic=moteado).
      - grayscale_p    : P(convertir a gris). MÍNIMO (≤0.05) — no destruir firma cromática.
      - global_blur_p  : P(GaussianBlur) en vistas globales. Simula variación de distancia.
      - global_rotation_p: P(RandomRotation ±15°). Las hojas son isótropas — rotar es válido.
    """
    aug = aug_cfg or {}
    
    # Opcion para desactivar augmentations
    disable_aug = aug.get('disable_online_augmentation', False)
    
    if disable_aug:
        # Pipeline básico sin distorsiones, solo recorte aleatorio suave para evitar colapso de MoCo
        def _make_disabled_pipeline():
            return T.Compose([
                T.RandomResizedCrop(global_size, scale=(0.8, 1.0)), # Solo recorte leve
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        return _make_disabled_pipeline(), _make_disabled_pipeline()

    hue             = aug.get('hue', 0.02)
    grayscale_p     = aug.get('grayscale_p', 0.05)
    blur_p          = aug.get('global_blur_p', 0.40)
    rotation_p      = aug.get('global_rotation_p', 0.30)

    def _make_global_pipeline():
        return T.Compose([
            T.RandomResizedCrop(global_size, scale=(0.3, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(p=0.5),
            # Rotación libre: texturas y hojas son isótropas (no hay 'arriba' canónico)
            T.RandomApply([T.RandomRotation(degrees=15)], p=rotation_p),
            # hue muy bajo para no corromper la firma cromática diagnóstica
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=hue),
            # grayscale mínimo: mosaic y potassium_deficiency son enfermedades de color
            T.RandomGrayscale(p=grayscale_p),
            # kernel proporcional a la resolución global (≈ global_size / 35, impar)
            T.RandomApply([T.GaussianBlur(kernel_size=11, sigma=(0.3, 1.5))], p=blur_p),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    return _make_global_pipeline(), _make_global_pipeline()


def get_local_transforms(local_size=128, ultra_size=96, aug_cfg=None):
    """ 4 vistas locales (`local_size`) y 1 vista ultra-local (`ultra_size`).

    Fase 1: 96px / 72px — micro-lesiones en texturas.
    Fase 2: 128px / 96px — región foliar con contexto sobre hoja completa.

    Parametros de augmentation (aug_cfg, leídos del YAML):
      - hue          : Más restrictivo que global (≤0.01). Los crops locales amplían
                       lesiones pequeñas donde el color es aún más discriminativo.
      - grayscale_p  : Igual que global (≤0.05).
      - local_blur_p : P(GaussianBlur) en vistas locales. Kernel chico para no destruir
                       micro-texturas de frog_eye y mosaic.
    """
    aug = aug_cfg or {}
    
    # Opcion para desactivar augmentations
    disable_aug = aug.get('disable_online_augmentation', False)
    
    if disable_aug:
        # Pipeline básico sin distorsiones para crops
        def _make_disabled_local(size, scale):
            return T.Compose([
                T.RandomResizedCrop(size, scale=scale), # Es necesario para generar las vistas
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        t_local = [_make_disabled_local(local_size, (0.08, 0.35)) for _ in range(4)]
        t_ultra = [_make_disabled_local(ultra_size, (0.04, 0.12)) for _ in range(1)]
        return t_local + t_ultra

    hue         = aug.get('hue', 0.01)
    grayscale_p = aug.get('grayscale_p', 0.05)
    blur_p      = aug.get('local_blur_p', 0.30)

    def _make_local_pipeline(size, scale):
        # kernel debe ser impar y proporcional al crop (≈ size / 13, mín 3)
        k = max(3, (size // 13) | 1)  # '| 1' garantiza impar
        return T.Compose([
            T.RandomResizedCrop(size, scale=scale),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(p=0.5),
            # contrast más alto para resaltar bordes de lesiones en crops pequeños
            T.ColorJitter(brightness=0.3, contrast=0.5, saturation=0.3, hue=hue),
            T.RandomGrayscale(p=grayscale_p),
            # kernel dinámico proporcional al tamaño del crop
            T.RandomApply([T.GaussianBlur(kernel_size=k, sigma=(0.1, 1.0))], p=blur_p),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    # 4 vistas del tamaño local principal (128px en Fase 2, 96px en Fase 1)
    t_local = [_make_local_pipeline(local_size, (0.08, 0.35)) for _ in range(4)]
    # 1 vista ultra-local para micro-patrones (96px en Fase 2, 72px en Fase 1)
    t_ultra = [_make_local_pipeline(ultra_size, (0.04, 0.12)) for _ in range(1)]

    return t_local + t_ultra

# ---------------------------------------------------------------------------
# Dataset e Índices
# ---------------------------------------------------------------------------

class MoCoDataset(Dataset):
    def __init__(self, paths, moco_config=None):
        self.paths = paths
        cfg = moco_config or {}
        # Resolución config-driven: Fase 1=384px, Fase 2=640px
        global_size = cfg.get('global_crop_size', 640)
        local_size  = cfg.get('local_crop_size', 128)
        ultra_size  = int(local_size * 0.75)  # Ultra = 75% del tamaño local
        # Parámetros de augmentation: leídos del bloque 'augmentation:' del YAML
        aug_cfg = cfg.get('augmentation', {})
        self.t_q, self.t_k = get_global_transforms(global_size=global_size, aug_cfg=aug_cfg)
        self.local_transforms = get_local_transforms(local_size=local_size, ultra_size=ultra_size, aug_cfg=aug_cfg)
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
        # M-7 FIX: El mensaje anterior decía "100 veces" hardcodeado, pero max_retries
        # es min(100, len(self.paths)). Con datasets pequeños el error era engañoso.
        raise RuntimeError(f"MoCoDataset falló {max_retries} veces. Último error: {last_err}")

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
            # R-3 FIX: Escanear todas las entradas con rglob("*") y filtrar por extensión
            # en minúsculas. El método anterior usaba 6 patrones fijos que no cubrían
            # variantes mixtas (.Jpg, .Jpeg) ni formatos adicionales (.webp, .bmp, .tiff).
            _IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}
            files = sorted([str(f) for f in Path(root).rglob("*") if f.suffix.lower() in _IMG_EXTS])
            if len(files) == 0:
                raise RuntimeError(f"Sin imágenes en {root}")
            np.save(cache_path, files)
            _logger.info(f"📁 Índice creado con {len(files)} imágenes.")
    
    if is_dist:
        # BUG-C3 FIX: device_ids solo es válido para el backend 'nccl' con CUDA.
        # En CPU (backend 'gloo') o en entornos de test, dist.barrier(device_ids=...)
        # lanza RuntimeError o provoca un deadlock silencioso.
        # Detectamos el backend activo antes de pasar device_ids.
        _backend = dist.get_backend() if dist.is_initialized() else ""
        if torch.cuda.is_available() and _backend == "nccl":
            dist.barrier(device_ids=[torch.cuda.current_device()])
        else:
            dist.barrier()
    
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
        
        # HIGH-4 FIX: z_norm (pre-normalización) es la señal diagnóstica correcta para
        # detectar colapso del projector (z_norm → 0). Se usa SIEMPRE como norma reportada,
        # incluso cuando use_predictor=True, porque p_norm (norma del predictor) es engañosa:
        # el predictor recibe z ya L2-normalizado, así que su norma de salida solo refleja
        # la escala de los pesos del predictor, no la salud del espacio latente.
        z_norm = z.norm(dim=1).mean()
        z = F.normalize(z.float(), dim=1, eps=1e-6).to(z.dtype)
        
        if use_predictor:
            p = self.predictor(z) # Shape: [B, 512]
            p = F.normalize(p.float(), dim=1, eps=1e-6).to(p.dtype)
            if return_norm: return p, z_norm  # HIGH-4 FIX: z_norm, no p_norm
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
        keys = F.normalize(keys, dim=1, eps=1e-6)
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
                # C-4 FIX: dist.broadcast requiere que el tensor esté en CUDA.
                # Sin el guard de device, falla con RuntimeError críptico en entornos
                # CPU (tests, CI) y puede causar deadlock en multi-nodo si la queue
                # está en un device inesperado. El broadcast solo tiene sentido en GPU.
                if (dist.is_available() and dist.is_initialized()
                        and self.queue.device.type != 'cpu'):
                    dist.broadcast(self.queue, src=0)