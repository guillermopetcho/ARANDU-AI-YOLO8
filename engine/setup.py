"""engine/setup.py — Funciones de inicialización y setup.

Extraído de train.py para modularizar la construcción de componentes.
"""

import os
import copy
import logging
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms as T
from PIL import Image
import glob

from models.moco import build_index, MoCoDataset, ModelBase, MoCoQueue

class YOLOClassificationDataset(torch.utils.data.Dataset):
    """
    Dataset wrapper que permite a las métricas SSL (KNN, Linear Probe) evaluar sobre 
    un dataset en formato YOLO (carpetas 'images' y 'labels'). Toma la clase del 
    primer bounding box de cada archivo .txt como etiqueta global de la imagen.

    Las clases reales se infieren desde el data.yaml del dataset (campo 'names').
    Si no se puede cargar el YAML, se genera un fallback dinámico basado en los IDs
    reales encontrados en los archivos de etiquetas.
    """
    def __init__(self, root, transform=None, data_yaml_path=None):
        self.root = root
        self.transform = transform
        self.images_dir = os.path.join(root, "images")
        self.labels_dir = os.path.join(root, "labels")
        
        self.image_files = []
        for ext in ('*.jpg', '*.jpeg', '*.png'):
            self.image_files.extend(glob.glob(os.path.join(self.images_dir, ext)))

        # --- Inferencia de clases reales desde data.yaml ---
        self.classes = self._load_class_names(data_yaml_path)
        
    def _load_class_names(self, data_yaml_path):
        """Intenta cargar los nombres de clase reales desde el data.yaml del dataset YOLO.
        
        Estrategia de búsqueda:
          1. Path explícito pasado por el caller.
          2. Autodescubrimiento: busca data.yaml en el directorio padre (../data.yaml).
          3. Fallback dinámico: escanea los .txt de labels para inferir los IDs presentes.
        """
        # L3 FIX: import yaml movido al top del módulo (ya no se importa aquí).
        # Estrategia 1: path explícito
        candidates = []
        if data_yaml_path and os.path.isfile(data_yaml_path):
            candidates.append(data_yaml_path)
            
        # Estrategia 2: autodescubrimiento en el directorio padre
        parent_dir = os.path.dirname(self.root.rstrip('/'))
        for name in ('data.yaml', 'dataset.yaml'):
            candidate = os.path.join(parent_dir, name)
            if os.path.isfile(candidate):
                candidates.append(candidate)
                
        for yaml_path in candidates:
            try:
                with open(yaml_path, 'r') as f:
                    cfg = yaml.safe_load(f)
                names = cfg.get('names', {})
                if isinstance(names, dict):
                    # Formato: {0: 'Clase_A', 1: 'Clase_B', ...}
                    return [names[i] for i in sorted(names.keys())]
                elif isinstance(names, list):
                    # Formato: ['Clase_A', 'Clase_B', ...]
                    return names
            except Exception:
                pass  # Intentar el siguiente candidato
                
        # Estrategia 3: fallback dinámico escaneando labels
        found_ids = set()
        for lf in glob.glob(os.path.join(self.labels_dir, '*.txt'))[:200]:  # cap para velocidad
            try:
                with open(lf) as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            found_ids.add(int(parts[0]))
            except Exception:
                pass
                
        if found_ids:
            max_id = max(found_ids)
            return [f"Class_{i}" for i in range(max_id + 1)]
            
        # Último recurso: lista genérica pequeña (no 100 que da falsos num_classes)
        return [f"Class_{i}" for i in range(10)]
        
    def __len__(self):
        return len(self.image_files)
        
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
            
        label_path = os.path.join(self.labels_dir, os.path.splitext(os.path.basename(img_path))[0] + ".txt")
        class_id = 0 # Fallback
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                line = f.readline().strip()
                if line:
                    # El formato YOLO es: <class_id> <x> <y> <w> <h>
                    raw_id = int(line.split()[0])
                    # M2 FIX: clampear al rango válido para evitar IndexError en CrossEntropyLoss
                    # si el dataset tiene IDs que superan el número de clases configuradas.
                    class_id = min(raw_id, len(self.classes) - 1)
                    
        return img, class_id

def build_eval_dataset(root, transform, data_yaml_path=None):
    """Auto-detecta el formato del dataset (YOLO o ImageFolder) y devuelve el wrapper correcto.
    
    Args:
        root: Directorio raíz del dataset (split 'train' o 'valid').
        transform: Transformaciones torchvision a aplicar.
        data_yaml_path: Ruta opcional al data.yaml de YOLO para obtener nombres de clase reales.
    """
    if os.path.isdir(os.path.join(root, "images")) and os.path.isdir(os.path.join(root, "labels")):
        return YOLOClassificationDataset(root, transform, data_yaml_path=data_yaml_path)
    else:
        return ImageFolder(root, transform=transform)


def resolve_kaggle_paths(paths_config, rank=0):
    """Auto-descubre la ubicación real del dataset en Kaggle."""
    logger = logging.getLogger("AranduSSL")
    dataset_root = paths_config.get("dataset_root", "")

    if os.path.isdir(dataset_root):
        return paths_config
    if not os.path.isdir("/kaggle/input"):
        return paths_config

    dataset_folder_name = os.path.basename(dataset_root)
    if not dataset_folder_name:
        return paths_config

    found = None
    for dirpath, dirnames, _ in os.walk("/kaggle/input"):
        if dataset_folder_name in dirnames:
            found = os.path.join(dirpath, dataset_folder_name)
            break

    if found is None:
        path_parts = dataset_root.rstrip("/").split("/")
        if len(path_parts) >= 2:
            dataset_slug = path_parts[-2]
            for dirpath, dirnames, _ in os.walk("/kaggle/input"):
                if dataset_slug in dirnames:
                    candidate = os.path.join(dirpath, dataset_slug, dataset_folder_name)
                    if os.path.isdir(candidate):
                        found = candidate
                        break

    if found is None:
        if rank == 0:
            logger.warning(f"⚠️ No se encontró '{dataset_folder_name}' bajo /kaggle/input/. "
                           f"Path original: {dataset_root}")
        return paths_config

    old_root = dataset_root
    new_root = found
    if rank == 0:
        logger.info(f"🔍 Auto-discovery: dataset encontrado en {new_root}")
        logger.info(f"   (path original del config: {old_root})")

    patched = {}
    for key, value in paths_config.items():
        if isinstance(value, str) and old_root in value:
            patched[key] = value.replace(old_root, new_root)
        else:
            patched[key] = value

    return patched


def make_eval_subset_loader(eval_ds, subset_size: int, num_workers: int) -> DataLoader:
    """Crea un DataLoader con un subconjunto aleatorio del dataset de evaluación.

    Cada llamada genera un nuevo subconjunto independiente, permitiendo
    rerandomizar periódicamente y obtener estimaciones no sesgadas de KNN.
    """
    indices = torch.randperm(len(eval_ds))[:min(subset_size, len(eval_ds))].tolist()
    return DataLoader(
        Subset(eval_ds, indices), batch_size=128,
        num_workers=num_workers, pin_memory=True
    )


def build_dataloaders(CONFIG, is_distributed, rank):
    paths = build_index(CONFIG["paths"]["dataset_root"], rank, CONFIG["paths"]["index_cache_path"])
    dataset = MoCoDataset(paths, moco_config=CONFIG["moco"])
    sampler = DistributedSampler(dataset, shuffle=True, drop_last=True) if is_distributed else None

    n_workers = CONFIG["training"]["num_workers"]
    train_loader = DataLoader(
        dataset, batch_size=CONFIG["training"]["batch_size"], shuffle=(sampler is None),
        sampler=sampler, num_workers=n_workers,
        pin_memory=True, drop_last=True,
        persistent_workers=(n_workers > 0),
        prefetch_factor=2 if n_workers > 0 else None
    )

    # Resolución de eval: match cercano a la resolución de entrenamiento
    eval_size = CONFIG["moco"].get("global_crop_size", 384)
    eval_transform = T.Compose([
        T.Resize(eval_size), T.CenterCrop(eval_size), T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    # Pasar el data_yaml para que YOLOClassificationDataset lea las clases reales
    data_yaml = CONFIG["paths"].get("data_yaml", None)
    eval_ds = build_eval_dataset(CONFIG["paths"]["eval_train_root"], transform=eval_transform, data_yaml_path=data_yaml)
    val_ds = build_eval_dataset(CONFIG["paths"]["eval_val_root"], transform=eval_transform, data_yaml_path=data_yaml)

    eval_workers = min(2, n_workers)
    # C3 FIX: Usar make_eval_subset_loader() para permitir rerandomización periódica.
    eval_train_loader = make_eval_subset_loader(eval_ds, CONFIG["eval"]["subset_size"], eval_workers)
    eval_val_loader = DataLoader(val_ds, batch_size=128, num_workers=eval_workers, pin_memory=True)

    return train_loader, eval_train_loader, eval_val_loader, eval_ds, val_ds


def build_model(CONFIG, is_distributed, device, rank, local_rank):
    logger = logging.getLogger("AranduSSL")
    model_base_raw = ModelBase(
        dim=CONFIG["moco"]["dim"],
        # B-PRED FIX: default cambiado a 1024 (coherente con el YAML corregido).
        predictor_hidden_dim=CONFIG["moco"].get("predictor_hidden_dim", 1024)
    ).to(device, memory_format=torch.channels_last)

    # B3 FIX: SyncBatchNorm no tiene efecto sobre ModelBase (usa LayerNorm, no BN).
    # Se elimina la llamada innecesaria para evitar confusión y posibles conflictos
    # si en el futuro se añaden capas BN explícitamente.
    # Si el modelo volviera a usar BN, descomentar la siguiente línea:
    # if is_distributed:
    #     model_base_raw = nn.SyncBatchNorm.convert_sync_batchnorm(model_base_raw)

    model_q = copy.deepcopy(model_base_raw)
    model_k = copy.deepcopy(model_base_raw).to(device, memory_format=torch.channels_last)

    # B-COMPILE FIX: torch.compile controlado por config, desactivado por defecto.
    # Activarlo solo en el segundo run (post-validación de estabilidad).
    # Para activar: añadir `use_compile: true` en la sección `training:` del YAML.
    use_compile = CONFIG.get("training", {}).get("use_compile", False)
    is_compiled = False
    if use_compile and hasattr(torch, "compile"):
        try:
            model_q = torch.compile(model_q, mode="reduce-overhead", dynamic=True)
            is_compiled = True
            if rank == 0: logger.info("⚡ torch.compile activado (reduce-overhead).")
        except Exception as e:
            if rank == 0: logger.warning(f"⚠️ torch.compile falló: {e}. Continuando sin compilar.")
    elif rank == 0 and not use_compile:
        logger.info("🔒 torch.compile desactivado (use_compile=false en config).")

    model_k.eval()
    for p in model_k.parameters(): p.requires_grad = False
    # Nota: El guard de BatchNorm se mantiene por retrocompatibilidad con checkpoints
    # anteriores que podían contener capas BN.
    for m in model_k.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.eval()
            m.track_running_stats = False

    # === FREEZE PARCIAL DEL BACKBONE ===
    # freeze_stages=2 congela las 2 primeras etapas del ConvNeXt (baja frecuencia).
    # Las etapas profundas (semántica) siguen entrenando con el LR bajo.
    freeze_stages = CONFIG["training"].get("freeze_stages", 0)
    if freeze_stages > 0:
        # Acceder al encoder real (desenvuelto de compile si aplica)
        raw_encoder = model_q._orig_mod.encoder if hasattr(model_q, '_orig_mod') else model_q.encoder
        # ConvNeXt en timm tiene: stem + stages[0..3]
        modules_to_freeze = []
        if hasattr(raw_encoder, 'stem'):
            modules_to_freeze.append(raw_encoder.stem)
        if hasattr(raw_encoder, 'stages'):
            for i in range(min(freeze_stages, len(raw_encoder.stages))):
                modules_to_freeze.append(raw_encoder.stages[i])
        frozen_params = 0
        for module in modules_to_freeze:
            for p in module.parameters():
                p.requires_grad = False
                frozen_params += p.numel()
        if rank == 0:
            logger.info(f"🧊 Freeze parcial: {len(modules_to_freeze)} módulos, "
                        f"{frozen_params/1e6:.1f}M parámetros congelados "
                        f"(freeze_stages={freeze_stages}).")

    if is_distributed: model_q = nn.parallel.DistributedDataParallel(model_q, device_ids=[local_rank])
    queue = MoCoQueue(dim=CONFIG["moco"]["dim"], K=CONFIG["moco"]["queue"]).to(device)

    return model_q, model_k, queue, is_compiled
