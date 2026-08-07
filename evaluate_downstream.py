import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import sys
import seaborn as sns

from models.moco import ModelBase
from engine.setup import build_eval_dataset

def get_val_transforms(eval_size: int = 384):
    """Transformaciones estándar para validación.

    BUG-12 FIX: La resolución debe coincidir con global_crop_size del encoder.
    Un encoder entrenado a 384px extrae features en una distribución diferente
    si se le pasan imágenes de 320px — degradación silenciosa de las métricas.
    Se lee desde config['moco']['global_crop_size'] en evaluate().
    """
    return transforms.Compose([
        transforms.Resize(eval_size),
        transforms.CenterCrop(eval_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def evaluate():
    parser = argparse.ArgumentParser(description="Análisis Profesional del Encoder Entrenado")
    parser.add_argument("--config", type=str, default="config/moco.yaml", help="Ruta al archivo yaml")
    args = parser.parse_args()

    # 1. Cargar Configuración
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Tipo de device para autocast (str requerido por torch.amp.autocast)
    device_type = device.type if hasattr(device, 'type') else str(device).split(':')[0]
    # BUG-12 FIX: resolución dinámica desde config — el encoder fue entrenado
    # a esta resolución; usar otra crea mismatch en el espacio latente.
    eval_size = config['moco'].get('global_crop_size', 384)
    # BUG-13 FIX: num_workers y batch_size desde config, no hardcodeados.
    n_workers  = config['training'].get('num_workers', 4)
    # BUG-AMB-10 FIX: max() forzaba eval_batch a ≥32 incluso cuando training batch_size=8
    # (fase 512px), causando OOM en GPUs con poca VRAM. engine/setup.py usa min() correctamente.
    # Usamos el mismo patrón: cap superior seguro en 64, sin exceder el batch de entrenamiento.
    eval_batch = min(config['training'].get('batch_size', 16), 64)
    use_amp    = config['training'].get('use_amp', False)
    print(f"[*] Evaluando en dispositivo: {device} | eval_size={eval_size}px | amp={use_amp}")

    # 2. Cargar Dataset de Validación
    val_dir = config['paths']['eval_val_root']
    if not os.path.exists(val_dir):
        print(f"[!] ADVERTENCIA: val_dir no encontrado en '{val_dir}'.")
        # R-4 FIX: Eliminar fallback hardcodeado con user/dataset ID específico.
        # Un path fijo como "/kaggle/input/datasets/guillermopetcho/..." rompe el script
        # en cualquier fork del proyecto o si el dataset cambia de nombre.
        # Se falla explícitamente con instrucciones claras en lugar de intentar un path
        # que con alta probabilidad no existe en el entorno del usuario.
        print(f" Error: No se encontró un directorio de validación válido en: '{val_dir}'")
        print("   Verifica 'paths.eval_val_root' en config/moco.yaml y que el dataset esté montado.")
        sys.exit(1)
    print(f"[*] Cargando dataset desde: {val_dir}")
    # Pasar el data_yaml para obtener nombres de clase reales (no dummies)
    data_yaml_path = config['paths'].get('data_yaml', None)
    # BUG-12 FIX: eval_size dinámico, no hardcodeado
    val_ds = build_eval_dataset(val_dir, transform=get_val_transforms(eval_size), data_yaml_path=data_yaml_path)
    val_loader = DataLoader(
        val_ds, batch_size=eval_batch, shuffle=False,
        num_workers=n_workers, pin_memory=True,
        persistent_workers=(n_workers > 0)
    )
    class_names = val_ds.classes
    num_classes = len(class_names)
    if any(c.startswith('Class_') for c in class_names):
        print("[!] ADVERTENCIA: Se usaron clases genéricas (Class_N). Verifica 'paths.data_yaml' en moco.yaml.")
    print(f"[*] Clases detectadas ({num_classes}): {class_names}")

    # 3. Reconstruir el Modelo
    encoder_path = config['paths']['encoder_export_path']
    head_path    = encoder_path.replace('.pth', '_head.pth')

    # Guard: verificar que los pesos existen antes de cargar
    for label, path in [("Encoder", encoder_path), ("Head lineal", head_path)]:
        if not os.path.exists(path):
            print(f"\u274c Error: {label} no encontrado en '{path}'.")
            print("   Ejecuta train.py primero para generar los pesos.")
            sys.exit(1)

    print(f"[*] Cargando Encoder: {encoder_path}")
    encoder = ModelBase(
        dim=config['moco']['dim'],
        predictor_hidden_dim=config['moco'].get('predictor_hidden_dim', 1024)
    )
    encoder.load_state_dict(torch.load(encoder_path, map_location='cpu', weights_only=True))
    encoder = encoder.to(device).eval()

    print(f"[*] Cargando Sonda Lineal (Clasificador): {head_path}")
    # BUG-12 FIX: dummy con eval_size real — no 320 hardcodeado.
    with torch.no_grad():
        dummy   = torch.randn(1, 3, eval_size, eval_size).to(device)
        raw_enc = encoder.module if hasattr(encoder, 'module') else encoder
        raw_enc = raw_enc._orig_mod if hasattr(raw_enc, '_orig_mod') else raw_enc
        if hasattr(raw_enc, 'forward_backbone'):
            sample_feat = raw_enc.forward_backbone(dummy)
        else:
            sample_feat = encoder(dummy, use_predictor=False)
        proj_dim = sample_feat.shape[-1]
    print(f"[*] Dimensión de características inferida: {proj_dim}")
    
    classifier = nn.Sequential(
        nn.LayerNorm(proj_dim),
        nn.Linear(proj_dim, num_classes)
    )
    classifier.load_state_dict(torch.load(head_path, map_location='cpu', weights_only=True))
    classifier = classifier.to(device)
    classifier.eval()

    # 4. Inferencia
    all_preds = []
    all_labels = []
    all_probs = []

    print("[*] Iniciando Inferencia sobre el conjunto de validación...")
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device, non_blocking=True)
            with torch.amp.autocast(device_type, enabled=use_amp):
                if hasattr(raw_enc, 'forward_backbone'):
                    feats = F.normalize(raw_enc.forward_backbone(x), dim=1)
                else:
                    feats = F.normalize(encoder(x, use_predictor=False), dim=1)
                logits = classifier(feats)
                probs  = F.softmax(logits, dim=1)
                preds  = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.numpy())
            all_probs.extend(probs.cpu().float().numpy())

    # Guard: dataset vacío o todos los samples fallaron
    if len(all_labels) == 0:
        print("\u274c Error: No se procesaron imágenes. Verifica que val_loader no esté vacío.")
        sys.exit(1)

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    # 5. Calcular Métricas Profesionales
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    
    print("\n" + "="*50)
    print("📊 REPORTE DE RENDIMIENTO (WEIGHTED)")
    print("="*50)
    print(f"Accuracy : {acc:.4f}  (Porcentaje total de aciertos)")
    print(f"Precision: {precision:.4f}  (Calidad de los positivos detectados)")
    print(f"Recall   : {recall:.4f}  (Capacidad de no omitir enfermos)")
    print(f"F1-Score : {f1:.4f}  (Media armónica Prec-Recall)")
    print("="*50)

    print("\n🔍 REPORTE POR CLASE:")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))

    # 6. Matriz de Confusión
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Matriz de Confusión - AranduSSL (Linear Probe)')
    plt.ylabel('Etiqueta Real')
    plt.xlabel('Predicción del Modelo')
    plt.tight_layout()
    
    cm_path = "/kaggle/working/confusion_matrix.png" if "kaggle" in val_dir else "confusion_matrix.png"
    plt.savefig(cm_path, dpi=300)
    print(f"\n Matriz de Confusión guardada en: {cm_path}")
    
    # 7. Análisis de Errores (Opcional pero muy útil)
    errores = np.sum(all_labels != all_preds)
    total = len(all_labels)
    print(f"\n Análisis Rápido: El modelo se equivocó en {errores} de {total} imágenes (Tasa de error: {(errores/total)*100:.2f}%).")
    print("\n Análisis Finalizado con Éxito.")

if __name__ == "__main__":
    evaluate()
