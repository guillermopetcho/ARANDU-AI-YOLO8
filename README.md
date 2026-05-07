# 🌿 ARANDU-AI — Detector de Enfermedades en Soja (SSL + YOLO)

Pipeline de investigación de grado producción para la detección de enfermedades foliares en soja, combinando aprendizaje auto-supervisado (MoCo v3) con detección de objetos (YOLOv8).

---

## Descripción

El sistema tiene dos etapas:

1. **AranduSSL** — Preentrenamiento auto-supervisado del backbone sobre imágenes de soja sin etiquetas, usando MoCo v3 con Multi-Crop DINO-style y un controlador adaptativo (AAC v3) que regula temperatura, momentum y LR en función del estado del espacio latente.

2. **ARANDU-YOLO** — Detector YOLOv8 que reemplaza su backbone estándar por el encoder MoCo preentrenado, conectado al neck PAN-FPN mediante adaptadores espaciales con **Context Gate** aprendible por píxel.

---

## Estructura del Proyecto

```
ENCODER_YOLO/
├── config/moco.yaml            ← Hiperparámetros y paths (Kaggle)
├── engine/
│   ├── trainer.py              ← Loop MoCo (pérdida simétrica + Multi-Crop)
│   ├── controller.py           ← AAC v3: PID 3-eje + GeoSat (detector saturación)
│   ├── checkpoint.py           ← Guardado atómico / carga con weights_only=True
│   ├── setup.py                ← Construcción de dataloaders y modelos
│   ├── loop.py                 ← handle_evaluation / handle_rollback
│   └── scheduler.py            ← Cosine warmup + momentum_update
├── models/
│   ├── moco.py                 ← ModelBase (ResNet50+Projector+Predictor), MoCoQueue, MoCoDataset
│   └── yolo_wrapper.py         ← AranduBackbone (P3/P4/P5) + SpatialFeatureAdapter + Context Gate
├── evaluation/
│   ├── knn.py                  ← KNN con FAISS GPU (fallback CPU → sklearn)
│   └── linear_probe.py         ← Linear Probe con AMP sobre representaciones de 256-dim
├── utils/
│   ├── metrics.py              ← Alignment, Uniformity (logsumexp), Cosine Sims, Welford stats
│   ├── distributed.py          ← Helpers DDP (concat_all_gather, batch_shuffle)
│   ├── visualize_alpha.py      ← Heatmaps del Context Gate sobre imagen individual
│   ├── evaluate_alpha_dataset.py ← Evaluación cuantitativa de α (Cohen's d, Pearson)
│   └── corrupt_dataset.py      ← Generador de imágenes corruptas para stress-test
├── tests/test_scheduler.py     ← Tests del scheduler
├── ablation_runner.py          ← Orquestador del estudio de ablación (3 modelos)
├── arandu_yolov8.yaml          ← Arquitectura YOLO con AranduBackbone
└── train.py                    ← Entry point del pipeline SSL
```

---

## Uso

### 1. Preentrenamiento SSL (AranduSSL)

```bash
# Single GPU
python train.py

# Multi-GPU (DDP)
torchrun --nproc_per_node=2 train.py
```

Configura los paths del dataset y checkpoints en `config/moco.yaml` antes de ejecutar.

**Modos de entrenamiento** (controlado por `exploitation_mode` en el config):
- `exploitation_mode: False` — Modo exploratorio con AAC v3 activo (regula τ, momentum y LR adaptativamente).
- `exploitation_mode: True` — Modo fine-tuning: AAC desactivado, temperatura y momentum fijos.

### 2. Estudio de Ablación (3 Modelos)

```bash
python ablation_runner.py \
    --data dataset_soja.yaml \
    --moco-ckpt /kaggle/working/moco_encoder_ready.pth \
    --epochs 100 --batch 16 --imgsz 640
```

Entrena secuencialmente con seed=42 idéntico:
- **M1 — Baseline:** YOLOv8n puro
- **M2 — Híbrido Rígido:** AranduBackbone sin Context Gate (α fijo en 0.5)
- **M3 — Híbrido Adaptativo:** AranduBackbone con Context Gate learnable ← objetivo principal

### 3. Análisis del Context Gate

**Visualización sobre una imagen:**
```bash
python utils/visualize_alpha.py \
    --image test_hoja.jpg \
    --model runs/detect/Model3_ContextGate/weights/best.pt \
    --output alpha_heatmaps.png
```

**Evaluación cuantitativa sobre el dataset de validación:**
```bash
python utils/evaluate_alpha_dataset.py \
    --model runs/detect/Model3_ContextGate/weights/best.pt \
    --data dataset_soja.yaml
```
Genera reporte de separación BG/GT (Cohen's d) y correlación Tamaño vs α (Pearson) por escala P3/P4/P5.

---

## Arquitectura: Context Gate

El `SpatialFeatureAdapter` aplica una combinación convexa por píxel entre el shortcut (features crudas) y las features localizadas:

```
output = α · shortcut + (1 - α) · local_feat
```

donde `α ∈ (0,1)` es predicho por una convolución 1×1 + Sigmoid inicializada a 0 (→ α=0.5 uniforme al inicio).

**Fases de entrenamiento del AranduBackbone:**

| Fase | Backbone | BNs del backbone | Adaptadores |
|------|----------|-----------------|-------------|
| 1    | Congelado | eval() | Entrenables |
| 2    | layer3 + layer4 libres | layer3/4 en train() | Entrenables |
| 3    | Todo libre | train() | Entrenables |

---

## Controlador Adaptativo (AAC v3)

El `TrainingController` regula 3 hiperparámetros en tiempo real basándose en el estado del espacio latente:

| Eje | Variable | Señal | Rango |
|-----|----------|-------|-------|
| A — Termostato | Temperatura τ | Uniformidad vs -1.8 | [0.05, 0.25] |
| B — Freno | Momentum EMA m | Drift de μ vs 5% | [0.95, 0.99999] |
| C — Amortiguador | Escala LR | Varianza de rango efectivo | [0.25, 1.0] |

**Acciones posibles:** `CONTINUE` / `ROLLBACK` (restaura best checkpoint) / `EARLY_STOP`.

---

## Configuración Principal (`config/moco.yaml`)

| Parámetro | Valor por defecto | Descripción |
|-----------|-------------------|-------------|
| `exploitation_mode` | `True` | Desactiva AAC para fine-tuning |
| `dim` | 256 | Dimensión del espacio latente |
| `queue` | 16384 | Buffer de negativos MoCo |
| `num_local_crops` | 4 | Vistas locales Multi-Crop (0 = desactivar) |
| `batch_size` | 48 | Batch por GPU |
| `grad_accum_steps` | 4 | Acumulación de gradientes |
| `knn_k` | 20 | Vecinos KNN para evaluación |

---

## Requisitos

```
torch >= 2.0
torchvision
ultralytics
faiss-gpu  # o faiss-cpu como fallback
scikit-learn
tqdm
wandb      # opcional
pyyaml
pillow
opencv-python
```

Instalar:
```bash
pip install -r requirements.txt
```

---

## Cómo interpretar las métricas SSL

| Métrica | Rango saludable | Señal de alerta |
|---------|----------------|-----------------|
| **Uniformity** | [-2.5, -1.2] (más negativo = mejor) | > -0.3 → colapso |
| **Alignment** | [0.0, 0.3] | > 0.5 → repulsión de positivos |
| **pos_sim** | [0.7, 0.99] | < 0.5 → embeddings no alineados |
| **neg_sim** | [0.0, 0.3] | > 0.5 → colapso de representaciones |
| **GeoSat Score** | Decreasing over time | Plateau o spike → saturación |
