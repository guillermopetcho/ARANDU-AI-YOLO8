#!/bin/bash
# ==============================================================================
# Pipeline 100% Automatizado: Preparación -> Entrenamiento -> Auto-Etiquetado AnyLabeling
# ==============================================================================

set -e

PROJECT_DIR="/home/ama-gi/Documentos/IAR/SojAI/code/ENCODER_YOLO"
cd "$PROJECT_DIR"

IMAGE_DIR="${1:-$PROJECT_DIR/sample_images}"
MODEL_PATH="${2:-$PROJECT_DIR/runs/segment/train/weights/best.pt}"

echo "=================================================================="
echo "🤖 PIPELINE AUTOMATIZADO DE AUTO-ETIQUETADO Y SEGMENTACIÓN"
echo "=================================================================="
echo "Carpeta Objetivo: $IMAGE_DIR"
echo "Modelo YOLO-Seg: $MODEL_PATH"
echo "=================================================================="

# 1. Si no existe un modelo personalizado entrenado, descargar modelo YOLOv8-Seg base
if [ ! -f "$MODEL_PATH" ]; then
    echo "⚠️ No se encontró modelo entrenado en $MODEL_PATH. Usando yolov8n-seg.pt base..."
    MODEL_PATH="yolov8n-seg.pt"
fi

# 2. Ejecutar auto-segmentación masiva en la carpeta objetivo
echo "🚀 Ejecutando Auto-Segmentación Masiva..."
python3 auto_label_folder.py --image-dir "$IMAGE_DIR" --model "$MODEL_PATH" --conf 0.25

# 3. Lanzar AnyLabeling Oficial listo con los archivos auto-segmentados
echo "🎉 ¡Auto-Etiquetado completado! Lanzando AnyLabeling Oficial..."
export PYTHONPATH="$PROJECT_DIR/anylabeling_upstream:$PYTHONPATH"
python3 "$PROJECT_DIR/anylabeling_upstream/anylabeling/app.py" "$IMAGE_DIR"
