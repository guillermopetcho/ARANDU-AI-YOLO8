#!/bin/bash
# ==============================================================================
# Lanzador Nativo de AnyLabeling Oficial Personalizable (PyQt Desktop)
# ==============================================================================

PROJECT_DIR="/home/ama-gi/Documentos/IAR/SojAI/code/ENCODER_YOLO"
ANYLABELING_DIR="$PROJECT_DIR/anylabeling_upstream"

cd "$PROJECT_DIR" || exit 1

export DISPLAY="${DISPLAY:-:0}"
export PYTHONPATH="$ANYLABELING_DIR:$PYTHONPATH"

# Ejecución nativa de la aplicación de escritorio AnyLabeling
python3 "$ANYLABELING_DIR/anylabeling/app.py" "$@"
