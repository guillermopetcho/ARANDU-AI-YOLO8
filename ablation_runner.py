import os
from ultralytics import YOLO
import ultralytics.nn.modules as nn_modules
import sys
from models.yolo_wrapper import AranduBackbone

def run_ablation(data_yaml, moco_ckpt, epochs=100, batch=16, imgsz=640):
    """
    Orquestador automático para el Día 1 del Estudio de Ablación.
    Garantiza Identidad Experimental: misma seed, mismo batch, mismo dataset.
    """
    
    # =========================================================
    # MODELO 1: BASELINE YOLOv8 (O YOLO26)
    # =========================================================
    print("\n" + "="*50)
    print("🚀 INICIANDO MODELO 1: BASELINE PURO")
    print("="*50)
    # Usar yolov8n.yaml o yolo26.yaml dependiendo de tu versión
    model1 = YOLO("yolov8n.yaml") 
    model1.train(
        data=data_yaml, 
        epochs=epochs, 
        imgsz=imgsz, 
        batch=batch, 
        project="Ablation_SojAI", 
        name="Model1_Baseline", 
        seed=42 # Identidad experimental
    )

    # =========================================================
    # MODELO 2: HÍBRIDO RÍGIDO (SIN GATE)
    # =========================================================
    print("\n" + "="*50)
    print("🚀 INICIANDO MODELO 2: HÍBRIDO RÍGIDO (Sin Gate)")
    print("="*50)
    
    # Inyectamos el wrapper forzando la desactivación del Context Gate
    class AranduYOLOWrapperNoGate(AranduBackbone):
        def __init__(self, *args, **kwargs):
            kwargs['use_context_gate'] = False
            kwargs['moco_checkpoint_path'] = moco_ckpt
            super().__init__(*args, **kwargs)
            
    setattr(nn_modules, 'AranduYOLOWrapper', AranduYOLOWrapperNoGate)
    setattr(sys.modules['ultralytics.nn.modules'], 'AranduYOLOWrapper', AranduYOLOWrapperNoGate)
    
    model2 = YOLO("arandu_yolov8.yaml")
    model2.train(
        data=data_yaml, 
        epochs=epochs, 
        imgsz=imgsz, 
        batch=batch,
        freeze=10, # Fase 1: Congela el backbone 10 epochs
        project="Ablation_SojAI", 
        name="Model2_NoGate", 
        seed=42
    )

    # =========================================================
    # MODELO 3: HÍBRIDO ADAPTATIVO (CONTEXT GATE SOTA)
    # =========================================================
    print("\n" + "="*50)
    print("🚀 INICIANDO MODELO 3: HÍBRIDO ADAPTATIVO (Con Gate)")
    print("="*50)
    
    # Inyectamos el wrapper habilitando el Context Gate
    class AranduYOLOWrapperGate(AranduBackbone):
        def __init__(self, *args, **kwargs):
            kwargs['use_context_gate'] = True
            kwargs['moco_checkpoint_path'] = moco_ckpt
            super().__init__(*args, **kwargs)
            
    setattr(nn_modules, 'AranduYOLOWrapper', AranduYOLOWrapperGate)
    setattr(sys.modules['ultralytics.nn.modules'], 'AranduYOLOWrapper', AranduYOLOWrapperGate)
    
    model3 = YOLO("arandu_yolov8.yaml")
    model3.train(
        data=data_yaml, 
        epochs=epochs, 
        imgsz=imgsz, 
        batch=batch,
        freeze=10, 
        project="Ablation_SojAI", 
        name="Model3_ContextGate", 
        seed=42
    )
    
    print("\n✅ ENTRENAMIENTO DE ABLACIÓN COMPLETADO.")
    print("👉 Revisa la carpeta 'Ablation_SojAI' para extraer los CSV de métricas.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Orquestador del Estudio de Ablación ARANDU-AI. "
                    "Entrena 3 modelos bajo condiciones idénticas para cuantificar "
                    "el impacto del backbone SSL y del Context Gate."
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Ruta al YAML de dataset YOLO (ej: dataset_soja.yaml)."
    )
    parser.add_argument(
        "--moco-ckpt",
        type=str,
        required=True,
        help="Ruta al checkpoint del encoder SSL exportado "
             "(ej: /kaggle/working/moco_encoder_ready.pth)."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Número de epochs de entrenamiento para cada modelo (default: 100)."
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Tamaño de batch (default: 16)."
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Tamaño de imagen de entrenamiento en píxeles (default: 640)."
    )

    args = parser.parse_args()

    # --- Validaciones de paths antes de lanzar el estudio ---
    import os
    errors = []
    if not os.path.isfile(args.data):
        errors.append(f"  ❌ --data no encontrado: '{args.data}'")
    if not os.path.isfile(args.moco_ckpt):
        errors.append(f"  ❌ --moco-ckpt no encontrado: '{args.moco_ckpt}'")

    if errors:
        print("\n[!] Errores en los argumentos. Corrígelos antes de ejecutar:\n")
        for e in errors:
            print(e)
        print("\nEjemplo de uso:")
        print("  python ablation_runner.py \\")
        print("    --data dataset_soja.yaml \\")
        print("    --moco-ckpt /kaggle/working/moco_encoder_ready.pth \\")
        print("    --epochs 100 --batch 16 --imgsz 640")
        raise SystemExit(1)

    print("\n" + "="*60)
    print("🧪 INICIANDO ESTUDIO DE ABLACIÓN ARANDU-AI")
    print(f"  Dataset : {args.data}")
    print(f"  MoCo ckpt: {args.moco_ckpt}")
    print(f"  Epochs  : {args.epochs} | Batch: {args.batch} | Imgsz: {args.imgsz}")
    print("="*60 + "\n")

    run_ablation(
        data_yaml=args.data,
        moco_ckpt=args.moco_ckpt,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
    )
