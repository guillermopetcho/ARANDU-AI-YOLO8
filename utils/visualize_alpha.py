import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

class AlphaMapExtractor:
    """
    Herramienta de extracción y visualización para el Context Gate.
    Utiliza PyTorch Hooks para interceptar el tensor `alpha` durante la inferencia
    sin necesidad de alterar el código interno de Ultralytics.
    """
    def __init__(self, model):
        self.model = model
        self.alphas = {}
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        # En YOLO, el AranduBackbone modificado suele estar en el índice 0 del Sequential
        try:
            backbone = self.model.model.model[0]
            
            # Función para capturar el tensor alpha
            def get_hook(name):
                def hook(module, input, output):
                    # Output del context_gate es (B, 1, H, W)
                    self.alphas[name] = output.detach().cpu().numpy()
                return hook
                
            # Enganchamos a las capas de Sigmoid finales de cada adaptador
            self.hooks.append(backbone.adapter_p3.context_gate.register_forward_hook(get_hook('P3 (Small)')))
            self.hooks.append(backbone.adapter_p4.context_gate.register_forward_hook(get_hook('P4 (Medium)')))
            self.hooks.append(backbone.adapter_p5.context_gate.register_forward_hook(get_hook('P5 (Large)')))
            print("[*] Hooks registrados exitosamente en P3, P4 y P5.")
        except Exception as e:
            print(f"[!] Error registrando hooks. Asegúrate de que el modelo cargado es el híbrido Arandu. Detalles: {e}")

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()


def generate_alpha_heatmaps(image_path, model_path, save_path="alpha_heatmaps.png"):
    print(f"[*] Cargando modelo YOLO híbrido desde {model_path}...")
    model = YOLO(model_path)
    extractor = AlphaMapExtractor(model)
    
    # Correr inferencia normal de YOLO (el forward dispara los hooks)
    results = model(image_path, verbose=False)
    
    # Preparar imagen base
    img_bgr = cv2.imread(image_path)
    # M5 FIX: cv2.imread retorna None si el archivo no existe o no se puede leer.
    # Sin este guard, cv2.cvtColor crashea con un mensaje opaco de OpenCV.
    if img_bgr is None:
        extractor.remove_hooks()
        raise FileNotFoundError(
            f"No se pudo cargar la imagen: '{image_path}'. "
            "Verifica que el archivo exista y sea un formato válido (jpg/png)."
        )
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # [VALIDACIÓN CUANTITATIVA]: Crear máscara de Bounding Boxes
    mask_bbox = np.zeros((img_rgb.shape[0], img_rgb.shape[1]), dtype=bool)
    if len(results[0].boxes) > 0:
        for box in results[0].boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, box)
            mask_bbox[y1:y2, x1:x2] = True
            
    print("\n📊 VALIDACIÓN CUANTITATIVA DEL GATE (BBox vs Fondo):")
    
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    
    # 1. Imagen Original
    axes[0].imshow(img_rgb)
    axes[0].set_title("Imagen Original (Soja)")
    axes[0].axis('off')
    
    # 2. Iterar sobre las escalas P3, P4, P5
    scales = ['P3 (Small)', 'P4 (Medium)', 'P5 (Large)']
    for idx, scale in enumerate(scales):
        alpha_map = extractor.alphas.get(scale, None)
        ax = axes[idx + 1]
        
        if alpha_map is not None:
            # Seleccionar batch 0, canal 0
            a_map = alpha_map[0, 0] 
            
            # Redimensionar el feature map (ej. 80x80) al tamaño original (ej. 640x640)
            a_map_resized = cv2.resize(a_map, (img_rgb.shape[1], img_rgb.shape[0]))
            
            # Extraer promedios cuantitativos
            alpha_box = a_map_resized[mask_bbox].mean() if mask_bbox.any() else 0.0
            alpha_bg = a_map_resized[~mask_bbox].mean() if (~mask_bbox).any() else 0.0
            print(f"  - {scale}: \u03B1 en Lesión (BBox) = {alpha_box:.4f} | \u03B1 en Fondo = {alpha_bg:.4f}")
            
            # Superponer el Heatmap sobre la imagen original
            ax.imshow(img_rgb)
            hm = ax.imshow(a_map_resized, cmap='jet', alpha=0.5, vmin=0.0, vmax=1.0)
            
            ax.set_title(f"Context Gate \u03B1\nBBox:{alpha_box:.2f} Bg:{alpha_bg:.2f}")
            ax.axis('off')
            fig.colorbar(hm, ax=ax, fraction=0.046, pad=0.04)
        else:
            ax.set_title(f"No se detectó {scale}")
            ax.axis('off')
            
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[+] Mapa de calor SOTA guardado en {save_path}")
    
    extractor.remove_hooks()

if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(
        description="Genera mapas de calor del Context Gate (alpha) sobre una imagen de soja."
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Ruta a la imagen de entrada (ej: test_hoja.jpg)."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Ruta al modelo YOLO híbrido entrenado (ej: runs/detect/Model3_ContextGate/weights/best.pt)."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="alpha_heatmaps.png",
        help="Ruta del PNG de salida (default: alpha_heatmaps.png)."
    )

    args = parser.parse_args()

    errors = []
    if not os.path.isfile(args.image):
        errors.append(f"  ❌ --image no encontrado: '{args.image}'")
    if not os.path.isfile(args.model):
        errors.append(f"  ❌ --model no encontrado: '{args.model}'")

    if errors:
        print("\n[!] Errores en los argumentos:\n")
        for e in errors:
            print(e)
        print("\nEjemplo de uso:")
        print("  python utils/visualize_alpha.py \\")
        print("    --image test_hoja.jpg \\")
        print("    --model runs/detect/Model3_ContextGate/weights/best.pt \\")
        print("    --output alpha_heatmaps.png")
        raise SystemExit(1)

    generate_alpha_heatmaps(args.image, args.model, save_path=args.output)
