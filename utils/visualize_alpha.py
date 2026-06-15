"""
utils/visualize_alpha.py — Visualización del Residual Gate (β) del AranduBackbone.

Genera mapas de calor que muestran la contribución relativa del adaptador
(local_feat) vs el shortcut (features SSL crudas) en cada escala P3/P4/P5.

El SpatialFeatureAdapter produce: Y = shortcut + β · local_feat
  - β → 0: El adaptador está inactivo, YOLO usa features SSL puras.
  - β >> 0: El adaptador aporta información adicional (atención, textura local).

Visualización:
  Se captura el feature map |β · local_feat| / (|shortcut| + |β · local_feat| + ε)
  como "ratio de contribución del adaptador" por posición espacial.

Uso:
    python utils/visualize_alpha.py \\
        --image test_hoja.jpg \\
        --model runs/detect/Model3_ContextGate/weights/best.pt \\
        --output alpha_heatmaps.png
"""

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO


class AdapterContributionExtractor:
    """
    Extrae la contribución del adaptador en cada escala P3/P4/P5
    usando forward hooks sobre el SpatialFeatureAdapter.

    Como el Residual Gate es ahora un parámetro escalar β (no un submodule),
    enganchamos el hook al forward del adapter completo y calculamos
    la contribución relativa: |output - shortcut| / (|output| + ε).

    REF: El gate anterior (context_gate) fue reemplazado por beta escalar
    en el refactor de SpatialFeatureAdapter. Ver models/yolo_wrapper.py.
    """
    def __init__(self, model):
        self.model = model
        self.contributions = {}
        self.beta_values = {}
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        try:
            backbone = self.model.model.model[0]

            def get_hook(name, adapter):
                def hook(module, input, output):
                    # input[0] = x original del adapter
                    # output = shortcut + beta * local_feat
                    x_in = input[0]

                    with torch.no_grad():
                        # Recalcular shortcut para obtener la contribución del adaptador
                        shortcut = module.shortcut(x_in)
                        adapter_contribution = output - shortcut  # = beta * local_feat

                        # Magnitud espacial de la contribución (norma L2 por posición)
                        contrib_magnitude = adapter_contribution.norm(dim=1, keepdim=True)  # [B, 1, H, W]
                        output_magnitude = output.norm(dim=1, keepdim=True)                 # [B, 1, H, W]

                        # Ratio de contribución del adaptador vs output total
                        ratio = contrib_magnitude / (output_magnitude + 1e-6)  # [B, 1, H, W]

                        self.contributions[name] = ratio.detach().cpu().numpy()
                        self.beta_values[name] = module.beta.item()

                return hook

            # Enganchar a los SpatialFeatureAdapter de P3, P4 y P5
            self.hooks.append(
                backbone.adapter_p3.register_forward_hook(get_hook('P3 (Small)', backbone.adapter_p3))
            )
            self.hooks.append(
                backbone.adapter_p4.register_forward_hook(get_hook('P4 (Medium)', backbone.adapter_p4))
            )
            self.hooks.append(
                backbone.adapter_p5.register_forward_hook(get_hook('P5 (Large)', backbone.adapter_p5))
            )
            print("[*] Hooks registrados exitosamente en P3, P4 y P5.")
        except Exception as e:
            print(f"[!] Error registrando hooks. Asegúrate de que el modelo cargado "
                  f"es el híbrido Arandu. Detalles: {e}")

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()


def generate_alpha_heatmaps(image_path, model_path, save_path="alpha_heatmaps.png"):
    print(f"[*] Cargando modelo YOLO híbrido desde {model_path}...")
    model = YOLO(model_path)
    extractor = AdapterContributionExtractor(model)

    # Correr inferencia normal de YOLO (el forward dispara los hooks)
    results = model(image_path, verbose=False)

    # Preparar imagen base
    img_bgr = cv2.imread(image_path)
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

    print("\n📊 VALIDACIÓN CUANTITATIVA DEL RESIDUAL GATE (BBox vs Fondo):")

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))

    # 1. Imagen Original
    axes[0].imshow(img_rgb)
    axes[0].set_title("Imagen Original (Soja)")
    axes[0].axis('off')

    # 2. Iterar sobre las escalas P3, P4, P5
    scales = ['P3 (Small)', 'P4 (Medium)', 'P5 (Large)']
    for idx, scale in enumerate(scales):
        contribution_map = extractor.contributions.get(scale, None)
        beta_val = extractor.beta_values.get(scale, None)
        ax = axes[idx + 1]

        if contribution_map is not None:
            # Seleccionar batch 0, canal 0
            c_map = contribution_map[0, 0]

            # Redimensionar el feature map al tamaño original
            c_map_resized = cv2.resize(c_map, (img_rgb.shape[1], img_rgb.shape[0]))

            # Extraer promedios cuantitativos
            contrib_box = c_map_resized[mask_bbox].mean() if mask_bbox.any() else 0.0
            contrib_bg = c_map_resized[~mask_bbox].mean() if (~mask_bbox).any() else 0.0
            beta_str = f"β={beta_val:.4f}" if beta_val is not None else "β=?"
            print(f"  - {scale}: Contrib. en BBox = {contrib_box:.4f} | "
                  f"Contrib. en Fondo = {contrib_bg:.4f} | {beta_str}")

            # Superponer el Heatmap sobre la imagen original
            ax.imshow(img_rgb)
            hm = ax.imshow(c_map_resized, cmap='jet', alpha=0.5, vmin=0.0, vmax=1.0)

            ax.set_title(f"Residual Gate {beta_str}\nBBox:{contrib_box:.2f} Bg:{contrib_bg:.2f}")
            ax.axis('off')
            fig.colorbar(hm, ax=ax, fraction=0.046, pad=0.04)
        else:
            ax.set_title(f"No se detectó {scale}")
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[+] Mapa de calor guardado en {save_path}")

    extractor.remove_hooks()

if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(
        description="Genera mapas de calor del Residual Gate (beta) sobre una imagen de soja."
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Ruta a la imagen de entrada (ej: test_hoja.jpg).",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Ruta al modelo YOLO híbrido entrenado (ej: runs/detect/Model3/weights/best.pt).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="alpha_heatmaps.png",
        help="Ruta del PNG de salida (default: alpha_heatmaps.png).",
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
        print("    --model runs/detect/Model3/weights/best.pt \\")
        print("    --output alpha_heatmaps.png")
        raise SystemExit(1)

    generate_alpha_heatmaps(args.image, args.model, save_path=args.output)
