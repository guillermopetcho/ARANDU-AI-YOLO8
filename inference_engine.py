import os
import json
import argparse
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    from sklearn.neighbors import NearestNeighbors
    HAS_FAISS = False

from evaluation.knn import extract_features_fast
from engine.setup import build_eval_dataset
from models.moco import ModelBase

class AranduInferenceEngine:
    def __init__(self, config_path="config/moco.yaml", device=None):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Las clases se resuelven dinámicamente en _resolve_class_names().
        # Nunca se usan valores hardcodeados para evitar predicciones silenciosamente erroneas.
        self.class_names = None
        self.num_classes = None

        # C-1 FIX: Leer resolución desde config para que coincida con la resolución
        # de entrenamiento del encoder. Antes estaba hardcodeada en 320px mientras
        # el encoder fue entrenado a 384px (global_crop_size en moco.yaml),
        # causando mismatch de distribución silencioso y degradación de accuracy.
        self.eval_size = self.config['moco'].get('global_crop_size', 384)

        self.transform = transforms.Compose([
            transforms.Resize(self.eval_size),
            transforms.CenterCrop(self.eval_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.encoder = None
        self.head = None
        self.reference_embeddings = None
        self.reference_labels = None
        self.knn_index = None

        self._resolve_class_names()
        self._load_models()

    def _resolve_class_names(self):
        """Resuelve los nombres de clase desde fuentes persistentes.

        Estrategia de resolucín en orden de prioridad:
          1. class_names.json guardado junto al encoder (más rápido, no requiere dataset).
          2. ImageFolder sobre eval_train_root (requiere acceso al dataset de training).
          3. Error explícito con instrucciones claras — nunca silencioso.
        """
        encoder_path = self.config['paths']['encoder_export_path']
        class_names_path = encoder_path.replace('.pth', '_class_names.json')

        # --- Prioridad 1: cache JSON ---
        if os.path.isfile(class_names_path):
            with open(class_names_path, 'r') as f:
                self.class_names = json.load(f)
            print(f"[*] Clases cargadas desde cache: {self.class_names}")

        # --- Prioridad 2: ImageFolder / YOLO dataset sobre train dir ---
        elif os.path.isdir(self.config['paths'].get('eval_train_root', '')):
            train_dir = self.config['paths']['eval_train_root']
            data_yaml = self.config['paths'].get('data_yaml', None)
            ds = build_eval_dataset(train_dir, transform=None, data_yaml_path=data_yaml)
            self.class_names = ds.classes
            # Guardar para futuras instancias
            with open(class_names_path, 'w') as f:
                json.dump(self.class_names, f, indent=2)
            print(f"[*] Clases inferidas del dataset y guardadas en cache: {self.class_names}")

        # --- Prioridad 3: Error explícito ---
        else:
            raise RuntimeError(
                f"No se pudieron resolver los nombres de clase.\n"
                f"  - Cache esperado en: '{class_names_path}'\n"
                f"  - Dataset de training no encontrado en: '{self.config['paths'].get('eval_train_root', '?')}'\n"
                f"Solución: Ejecuta train.py para generar el encoder y el cache de clases, "
                f"o crea manualmente '{class_names_path}' con formato [\"clase1\", \"clase2\", ...]."
            )

        self.num_classes = len(self.class_names)


    def _load_models(self):
        encoder_path = self.config['paths']['encoder_export_path']
        head_path = encoder_path.replace('.pth', '_head.pth')
        
        print("[*] Cargando Encoder...")
        self.encoder = ModelBase(
            dim=self.config['moco']['dim'],
            predictor_hidden_dim=self.config['moco'].get('predictor_hidden_dim', 1024)
        )
        self.encoder.load_state_dict(torch.load(encoder_path, map_location='cpu', weights_only=True))
        self.encoder = self.encoder.to(self.device).eval()

        print("[*] Cargando Head Lineal...")
        with torch.no_grad():
            # C-1 FIX: Usar self.eval_size (leído desde config) en lugar de 320 hardcodeado.
            dummy = torch.randn(1, 3, self.eval_size, self.eval_size).to(self.device)
            proj_dim = self.encoder(dummy, use_predictor=False).shape[-1]
            
        self.head = nn.Sequential(
            nn.LayerNorm(proj_dim),
            nn.Linear(proj_dim, self.num_classes)
        )
        self.head.load_state_dict(torch.load(head_path, map_location='cpu', weights_only=True))
        self.head = self.head.to(self.device).eval()

    def build_or_load_reference_db(self, force_rebuild=False):
        # BUG-AMB-11 FIX: El hash anterior se calculaba sobre la RUTA del data_yaml (un string),
        # no sobre su CONTENIDO. Si el archivo cambiaba de contenido (nueva clase, renombre)
        # pero su path permanecía igual, el caché NO se invalidaba y las referencias KNN
        # quedaban desactualizadas silenciosamente.
        # Ahora hasheamos el CONTENIDO del archivo si existe, o el string del path como fallback.
        import hashlib
        data_yaml = self.config['paths'].get('data_yaml', '')
        if data_yaml and os.path.isfile(data_yaml):
            with open(data_yaml, 'rb') as _f:
                yaml_hash = hashlib.md5(_f.read()).hexdigest()[:8]
        else:
            yaml_hash = hashlib.md5(data_yaml.encode()).hexdigest()[:8]

        train_dir = self.config['paths'].get('eval_train_root', '')
        base_cache = "/kaggle/working/reference_db" if "kaggle" in train_dir else "reference_db"
        cache_path = f"{base_cache}_{yaml_hash}.pt"
        
        if os.path.exists(cache_path) and not force_rebuild:
            print("[*] Cargando base de datos KNN desde caché...")
            data = torch.load(cache_path, weights_only=True)
            self.reference_embeddings = data['embeddings']
            self.reference_labels = data['labels']
        else:
            print("[*] Construyendo base de datos KNN desde Train Dataset (puede demorar)...")
            # BUG-M5 FIX: Se eliminó el fallback hardcodeado a un path de Kaggle específico.
            # El patrón anti-portabilidad fue identificado y removido de evaluate_downstream.py
            # (R-4 FIX) pero permanecía aquí. Un path con usuario/dataset específico falla
            # silenciosamente en cualquier entorno que no sea el original.
            # Fallamos explícitamente con instrucciones claras.
            if not train_dir or not os.path.exists(train_dir):
                raise RuntimeError(
                    f"[InferenceEngine] No se encontró el directorio de training en: '{train_dir}'.\n"
                    f"  Verifica 'paths.eval_train_root' en config/moco.yaml y que el dataset esté montado."
                )
            
            # C1 FIX: pasar data_yaml_path para que el dataset use las clases reales
            train_ds = build_eval_dataset(train_dir, transform=self.transform, data_yaml_path=data_yaml or None)
            loader = DataLoader(train_ds, batch_size=256, shuffle=False, num_workers=4)
            
            feats, labels = extract_features_fast(self.encoder, loader, self.device)
            self.reference_embeddings = feats
            self.reference_labels = labels
            
            torch.save({'embeddings': feats, 'labels': labels}, cache_path)
            
        print(f"[*] Base KNN lista: {len(self.reference_labels)} referencias.")
        self._build_knn_index()

    def _build_knn_index(self):
        # L2 Normalizar embeddings de referencia
        # M6 FIX: np.ascontiguousarray solo copia si el dtype o layout es incorrecto.
        # .astype(np.float32) siempre crea una copia aunque el array ya sea float32,
        # lo que desperdicia ~50MB en datasets grandes (50k imgs × 256 dims).
        self.reference_embeddings = np.ascontiguousarray(self.reference_embeddings, dtype=np.float32)
        if HAS_FAISS:
            faiss.normalize_L2(self.reference_embeddings)
            self.knn_index = faiss.IndexFlatIP(self.reference_embeddings.shape[1])
            self.knn_index.add(self.reference_embeddings)
        else:
            norms = np.linalg.norm(self.reference_embeddings, axis=1, keepdims=True)
            self.reference_embeddings = self.reference_embeddings / (norms + 1e-8)
            self.knn_index = NearestNeighbors(n_neighbors=20, metric='cosine', n_jobs=-1)
            self.knn_index.fit(self.reference_embeddings)

    def predict(self, image_path, k=5):
        if self.knn_index is None:
            self.build_or_load_reference_db()
            
        img = Image.open(image_path).convert('RGB')
        x = self.transform(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # 1. Extraer Feature y Normalizar
            z = self.encoder(x, use_predictor=False)
            z_norm = F.normalize(z, dim=1)
            
            # 2. Linear Head
            logits = self.head(z_norm)
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()
            
        # 3. Métricas Head Lineal
        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        entropy = float(-np.sum(probs * np.log(probs + 1e-8)))
        predicted_class = self.class_names[pred_idx]
        
        # 4. KNN Search
        z_np = z_norm.cpu().numpy().astype(np.float32)
        if HAS_FAISS:
            faiss.normalize_L2(z_np)
            distances, indices = self.knn_index.search(z_np, k)
            # En FAISS Inner Product (Cosine sim): distance = 1.0 es identidad.
            # Convertimos a distancia angular aproximada: 1 - sim
            distances = 1.0 - distances[0]
            indices = indices[0]
        else:
            z_np = z_np / (np.linalg.norm(z_np, axis=1, keepdims=True) + 1e-8)
            distances, indices = self.knn_index.kneighbors(z_np, n_neighbors=k)
            distances = distances[0]
            indices = indices[0]
            
        neighbor_labels = [self.class_names[self.reference_labels[i]] for i in indices]
        consistency = neighbor_labels.count(predicted_class) / k
        avg_distance = float(np.mean(distances))
        
        # 5. Lógica de Decisión Experta (Fusión)
        if confidence > 0.8 and entropy < 0.5 and consistency >= 0.6:
            diagnosis = "🟢 Confiable. Diagnóstico sólido confirmado por morfología latente."
        elif confidence >= 0.5 and consistency >= 0.4:
            diagnosis = "🟡 Dudosa. El modelo sugiere esto, pero la morfología latente es mixta. Revisar manualmente."
        else:
            diagnosis = "🔴 Peligro (Posible OOD). Alta incertidumbre y baja consistencia estructural."
            
        # 6. Formatear Salida
        result = {
            "predicted_class": predicted_class,
            "confidence": round(confidence, 4),
            "entropy": round(entropy, 4),
            "knn": {
                "neighbors": neighbor_labels,
                "consistency": round(consistency, 4),
                "avg_distance": round(avg_distance, 4)
            },
            "diagnosis": diagnosis
        }
        
        return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Ruta a la imagen de la hoja de soja")
    args = parser.parse_args()
    
    engine = AranduInferenceEngine()
    result = engine.predict(args.image)
    
    print("\n" + "="*50)
    print("🔬 REPORTE DE INFERENCIA ARANDUSSL")
    print("="*50)
    print(json.dumps(result, indent=4, ensure_ascii=False))
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
