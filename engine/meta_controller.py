import json
import os
import logging

class MetaController:
    """
    Controlador Semántico (Nivel 2).
    Metric-Driven Curriculum Controller.
    Evalúa métricas reales del espacio latente para determinar el dominio de:
    - Textura (Local vs Global Loss, Local Alignment)
    - Estructura (Global Alignment, Uniformidad)
    - Invariancia (Estabilidad inter-escalas, KNN)
    
    Ajusta dinámicamente las políticas de Data Augmentation y Pesos Locales/Globales
    basándose en el conocimiento de fases anteriores y adaptándose cada N epochs.
    """
    def __init__(self, current_imgsz, prev_metrics_path=None):
        self.current_imgsz = current_imgsz
        self.prev_metrics_path = prev_metrics_path
        self.logger = logging.getLogger("AranduSSL")
        
        # Estado semántico real basado en métricas
        self.semantic_state = {
            "texture": 0.0,
            "structure": 0.0,
            "scale_invariance": 0.0,
            "recommended_phase": "INITIALIZING",
            "health_score": 0.0
        }
        
        self.curriculum_params = {
            "num_local_crops": 4,
            "local_loss_weight": 0.2
        }
        
    def _load_prev_knowledge(self):
        if not self.prev_metrics_path or not os.path.exists(self.prev_metrics_path):
            return None
        try:
            with open(self.prev_metrics_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.warning(f"No se pudo cargar el conocimiento previo: {e}")
            return None

    def build_curriculum_profile(self):
        """Inicializa el perfil óptimo basado en la base empírica y el estado heredado."""
        # LOW-6 FIX: Acceso seguro con error descriptivo en lugar de KeyError opaco.
        _profiles = {
            384: {"global_crop_size": 384, "local_crop_size": 96, "batch_size": 16, "warmup_epochs": 2, "local_loss_weight": 0.2, "num_local_crops": 4},
            512: {"global_crop_size": 512, "local_crop_size": 160, "batch_size": 8, "warmup_epochs": 3, "local_loss_weight": 0.25, "num_local_crops": 4},
            640: {"global_crop_size": 640, "local_crop_size": 192, "batch_size": 4, "warmup_epochs": 4, "local_loss_weight": 0.15, "num_local_crops": 2}
        }
        profile = _profiles.get(self.current_imgsz)
        if profile is None:
            raise ValueError(
                f"MetaController: imgsz={self.current_imgsz} no soportado. "
                f"Resoluciones válidas: {sorted(_profiles.keys())}. "
                f"Para agregar soporte, define un perfil en build_curriculum_profile()."
            )

        prev = self._load_prev_knowledge()
        if prev:
            # Transferir conocimiento previo a nuestra inicialización (KNN, loss, etc)
            knn = prev.get("acc", 0.0)
            self.logger.info(f"🧠 MetaController: Conocimiento previo cargado (KNN {knn*100:.1f}%)")
            # Ajustes iniciales leves basados en herencia, pero el control real es dinámico
            if self.current_imgsz == 512 and knn > 0.90:
                profile["num_local_crops"] = 2
                profile["local_loss_weight"] = 0.15
            elif self.current_imgsz == 640 and knn > 0.90:
                profile["num_local_crops"] = 0
                profile["local_loss_weight"] = 0.0

        self.curriculum_params["num_local_crops"] = profile["num_local_crops"]
        self.curriculum_params["local_loss_weight"] = profile["local_loss_weight"]
        
        return profile
        
    def update_dynamic_state(self, epoch, metrics, curr_acc):
        """
        Actualiza el estado semántico mid-training evaluando señales reales.
        - Texture: Inversamente proporcional a local_loss
        - Structure: Basado en global_loss y alignment
        - Invariance: Basado en KNN y uniformity
        """
        global_loss = metrics.get('global_loss', 1.0)
        local_loss = metrics.get('local_loss', 1.0)
        align = metrics.get('align', 1.0)
        unif = abs(metrics.get('unif', 1.0))
        std = metrics.get('std', 0.0)
        
        # 1. Health Score Compuesto (El "Curriculum Score")
        # Combina KNN (si está disponible), alineación, uniformidad y desviación estándar.
        safe_knn = curr_acc if curr_acc >= 0 else 0.5
        norm_align = max(0.0, 1.0 - align)  # Alignment ideal es bajo
        norm_unif = min(1.0, unif / 2.0)    # Uniformity ideal es alta (~2.0 absoluta)
        
        health_score = (0.4 * safe_knn) + (0.2 * norm_align) + (0.2 * norm_unif) + (0.2 * std)
        self.semantic_state["health_score"] = health_score * 100
        
        # 2. Texture Learning (Cross-View Agreement)
        # I-1 FIX: La fórmula anterior (1 - local/global * 0.5) producía siempre ≈48%
        # porque ambas pérdidas decaen proporcionalmente al compartir backbone y optimizer.
        # Ahora usamos el coseno medio entre queries globales y locales del mismo batch:
        # si el encoder aprendió texturas, un crop de 96px debe producir features
        # coherentes con el crop global de 384px.
        texture_agreement = metrics.get('texture_agreement', -999.0)
        if texture_agreement > -2.0:  # Any valid cosine similarity is >= -1.0
            # Rango original: [-1, 1] → escalado a [0, 100]
            # Agreement 0.7 → 85%, Agreement 0.85 → 92.5%
            # Escalar linealmente: (agreement + 1) / 2 -> [0, 1]
            self.semantic_state["texture"] = ((texture_agreement + 1.0) / 2.0) * 100
        else:
            # Fallback si no hubo crops locales (ej. curriculum desactivado)
            texture_score = 1.0 - min(1.0, (local_loss / max(global_loss, 1e-5)) * 0.5)
            self.semantic_state["texture"] = texture_score * 100
        
        # 3. Structure Learning (Pérdida Global y Alineación)
        structure_score = (norm_align + norm_unif) / 2.0
        self.semantic_state["structure"] = structure_score * 100
        
        # 4. Scale Invariance (KNN robusto y Features estables)
        # I-2 FIX: La fórmula anterior (safe_knn * std) mezclaba magnitudes incompatibles.
        # KNN ∈ [0,1] × std ∈ [0.03, 0.08] → resultado siempre en [0%, 8%].
        # Nueva fórmula: combina KNN (peso 70%) con std normalizada (peso 30%).
        # std_normalized escala std al rango [0,1] donde 0.06+ es "bueno" para dim=256.
        std_normalized = min(1.0, std / 0.06)  # std=0.06 → 1.0, std=0.03 → 0.5
        invariance_score = 0.7 * safe_knn + 0.3 * std_normalized
        self.semantic_state["scale_invariance"] = invariance_score * 100

        # --- RE-EVALUACIÓN CURRICULAR (CADA 5 EPOCHS) ---
        if epoch > 0 and epoch % 5 == 0:
            self._recompute_curriculum()
            
    def _recompute_curriculum(self):
        """Adapta la estrategia de aprendizaje en pleno vuelo (Mid-Training)."""
        health = self.semantic_state["health_score"]
        tex = self.semantic_state["texture"]
        
        if self.current_imgsz == 384:
            if tex > 85.0:
                self.semantic_state["recommended_phase"] = "TEXTURE MASTERED -> PREP STRUCTURE"
                self.curriculum_params["local_loss_weight"] = max(0.1, self.curriculum_params["local_loss_weight"] - 0.05)
            else:
                self.semantic_state["recommended_phase"] = "FOCUS TEXTURE"
                
        elif self.current_imgsz == 512:
            if tex > 90.0 and health > 80.0:
                self.semantic_state["recommended_phase"] = "STRUCTURE MASTERED -> PREP SCALE"
                self.curriculum_params["local_loss_weight"] = 0.1
                self.curriculum_params["num_local_crops"] = max(2, self.curriculum_params["num_local_crops"] - 2)
                self.logger.info("🧠 MetaController: Texturas y estructura dominadas. Reduciendo crops locales.")
            elif tex < 70.0:
                self.semantic_state["recommended_phase"] = "RECOVER TEXTURE"
                self.curriculum_params["local_loss_weight"] = 0.3
                # MED-4 FIX: Permitir recuperar crops locales si se pierde el dominio de textura
                self.curriculum_params["num_local_crops"] = min(4, self.curriculum_params.get("num_local_crops", 0) + 2)
            else:
                self.semantic_state["recommended_phase"] = "FOCUS STRUCTURE"
                
        elif self.current_imgsz == 640:
            if health > 85.0:
                self.semantic_state["recommended_phase"] = "SCALE INVARIANCE ACHIEVED"
                self.curriculum_params["local_loss_weight"] = 0.05
            else:
                self.semantic_state["recommended_phase"] = "FOCUS SCALE INVARIANCE"
                # En 640 forzamos a que ignore detalles locales y mire la hoja entera
                self.curriculum_params["local_loss_weight"] = 0.1
                
    def get_summary_string(self):
        return (f"🧠 Semantic State -> "
                f"Health: {self.semantic_state['health_score']:.1f}% | "
                f"Texture: {self.semantic_state['texture']:.1f}% | "
                f"Structure: {self.semantic_state['structure']:.1f}% | "
                f"Invariance: {self.semantic_state['scale_invariance']:.1f}%\n"
                f"    - Phase: {self.semantic_state['recommended_phase']} | "
                f"LocalWeight: {self.curriculum_params['local_loss_weight']:.2f}")
