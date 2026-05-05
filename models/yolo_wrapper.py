import torch
import torch.nn as nn
from models.moco import ModelBase

class AranduBackbone(nn.Module):
    """
    Backbone Wrapper de ARANDU-AI para la integración con arquitecturas YOLO (YOLOv8/YOLO26n).
    Extrae representaciones pre-entrenadas mediante MoCo v3 y proporciona
    los mapas de características (Feature Maps) necesarios para el Neck (PANet/FPN) de YOLO.
    """
    def __init__(self, moco_checkpoint_path=None, freeze_phase=1):
        super().__init__()
        
        # 1. Instanciar la arquitectura base de MoCo para obtener la ResNet-50
        print("[*] Inicializando AranduBackbone (ResNet-50 MoCo v3)...")
        moco_model = ModelBase()
        self.resnet = moco_model.encoder
        
        # 2. Carga inteligente de pesos pre-entrenados
        if moco_checkpoint_path:
            print(f"[*] Cargando pesos desde: {moco_checkpoint_path}")
            state_dict = torch.load(moco_checkpoint_path, map_location='cpu', weights_only=True)
            
            # Limpiar llaves (quitar prefijos de DDP o MoCo)
            encoder_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.encoder.'):
                    encoder_state_dict[k.replace('module.encoder.', '')] = v
                elif k.startswith('encoder.'):
                    encoder_state_dict[k.replace('encoder.', '')] = v
                # En caso de que se haya guardado solo el dict del encoder o modelo base puro
                elif not k.startswith('projector.') and not k.startswith('predictor.') and not k.startswith('queue'):
                    encoder_state_dict[k] = v
            
            # Cargar pesos en el encoder
            missing, unexpected = self.resnet.load_state_dict(encoder_state_dict, strict=False)
            if len(missing) > 0:
                print(f"[!] Faltaron pesos para algunas capas (esperado si eliminamos fc): {missing}")
            
        # 3. Eliminar capas innecesarias para YOLO (Global Average Pooling y Fully Connected)
        if hasattr(self.resnet, 'avgpool'):
            del self.resnet.avgpool
        if hasattr(self.resnet, 'fc'):
            del self.resnet.fc
            
        # 4. Aplicar la lógica de congelación según la Fase de entrenamiento
        self.set_training_phase(freeze_phase)

    def set_training_phase(self, phase):
        """
        Configura los gradientes del backbone basándose en la estrategia del informe.
        - Fase 1: Entrenamiento de Cabecera (Frozen Backbone)
        - Fase 2: Ajuste Fino Selectivo (Fine-Tuning de layer4/P5)
        - Fase 3: Unfreeze Global
        """
        print(f"[*] Configurando Fase de Entrenamiento: {phase}")
        
        if phase == 1:
            # Congelar todo el backbone
            for param in self.resnet.parameters():
                param.requires_grad = False
            print("    -> Fase 1: Backbone completamente congelado.")
            
        elif phase == 2:
            # Congelar todo excepto layer4 (P5)
            for param in self.resnet.parameters():
                param.requires_grad = False
            for param in self.resnet.layer4.parameters():
                param.requires_grad = True
            print("    -> Fase 2: Ajuste Fino Selectivo. Solo 'layer4' (P5) puede actualizarse.")
            
        elif phase == 3:
            # Descongelar todo (se recomienda LR bajo = 1e-6)
            for param in self.resnet.parameters():
                param.requires_grad = True
            print("    -> Fase 3: Unfreeze Global. Todo el backbone actualizará gradientes.")
            
        else:
            raise ValueError(f"Fase de entrenamiento no reconocida: {phase}. Usa 1, 2 o 3.")

    def forward(self, x):
        """
        Extrae y retorna los mapas de características en 3 escalas distintas
        requeridas por la arquitectura YOLO (P3, P4, P5).
        """
        # --- Stem ---
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        # --- Bloques Residuales ---
        x = self.resnet.layer1(x)  # Stride 4
        
        # P3 (Small) - Salida de layer2 (Stride 8)
        # Dimensión típica en ResNet50: [B, 512, H/8, W/8]
        P3 = self.resnet.layer2(x)
        
        # P4 (Medium) - Salida de layer3 (Stride 16)
        # Dimensión típica en ResNet50: [B, 1024, H/16, W/16]
        P4 = self.resnet.layer3(P3)
        
        # P5 (Large) - Salida de layer4 (Stride 32)
        # Dimensión típica en ResNet50: [B, 2048, H/32, W/32]
        P5 = self.resnet.layer4(P4)

        return [P3, P4, P5]


def inject_arandu_into_yolo(yolo_model, arandu_backbone):
    """
    Función de utilidad para inyectar este backbone dentro de un modelo Ultralytics.
    Esta función reemplaza el extractor de características de YOLO por nuestro ResNet-50.
    """
    print("[!] Modificando el modelo YOLO...")
    # NOTA: La integración exacta depende de la estructura interna del yolo_model (Ultralytics).
    # En YOLOv8, el backbone abarca las capas 0 a 9 típicamente. 
    # Sustituir directamente requiere que el modelo soporte un custom backbone o envolver el model.model.
    # Aquí se muestra un patrón conceptual:
    
    # yolo_model.model.model[:10] = arandu_backbone
    
    # Una forma más segura en Ultralytics es heredar de su BaseTensorModel o modificar el nn.Sequential interno.
    print("[*] Revisa el código de integración específico en tu script de entrenamiento YOLO.")
    return yolo_model
