import yaml
import torch
import torch.nn as nn
from ultralytics.nn.tasks import parse_model
from models.yolo_wrapper import AranduBackbone
import ultralytics.nn.modules as nn_modules
import sys

class AranduYOLOClsWrapper(AranduBackbone):
    def __init__(self, *args, **kwargs):
        print("ARGS RECIBIDOS:", args)
        kwargs['moco_checkpoint_path'] = None
        kwargs['freeze_phase'] = 3
        kwargs['use_coord_attn'] = False
        super().__init__(**kwargs)

    def forward(self, x):
        features = super().forward(x)
        return features[-1]

setattr(nn_modules, 'AranduYOLOClsWrapper', AranduYOLOClsWrapper)
setattr(sys.modules['ultralytics.nn.modules'], 'AranduYOLOClsWrapper', AranduYOLOClsWrapper)
import ultralytics.nn.tasks
setattr(sys.modules['ultralytics.nn.tasks'], 'AranduYOLOClsWrapper', AranduYOLOClsWrapper)

d = {'nc': 5, 'scales': {'n': [1.0, 1.0, 1024]}, 'backbone': [[-1, 1, 'AranduYOLOClsWrapper', []]], 'head': [[-1, 1, 'Classify', [5]]]}
model, save = parse_model(d, ch=3)
print("Parseo exitoso!")
