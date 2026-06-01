import torch
import torch.nn as nn
import ultralytics.nn.tasks as tasks
from ultralytics.nn.modules import Conv, C2f, Concat, Detect

class AranduYOLOWrapper(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.ch = [512, 1024, 2048] # YOLO hack to provide channel sizes
        
    def forward(self, x):
        p3 = torch.randn(1, 512, 80, 80)
        p4 = torch.randn(1, 1024, 40, 40)
        p5 = torch.randn(1, 2048, 20, 20)
        return p3, p4, p5

# Let's add it to ultralytics globals
import sys
setattr(sys.modules['ultralytics.nn.modules'], 'AranduYOLOWrapper', AranduYOLOWrapper)
# Also need it in tasks.py namespace just in case
setattr(tasks, 'AranduYOLOWrapper', AranduYOLOWrapper)

import yaml
with open("arandu_yolo26.yaml", "r") as f:
    d = yaml.safe_load(f)

try:
    model, save = tasks.parse_model(d, ch=3)
    print("PARSE MODEL SUCCESSFUL")
    
    # Try forward pass
    x = torch.randn(1, 3, 640, 640)
    out = model(x)
    print("FORWARD SUCCESSFUL")
except Exception:
    import traceback
    traceback.print_exc()
