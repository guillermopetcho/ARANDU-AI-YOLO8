import torch
from models.yolo_wrapper import AranduBackbone

model = AranduBackbone()
for name, param in model.named_parameters():
    print(name)
