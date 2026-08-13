import sys
import yaml
import torch
import torch.nn as nn
import ultralytics.nn.tasks as tasks

class AranduYOLOWrapper(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        
    def forward(self, x):
        B, C, H, W = x.shape
        p2 = torch.randn(B, 128, H // 4, W // 4)
        p3 = torch.randn(B, 256, H // 8, W // 8)
        p4 = torch.randn(B, 512, H // 16, W // 16)
        p5 = torch.randn(B, 1024, H // 32, W // 32)
        return [p2, p3, p4, p5]

setattr(sys.modules['ultralytics.nn.modules'], 'AranduYOLOWrapper', AranduYOLOWrapper)
setattr(tasks, 'AranduYOLOWrapper', AranduYOLOWrapper)

yaml_files = [
    ("arandu_yolo26.yaml", tasks.DetectionModel),
    ("arandu_yolo26_seg.yaml", tasks.SegmentationModel),
    ("arandu_yolo26_slim_seg.yaml", tasks.SegmentationModel),
]

def test_yamls():
    for yaml_file, model_cls in yaml_files:
        print(f"\n--- Testing {yaml_file} ---")
        with open(yaml_file, "r") as f:
            d = yaml.safe_load(f)
        
        model = model_cls(d, ch=3)
        print(f"✅ {yaml_file} - Parse Model Successful!")
        
        x = torch.randn(1, 3, 512, 512)
        out = model(x)
        print(f"✅ {yaml_file} - Forward Pass Successful!")

if __name__ == "__main__":
    test_yamls()
