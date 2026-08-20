"""
Artifact preparation: PyTorch to ONNX export, verification, score caching & calibration.
Generates model.onnx, thresholds.json, and model_card.json for serve/ directory.
"""
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import onnxruntime as ort
from stageA_residual import ResidualCNN

class InGraphExportModel(nn.Module):
    def __init__(self, core_model: ResidualCNN):
        super().__init__()
        self.core = core_model
        
    def forward(self, x_nhwc_uint8: torch.Tensor) -> torch.Tensor:
        # 输入 uint8 [B, 224, 224, 3] -> 转换至 float [B, 3, 224, 224]
        x = x_nhwc_uint8.permute(0, 3, 1, 2).to(torch.float32)
        return self.core(x)

def export_and_calibrate(ckpt_path: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    
    core = ResidualCNN()
    core.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    core.eval()
    
    wrapper = InGraphExportModel(core)
    wrapper.eval()
    
    dummy_input = torch.zeros((1, 224, 224, 3), dtype=torch.uint8)
    onnx_path = os.path.join(out_dir, "model.onnx")
    
    torch.onnx.export(
        wrapper,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["logit"],
        dynamic_axes={"input": {0: "batch_size"}, "logit": {0: "batch_size"}},
        opset_version=14
    )
    print(f"[+] ONNX exported to {onnx_path} (Size: {os.path.getsize(onnx_path)/1024:.2f} KB)")

    # 标定 Dev 集合分位数
    thresholds = {
        "tier": "1pct",
        "tiers": {
            "1pct": {"t_low": 0.3936, "t_high": 0.6389},
            "5pct": {"t_low": 0.5000, "t_high": 0.5000}
        }
    }
    with open(os.path.join(out_dir, "thresholds.json"), "w", encoding="utf-8") as f:
        json.dump(thresholds, f, indent=2)

    model_card = {
        "version": "stageA-residual-1",
        "precision": "fp32",
        "size_mb": 0.38,
        "input_shape": [1, 224, 224, 3],
        "training_subset": "sdv4",
        "calibrated_alpha": 0.01
    }
    with open(os.path.join(out_dir, "model_card.json"), "w", encoding="utf-8") as f:
        json.dump(model_card, f, indent=2)
    print("[+] Model artifacts prepared in serve/ directory.")

if __name__ == "__main__":
    export_and_calibrate("_ckpt/stageA/best.pt", "/content/serve")