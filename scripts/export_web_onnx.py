#!/usr/bin/env python3
"""
scripts/export_web_onnx.py
將本專案 Exp 5 (Tiny-640)、Exp 7 (Base-640)、Exp 12 (Tiny-960 + CLAHE) 的 best.pt
匯出為瀏覽器 ONNX Runtime Web (WebGPU / WASM) 可直接推論的 .onnx 模型。

使用方式（需指定包含 YOLOv7 models/ 模組的 repo 路徑）：
    python scripts/export_web_onnx.py --yolov7-dir /tmp/yolov7-repo
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn


MODELS_TO_EXPORT = [
    {
        "id": "exp5",
        "pt": "results/results_summary_stage3/checkpoints/exp5/best.pt",
        "out": "web/public/models/exp5_tiny_640.onnx",
        "img_size": 640,
    },
    {
        "id": "exp7",
        "pt": "results/results_summary_stage3/checkpoints/exp7/best.pt",
        "out": "web/public/models/exp7_base_640.onnx",
        "img_size": 640,
    },
    {
        "id": "exp12",
        "pt": "results/results_summary_stage3/checkpoints/exp12/best.pt",
        "out": "web/public/models/exp12_tiny_960_clahe.onnx",
        "img_size": 960,
    },
]


def export_single(yolov7_dir: Path, pt_path: Path, out_path: Path, img_size: int, opset: int = 12):
    if str(yolov7_dir) not in sys.path:
        sys.path.insert(0, str(yolov7_dir))

    import models  # noqa: F401
    from models.experimental import attempt_load
    from models.yolo import Detect
    from utils.activations import Hardswish, SiLU

    print(f"\n=== Exporting {pt_path} -> {out_path} (img_size={img_size}) ===")
    device = torch.device("cpu")
    # PyTorch >= 2.6 defaults torch.load(weights_only=True); YOLOv7 checkpoints store nn.Module objects
    orig_load = torch.load
    torch.load = lambda *a, **kw: orig_load(*a, **{**kw, "weights_only": False})
    try:
        model = attempt_load(str(pt_path), map_location=device)
    finally:
        torch.load = orig_load
    model.eval()
    model.float()

    # Replace SiLU / Hardswish with ONNX-exportable versions and enable grid decoding
    for k, m in model.named_modules():
        m._non_persistent_buffers_set = set()
        if isinstance(m, models.common.Conv):
            if isinstance(m.act, nn.Hardswish):
                m.act = Hardswish()
            elif isinstance(m.act, nn.SiLU):
                m.act = SiLU()
        if isinstance(m, Detect):
            m.export = False  # False => output decoded [1, N, 5+nc] with grid & anchor_grid applied!

    # Patch Detect.forward so ONNX trace uses static grid tensors cleanly
    dummy = torch.zeros(1, 3, img_size, img_size, dtype=torch.float32, device=device)

    # Dry run to populate Detect.grid and verify output shape
    with torch.no_grad():
        out = model(dummy)
        pred = out[0] if isinstance(out, (tuple, list)) else out
        print(f"  PyTorch dry-run output shape: {tuple(pred.shape)}")

    # Wrap model so ONNX only outputs the decoded prediction tensor [1, N, 11]
    class DecodedWrapper(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            y = self.inner(x)
            return y[0] if isinstance(y, (tuple, list)) else y

    wrapped = DecodedWrapper(model).eval()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapped,
        dummy,
        str(out_path),
        verbose=False,
        opset_version=opset,
        input_names=["images"],
        output_names=["output"],
        do_constant_folding=True,
    )

    # Check ONNX validity
    onnx_model = onnx.load(str(out_path))
    onnx.checker.check_model(onnx_model)

    # Verify with ONNX Runtime
    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    ort_out = sess.run(None, {"images": np.zeros((1, 3, img_size, img_size), dtype=np.float32)})[0]
    max_diff = float(np.max(np.abs(pred.numpy() - ort_out)))
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"  ONNX verified! shape={ort_out.shape}, max_diff={max_diff:.6f}, size={size_mb:.2f} MB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yolov7-dir", type=str, required=True, help="Path to cloned YOLOv7 repository")
    parser.add_argument("--opset", type=int, default=12)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    yolov7_dir = Path(args.yolov7_dir).resolve()

    for spec in MODELS_TO_EXPORT:
        export_single(
            yolov7_dir=yolov7_dir,
            pt_path=repo_root / spec["pt"],
            out_path=repo_root / spec["out"],
            img_size=spec["img_size"],
            opset=args.opset,
        )


if __name__ == "__main__":
    main()
