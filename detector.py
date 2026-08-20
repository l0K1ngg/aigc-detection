import io
import json
import math
import os
import time
from typing import Any, Dict, Tuple

import numpy as np
import onnxruntime as ort
from PIL import Image

MAX_PIXELS = 50_000_000
MIN_EDGE_ALLOWED = 32
TARGET_SIZE = 224


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class AIGCDetector:
    def __init__(self, serve_dir: str = "serve"):
        self.serve_dir = serve_dir
        self.model_path = os.path.join(serve_dir, "model.onnx")
        self.thresholds_path = os.path.join(serve_dir, "thresholds.json")
        self.card_path = os.path.join(serve_dir, "model_card.json")
        self.golden_path = os.path.join(serve_dir, "golden.json")

        self.session = None
        self.thresholds: Dict[str, Any] = {}
        self.model_card: Dict[str, Any] = {}
        self.is_ready = False

        self._load()

    def _load(self):
        if not (os.path.exists(self.model_path) and os.path.exists(self.thresholds_path)):
            return

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            self.model_path,
            sess_options=opts,
            providers=["CPUExecutionProvider"]
        )

        with open(self.thresholds_path, "r", encoding="utf-8") as f:
            self.thresholds = json.load(f)

        if os.path.exists(self.card_path):
            with open(self.card_path, "r", encoding="utf-8") as f:
                self.model_card = json.load(f)

        self._verify_golden()
        self.is_ready = True

    def _verify_golden(self):
        if not os.path.exists(self.golden_path):
            return

        with open(self.golden_path, "r", encoding="utf-8") as f:
            golden_data = json.load(f)

        test_input = np.zeros((1, TARGET_SIZE, TARGET_SIZE, 3), dtype=np.uint8)
        ort_inputs = {self.session.get_inputs()[0].name: test_input}
        out_logit = float(self.session.run(None, ort_inputs)[0].flatten()[0])

        expected_logit = float(golden_data.get("golden_logit", 0.0))
        tolerance = float(golden_data.get("tolerance", 1e-4))

        if abs(out_logit - expected_logit) > tolerance:
            raise RuntimeError(
                f"Golden fingerprint mismatch! Got {out_logit:.6f}, expected {expected_logit:.6f}"
            )

    def preprocess_image(self, image_bytes: bytes) -> Tuple[np.ndarray, Dict[str, Any], float]:
        t0 = time.perf_counter()
        img = Image.open(io.BytesIO(image_bytes))

        if img.width * img.height > MAX_PIXELS:
            raise ValueError("TOO_MANY_PIXELS")

        min_edge = min(img.width, img.height)
        if min_edge < MIN_EDGE_ALLOWED:
            raise ValueError("IMAGE_TOO_SMALL")

        exif_present = bool(img.getexif()) if hasattr(img, "getexif") else False
        upscaled = min_edge < TARGET_SIZE

        img = img.convert("RGB")
        w, h = img.size

        scale = 256.0 / min(w, h)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)

        left = (new_w - TARGET_SIZE) // 2
        top = (new_h - TARGET_SIZE) // 2
        img = img.crop((left, top, left + TARGET_SIZE, top + TARGET_SIZE))

        arr = np.array(img, dtype=np.uint8)
        arr = np.expand_dims(arr, axis=0)

        dt = (time.perf_counter() - t0) * 1000.0
        meta = {
            "decoded_mode": "RGB",
            "source_min_edge": min_edge,
            "upscaled": upscaled,
            "exif_orientation_present": exif_present,
        }
        return arr, meta, dt

    def infer(self, tensor_uint8: np.ndarray) -> Tuple[float, float, float]:
        t0 = time.perf_counter()
        ort_inputs = {self.session.get_inputs()[0].name: tensor_uint8}
        raw_out = self.session.run(None, ort_inputs)[0].flatten()[0]
        logit = float(raw_out)
        prob = sigmoid(logit)
        dt = (time.perf_counter() - t0) * 1000.0
        return logit, prob, dt

    def decide(self, score: float, tier: str = "1pct") -> Tuple[str, Dict[str, Any]]:
        tier_cfg = self.thresholds.get("tiers", {}).get(tier, self.thresholds.get(tier, {}))
        t_low = float(tier_cfg.get("t_low", 0.3936))
        t_high = float(tier_cfg.get("t_high", 0.6389))

        if score >= t_high:
            verdict = "generated"
        elif score <= t_low:
            verdict = "no_traces_detected"
        else:
            verdict = "uncertain"

        return verdict, {"tier": tier, "t_low": t_low, "t_high": t_high}
