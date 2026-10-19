"""Time the OpenCV operations GaugeGuard depends on, so two OpenCV builds can be compared.

Run the same command on each build (for example stock opencv-python-headless and COOL on an AWS
Graviton instance) and compare the JSON files:

    python bench/bench_ops.py --out stock.json
    python bench/bench_ops.py --out cool.json

This script makes no claim by itself; it only measures the build it runs on.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gaugeguard.synth import make_sample  # noqa: E402
from gaugeguard.vision import assess_quality, enhance, locate_dial, read_gauge  # noqa: E402


def timeit(fn, repeat):
    fn()  # warm-up
    samples = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000)
    return {"median_ms": round(float(np.median(samples)), 3), "p95_ms": round(float(np.percentile(samples, 95)), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=30)
    ap.add_argument("--out", default="bench.json")
    args = ap.parse_args()

    img, truth = make_sample(1, "tilt", size=960)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    H = np.array([[0.9, 0.1, 20], [-0.05, 0.95, 10], [1e-4, 2e-5, 1]])
    dial = locate_dial(img)["dial"]
    ops = {
        "resize_area_0.5x": lambda: cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA),
        "gaussian_blur_sigma2": lambda: cv2.GaussianBlur(gray, (0, 0), 2.0),
        "adaptive_threshold": lambda: cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                                            cv2.THRESH_BINARY, 151, -4),
        "find_contours": lambda: cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE),
        "warp_perspective": lambda: cv2.warpPerspective(img, H, (960, 960)),
        "morph_close_29": lambda: cv2.morphologyEx(gray, cv2.MORPH_CLOSE,
                                                   cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (29, 29))),
        "pipeline_assess_quality": lambda: assess_quality(img),
        "pipeline_locate_dial": lambda: locate_dial(img),
        "pipeline_read_gauge": lambda: read_gauge(img, dial, truth.spec),
        "pipeline_denoise": lambda: enhance(img, "denoise"),
    }
    result = {"opencv": cv2.__version__, "machine": platform.machine(), "processor": platform.processor(),
              "python": platform.python_version(), "threads": cv2.getNumThreads(),
              "cpu_features": cv2.getCPUFeaturesLine(), "repeat": args.repeat,
              "ops": {name: timeit(fn, args.repeat) for name, fn in ops.items()}}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
