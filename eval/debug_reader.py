"""Show what the reader sees on real photos: rectified dial with detections, and the polar ink map.

    python eval/debug_reader.py [--out eval/results/debug] [files...]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gaugeguard.agent.loop import fit_image  # noqa: E402
from gaugeguard.vision import locate_dial, read_gauge  # noqa: E402
from gaugeguard.vision import reader as rd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def panel(img, label):
    loc = locate_dial(img)
    spec = {k: label[k] for k in ("vmin", "vmax", "units", "sweep_deg")}
    r = read_gauge(img, loc["dial"], spec, debug=True) if loc["found"] else None
    if r is None:
        return np.full((480, 1440, 3), 255, np.uint8)
    dbg = r.pop("debug")
    rect = dbg["rect"].copy()
    cv2.circle(rect, (int(rd.C), int(rd.C)), int(rd.R), (0, 200, 255), 1)
    if r["scale"]:
        for a in (r["scale"]["start_deg"], r["scale"]["end_deg"]):
            cv2.circle(rect, tuple(int(v) for v in rd._on_circle(a, dbg["ring_in"])), 7, (255, 160, 0), 2)
    cv2.circle(rect, (int(dbg["ring_in"] + rd.C), int(rd.C)), 3, (0, 255, 0), -1)
    if r.get("needle_deg") is not None:
        cv2.line(rect, (int(rd.C), int(rd.C)), tuple(int(v) for v in rd._on_circle(r["needle_deg"], 0.95 * dbg["ring_in"])),
                 (255, 0, 255), 2)
    seg_dbg = rd._scale_from_segments(dbg["rect"], label["sweep_deg"])
    if seg_dbg:
        for a in (seg_dbg["start_deg"], seg_dbg["end_deg"]):
            cv2.circle(rect, tuple(int(v) for v in rd._on_circle(a, seg_dbg["ring_out"])), 5, (0, 0, 255), -1)
    polar = cv2.cvtColor((np.clip(dbg["polar"] * 3, 0, 1) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    polar = cv2.resize(polar, (480, 480))  # rows = angle (0..360 deg top to bottom), cols = radius
    x = int(dbg["ring_in"] / rd.R * 480)
    cv2.line(polar, (x, 0), (x, 479), (0, 255, 0), 1)
    small = img.copy()
    h, w = small.shape[:2]
    s = 480 / max(h, w)
    small = cv2.resize(small, (int(w * s), int(h * s)))
    small = cv2.copyMakeBorder(small, 0, 480 - small.shape[0], 0, 480 - small.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255))
    out = np.hstack([small, rect, polar])
    txt = f"{label['file'][:28]} truth {label['truth']} read {r['value']} conf {r['confidence']} {','.join(r['flags'])}"
    cv2.rectangle(out, (0, 0), (out.shape[1], 26), (20, 20, 20), -1)
    cv2.putText(out, txt, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "results", "debug"))
    ap.add_argument("--labels", default=os.path.join(HERE, "real", "labels.json"))
    ap.add_argument("--images", default=os.path.join(HERE, "real", "images"))
    ap.add_argument("files", nargs="*")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    labels = [l for l in json.load(open(args.labels, encoding="utf-8")) if l.get("truth") is not None]
    if args.files:
        labels = [l for l in labels if l["file"] in args.files]
    for i in range(0, len(labels), 3):
        rows = [panel(fit_image(cv2.imread(os.path.join(args.images, l["file"]))), l) for l in labels[i:i + 3]]
        cv2.imwrite(os.path.join(args.out, f"debug_{i // 3:02d}.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 82])


if __name__ == "__main__":
    main()
