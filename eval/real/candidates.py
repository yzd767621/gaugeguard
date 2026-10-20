"""Pick crawled photos that may show a readable analog dial, and draw numbered contact sheets for labelling.

Only the dial locator runs here, never the reader, so labels are made without seeing any reading.

    python eval/real/candidates.py --crawl "%USERPROFILE%\\Downloads\\gauge_images"
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gaugeguard.agent.loop import fit_image  # noqa: E402
from gaugeguard.vision import locate_dial  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crawl", required=True)
    ap.add_argument("--per-sheet", type=int, default=12)
    args = ap.parse_args()
    out = os.path.join(args.crawl, "candidates")
    os.makedirs(out, exist_ok=True)
    rows = [json.loads(l) for l in open(os.path.join(args.crawl, "manifest.jsonl"), encoding="utf-8")]
    picked = []
    for m in sorted(rows, key=lambda m: m["sha1"]):
        img = cv2.imread(os.path.join(args.crawl, "images", m["file"]))
        if img is None:
            continue
        img = fit_image(img)
        loc = locate_dial(img)
        if not loc["found"]:
            continue
        d = loc["dial"]
        h, w = img.shape[:2]
        if min(d["axes"]) < 0.22 * min(h, w) or min(d["axes"]) / max(d["axes"]) < 0.6 or d["score"] < 0.6:
            continue
        picked.append((m, img))
    print(f"{len(picked)} candidates of {len(rows)} files")
    index = []
    for i in range(0, len(picked), args.per_sheet):
        tiles = []
        for j, (m, img) in enumerate(picked[i:i + args.per_sheet]):
            n = i + j
            index.append({"id": n, "file": m["file"]})
            h, w = img.shape[:2]
            s = 320 / max(h, w)
            t = cv2.resize(img, (int(w * s), int(h * s)))
            t = cv2.copyMakeBorder(t, 0, 320 - t.shape[0], 0, 320 - t.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255))
            cv2.rectangle(t, (0, 0), (64, 30), (0, 0, 0), -1)
            cv2.putText(t, str(n), (4, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            tiles.append(t)
        while len(tiles) % 4:
            tiles.append(np.full((320, 320, 3), 255, np.uint8))
        sheet = np.vstack([np.hstack(tiles[k:k + 4]) for k in range(0, len(tiles), 4)])
        cv2.imwrite(os.path.join(out, f"sheet_{i // args.per_sheet:03d}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 80])
    json.dump(index, open(os.path.join(out, "index.json"), "w"), indent=0)


if __name__ == "__main__":
    main()
