"""Draw the evidence behind a reading onto the image, so a human can check it at a glance."""
from __future__ import annotations

import cv2
import numpy as np

_GOOD, _WARN, _BAD = (80, 200, 60), (0, 190, 255), (60, 60, 235)


def _text(img, text, org, color, size=22):
    """OpenCV 5 TrueType rendering (embedded Rubik font, Unicode capable)."""
    cv2.putText(img, text, org, color, cv2.FontFace("sans"), size, 600)


def annotate(img, reading: dict | None, label: str = "", status: str = "ok"):
    out = img.copy()
    color = {"ok": _GOOD, "warn": _WARN, "alert": _BAD}.get(status, _WARN)
    ov = (reading or {}).get("overlay")
    if ov:
        rim = np.array(ov["rim"], np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [rim], True, color, 2, cv2.LINE_AA)
        p, t = tuple(int(v) for v in ov["pivot"]), tuple(int(v) for v in ov["tip"])
        cv2.line(out, p, t, (255, 0, 255), 2, cv2.LINE_AA)
        cv2.circle(out, p, 4, (255, 0, 255), -1, cv2.LINE_AA)
        for key, col in (("scale_start", (255, 160, 0)), ("scale_end", (255, 160, 0))):
            cv2.circle(out, tuple(int(v) for v in ov[key]), 6, col, 2, cv2.LINE_AA)
    if label:
        h = 34
        cv2.rectangle(out, (0, 0), (out.shape[1], h), (20, 20, 20), -1)
        _text(out, label, (8, h - 10), color)
    return out
