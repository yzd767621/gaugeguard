"""Image-enhancement operators the agent can apply before re-reading a gauge."""
from __future__ import annotations

import cv2
import numpy as np


def _clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    lab[..., 0] = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)


def _brighten(img):
    """Stretch so the 99th percentile reaches near-white, then apply a mild gamma."""
    hi = max(float(np.percentile(img, 99)), 1.0)
    out = np.clip(img.astype(np.float32) * (235.0 / hi), 0, 255) / 255.0
    return (np.power(out, 0.85) * 255).astype(np.uint8)


def _denoise(img):
    return cv2.fastNlMeansDenoisingColored(img, None, 12, 12, 7, 21)


def _sharpen(img):
    soft = cv2.GaussianBlur(img, (0, 0), 3.0)
    return cv2.addWeighted(img, 2.2, soft, -1.2, 0)


def _deglare(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = ((hsv[..., 2] > 245) & (hsv[..., 1] < 30)).astype(np.uint8) * 255
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    return cv2.inpaint(img, mask, 5, cv2.INPAINT_TELEA)


ENHANCE_OPS = {"clahe": _clahe, "brighten": _brighten, "denoise": _denoise, "sharpen": _sharpen,
               "deglare": _deglare}


def enhance(img, op: str):
    if op not in ENHANCE_OPS:
        raise ValueError(f"unknown enhancement {op!r}; choose one of {sorted(ENHANCE_OPS)}")
    return ENHANCE_OPS[op](img)
