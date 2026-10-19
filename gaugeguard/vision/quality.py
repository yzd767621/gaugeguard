"""Image-quality assessment. The agent uses these numbers to decide what to do next."""
from __future__ import annotations

import cv2
import numpy as np

# Thresholds were set on the synthetic development split (seeds < 1000); see docs/TECHNICAL_REPORT.md.
SHARPNESS_MIN = 250.0
BRIGHTNESS_MIN = 60.0
NOISE_RATIO_MAX = 0.025  # noise sigma relative to mean brightness, so dark frames are judged fairly
GLARE_MAX = 0.008
CONTRAST_MIN = 28.0


def _roi(img, dial):
    """Crop to the dial's bounding box when a dial has been located."""
    if not dial:
        return img
    (cx, cy), (a, b) = dial["center"], dial["axes"]
    r = max(a, b)
    h, w = img.shape[:2]
    x0, x1 = int(max(0, cx - r)), int(min(w, cx + r))
    y0, y1 = int(max(0, cy - r)), int(min(h, cy + r))
    return img[y0:y1, x0:x1] if x1 - x0 > 16 and y1 - y0 > 16 else img


def assess_quality(img, dial=None) -> dict:
    roi = _roi(img, dial)
    scale = 384.0 / max(roi.shape[:2])
    small = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    p5, p95 = np.percentile(gray, [5, 95])
    contrast = float(p95 - p5)
    # Immerkaer's fast noise estimate: response to a Laplacian-difference mask
    mask = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    resp = cv2.filter2D(gray.astype(np.float32), -1, mask)
    noise = float(np.sqrt(np.pi / 2) * np.abs(resp[1:-1, 1:-1]).mean() / 6.0)
    glare = float(((hsv[..., 2] > 250) & (hsv[..., 1] < 25)).mean())

    issues = []
    if brightness < BRIGHTNESS_MIN:
        issues.append("dark")
    if noise / max(brightness, 1.0) > NOISE_RATIO_MAX:
        issues.append("noisy")
    # noise inflates Laplacian variance and clips to white, so blur and glare are only judged on clean frames
    if sharpness < SHARPNESS_MIN and "noisy" not in issues:
        issues.append("blurry")
    if glare > GLARE_MAX and "noisy" not in issues:
        issues.append("glare")
    if contrast < CONTRAST_MIN:
        issues.append("low_contrast")
    return {"sharpness": round(sharpness, 1), "brightness": round(brightness, 1),
            "contrast": round(contrast, 1), "noise": round(noise, 2), "glare_fraction": round(glare, 4),
            "issues": issues, "scope": "dial" if dial else "frame"}
