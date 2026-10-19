"""Find the dial rim as an ellipse (a circle seen from an arbitrary viewpoint)."""
from __future__ import annotations

import math

import cv2
import numpy as np

MAX_CANDIDATES = 4


def _ellipse_residual(points, ellipse):
    """Mean |normalised radius - 1| of points relative to an ellipse; 0 means a perfect fit."""
    (cx, cy), (w, h), ang = ellipse
    a = math.radians(ang)
    ca, sa = math.cos(a), math.sin(a)
    p = points.reshape(-1, 2).astype(np.float64) - (cx, cy)
    u = (p[:, 0] * ca + p[:, 1] * sa) / max(w / 2, 1e-6)
    v = (-p[:, 0] * sa + p[:, 1] * ca) / max(h / 2, 1e-6)
    return float(np.abs(np.hypot(u, v) - 1.0).mean())


def _score_contour(contour, shape):
    h, w = shape[:2]
    area = cv2.contourArea(contour)
    if area < 0.012 * h * w or len(contour) < 30:
        return None
    hull = cv2.convexHull(contour)
    if len(hull) < 8:
        return None
    ellipse = cv2.fitEllipse(hull)
    (cx, cy), (d1, d2), ang = ellipse
    major, minor = max(d1, d2) / 2, min(d1, d2) / 2
    if minor < 0.06 * min(h, w) or major > 0.62 * min(h, w) or minor / major < 0.45:
        return None
    if not (0.02 * w < cx < 0.98 * w and 0.02 * h < cy < 0.98 * h):
        return None
    ell_area = math.pi * major * minor
    solidity = cv2.contourArea(hull) / ell_area
    fill = area / ell_area
    resid = _ellipse_residual(hull, ellipse)
    if not 0.80 < solidity < 1.12 or fill < 0.55:
        return None
    score = max(0.0, 1.0 - resid / 0.06) * min(1.0, solidity) * min(1.0, fill / 0.9)
    if score <= 0.05:
        return None
    return {"center": [float(cx), float(cy)], "axes": [float(d1 / 2), float(d2 / 2)], "angle": float(ang),
            "score": round(float(score), 3), "method": "contour"}


def _masks(gray):
    blur = cv2.GaussianBlur(gray, (0, 0), 2.0)
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    block = (min(gray.shape[:2]) // 6) | 1
    adaptive = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block, -4)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    for m in (otsu, adaptive):
        yield cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
        yield cv2.morphologyEx(cv2.bitwise_not(m), cv2.MORPH_OPEN, kernel)


def _dedupe(cands):
    kept = []
    for c in sorted(cands, key=lambda c: -c["score"]):
        r = max(c["axes"])
        if any(math.dist(c["center"], k["center"]) < 0.08 * r and abs(max(k["axes"]) - r) < 0.06 * r
               for k in kept):
            continue
        kept.append(c)
    return kept[:MAX_CANDIDATES]


def _locate_contour(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cands = []
    for mask in _masks(gray):
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        for c in contours:
            cand = _score_contour(c, img.shape)
            if cand:
                cands.append(cand)
    return _dedupe(cands)


def _locate_hough(img):
    """Circle voting; survives a rim that is partly occluded but assumes a near front-on view."""
    gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (0, 0), 2.0)
    m = min(gray.shape[:2])
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT_ALT, dp=1.5, minDist=m * 0.2, param1=120, param2=0.6,
                               minRadius=int(m * 0.1), maxRadius=int(m * 0.6))
    if circles is None:
        return []
    out = []
    for i, (x, y, r) in enumerate(circles.reshape(-1, 3)[:MAX_CANDIDATES]):
        out.append({"center": [float(x), float(y)], "axes": [float(r), float(r)], "angle": 0.0,
                    "score": round(0.6 - 0.1 * i, 3), "method": "hough"})
    return out


def locate_dial(img, method: str = "contour") -> dict:
    """Return ``{"found", "dial", "candidates"}``. ``method`` is "contour" or "hough"."""
    if method not in ("contour", "hough"):
        raise ValueError("method must be 'contour' or 'hough'")
    cands = _locate_contour(img) if method == "contour" else _locate_hough(img)
    return {"found": bool(cands), "dial": cands[0] if cands else None, "candidates": cands, "method": method}
