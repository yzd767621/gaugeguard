"""Read a located dial: rectify it, find the scale, find the needle, convert to a value.

The dial rim (an ellipse in the image) and the needle pivot together fix the
plane-to-image homography up to a rotation about the pivot. That leftover
rotation is resolved by the gap in the tick ring, so the reading needs no
assumption that the camera was level or front-on.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

N = 480  # rectified image side
R = 200.0  # rim radius in the rectified image
C = N / 2.0
N_ANG = 720  # polar angular samples (0.5 degrees each)
DEG = N_ANG / 360.0


# ------------------------------------------------------------ rectification

def _affine_h(dial):
    """Homography (affine) that maps the rim ellipse to the circle of radius R centred at (C, C)."""
    (cx, cy), (sx, sy), ang = dial["center"], dial["axes"], math.radians(dial["angle"])
    rot = np.array([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]])
    A = rot @ np.diag([R / sx, R / sy]) @ rot.T
    t = np.array([C, C]) - A @ np.array([cx, cy])
    H = np.eye(3)
    H[:2, :2], H[:2, 2] = A, t
    return H


def _disc_translation(e):
    """Projective map of the rim circle onto itself that moves point ``e`` (rectified px) to the centre."""
    ex, ey = (e[0] - C) / R, (e[1] - C) / R
    s = math.hypot(ex, ey)
    if s < 1e-6:
        return np.eye(3)
    s = min(s, 0.6)
    phi = math.atan2(ey, ex)
    rot = np.array([[math.cos(phi), -math.sin(phi), 0], [math.sin(phi), math.cos(phi), 0], [0, 0, 1]])
    K = np.array([[1, 0, -s], [0, math.sqrt(1 - s * s), 0], [-s, 0, 1]])
    norm = np.array([[1 / R, 0, -C / R], [0, 1 / R, -C / R], [0, 0, 1]])
    return np.linalg.inv(norm) @ rot @ K @ rot.T @ norm


def _warp(img, H):
    return cv2.warpPerspective(img, H, (N, N), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def _disc_mask(radius):
    m = np.zeros((N, N), np.uint8)
    cv2.circle(m, (int(C), int(C)), int(radius), 255, -1)
    return m


def _light_face(mono):
    vals = mono[_disc_mask(0.8 * R) > 0]
    p5, med, p95 = np.percentile(vals, [5, 50, 95])
    return (p95 - med) <= (med - p5), float(med)


def _find_pivot(mono, light):
    """Needle hub = the thickest ink blob near the centre. Returns (x, y) or None."""
    src = mono if light else 255 - mono
    face = float(np.median(src[_disc_mask(0.8 * R) > 0]))
    dark = ((src < 0.55 * face) & (_disc_mask(0.4 * R) > 0)).astype(np.uint8)
    if dark.sum() < 20:
        return None
    dist = cv2.distanceTransform(dark, cv2.DIST_L2, 5)
    peak = float(dist.max())
    if peak < 0.025 * R:
        return None
    blob = (dist > 0.6 * peak).astype(np.uint8)
    n, labels, stats, cents = cv2.connectedComponentsWithStats(blob)
    py, px = np.unravel_index(int(dist.argmax()), dist.shape)
    return tuple(float(v) for v in cents[labels[py, px]])


def _pivot_plausible(dial, pivot, offset):
    """A pivot away from the rim centre is only possible on a tilted dial, along the tilt direction."""
    sx, sy = dial["axes"]
    sin_tilt = math.sqrt(max(0.0, 1.0 - (min(sx, sy) / max(sx, sy)) ** 2))
    if offset > 0.35 * sin_tilt + 0.02:
        return False
    if offset > 0.03:
        minor = math.radians(dial["angle"] + (0.0 if sx < sy else 90.0))
        phi = math.atan2(pivot[1] - C, pivot[0] - C)
        return abs(math.cos(phi - minor)) > 0.85
    return True


# ---------------------------------------------------------------- ink / polar

def _ink(rect, light):
    """Relative darkness of thin structures (needle, ticks, digits), robust to uneven lighting."""
    mono = rect.min(axis=2) if light else 255 - rect.max(axis=2)
    k = 2 * int(0.075 * R) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    closed = cv2.morphologyEx(mono, cv2.MORPH_CLOSE, kernel)
    ink = (closed.astype(np.float32) - mono.astype(np.float32)) / (closed.astype(np.float32) + 12.0)
    ink[_disc_mask(0.985 * R) == 0] = 0
    return np.clip(ink, 0, 1)


def _polar(ink):
    return cv2.warpPolar(ink, (int(R), N_ANG), (C, C), R, cv2.WARP_POLAR_LINEAR + cv2.INTER_LINEAR)


def _circ_smooth(x, width):
    width = max(1, int(round(width)))
    if width <= 1:
        return x
    pad = np.concatenate([x[-width:], x, x[:width]])
    return np.convolve(pad, np.ones(width) / width, mode="same")[width:-width]


def _runs(binary):
    """Circular runs of a boolean array as (start, length, value); a constant array is one run."""
    n = len(binary)
    change = np.flatnonzero(binary != np.roll(binary, 1))
    if len(change) == 0:
        return [(0, n, bool(binary[0]))]
    out = []
    for i, s in enumerate(change):
        e = change[(i + 1) % len(change)]
        out.append((int(s), int((e - s) % n) or n, bool(binary[s])))
    return out


# --------------------------------------------------------------------- scale

def _find_scale(polar, expected_sweep):
    r_lo, r_hi = int(0.45 * R), int(0.975 * R)
    band = polar[:, r_lo:r_hi]
    smooth = np.apply_along_axis(_circ_smooth, 0, band, 9 * DEG)
    tickness = np.convolve(np.abs(band - smooth).mean(axis=0), np.ones(5) / 5, mode="same")
    if tickness.max() < 0.01:
        return None
    # digits also score highly, but ticks are always the outermost printed ring
    strong = np.flatnonzero(tickness >= 0.5 * tickness.max())
    outer = strong[strong >= strong[-1] - int(0.08 * R)]
    best = int(outer[tickness[outer].argmax()])
    lo = hi = best
    while lo > 0 and tickness[lo - 1] > 0.5 * tickness[best]:
        lo -= 1
    while hi < len(tickness) - 1 and tickness[hi + 1] > 0.5 * tickness[best]:
        hi += 1
    ring_in, ring_out = r_lo + lo, r_lo + hi + 1
    occ = _circ_smooth(polar[:, ring_in:ring_out].mean(axis=1), 1.0 * DEG)
    thresh = 0.35 * float(np.percentile(occ, 95))
    if thresh < 0.01:
        return None
    on = occ > thresh
    # drop weak blips (noise) so they cannot split the scale gap
    runs = _runs(on)
    peaks = [float(occ.take(range(s, s + l), mode="wrap").max()) for s, l, v in runs if v]
    if not peaks:
        return None
    typical = float(np.median(peaks))
    for s, l, v in runs:
        idx = np.arange(s, s + l) % N_ANG
        if v and (occ[idx].max() < 0.5 * typical or l < 2):
            on[idx] = False
    gaps = [(s, l) for s, l, v in _runs(on) if not v and l >= 25 * DEG]
    if not gaps:
        return None
    expected_gap = (360.0 - expected_sweep) * DEG
    gs, gl = min(gaps, key=lambda g: abs(g[1] - expected_gap))

    def tick_centre(first, step):
        """Centroid of the tick that borders the gap, walking ``step`` from index ``first``."""
        idx = [first % N_ANG]
        while on[(idx[-1] + step) % N_ANG] and len(idx) < 6 * DEG:
            idx.append((idx[-1] + step) % N_ANG)
        w = occ[idx]
        off = np.arange(len(idx)) * step
        return (first + float((w * off).sum() / w.sum())) / DEG

    start = tick_centre(gs + gl, +1) % 360.0
    end = tick_centre(gs - 1, -1) % 360.0
    sweep = (end - start) % 360.0
    mismatch = abs(sweep - expected_sweep)
    conf = float(np.clip(1.0 - (mismatch - 6.0) / 24.0, 0.0, 1.0))
    flags = []
    if len(gaps) > 1:
        conf *= 0.85
        flags.append("scale_partially_hidden")
    if mismatch > 6.0:
        flags.append("scale_sweep_mismatch")
    return {"start_deg": start, "end_deg": end, "sweep_deg": sweep, "ring_in": ring_in, "ring_out": ring_out,
            "confidence": conf, "flags": flags}


# -------------------------------------------------------------------- needle

def _find_needle(polar, ring_in, allowed=None):
    r0, r1 = int(0.28 * ring_in), int(0.62 * ring_in)
    prof = _circ_smooth(np.median(polar[:, r0:r1], axis=1), 1.0 * DEG)
    search = prof.copy()
    if allowed is not None:
        start, sweep = allowed
        ang = np.arange(N_ANG) / DEG
        rel = (ang - (start - 8.0)) % 360.0
        search[rel > sweep + 16.0] = 0
    peak = int(search.argmax())
    height = float(search[peak])
    if height <= 1e-4:
        return None
    win = int(3 * DEG)
    idx = np.arange(peak - win, peak + win + 1)
    w = np.clip(prof.take(idx, mode="wrap") - 0.3 * height, 0, None)
    angle = float((w * idx).sum() / w.sum()) / DEG % 360.0
    far = np.abs((np.arange(N_ANG) - peak + N_ANG // 2) % N_ANG - N_ANG // 2) > 12 * DEG
    rival = float(search[far].max()) if far.any() else 0.0
    conf = float(np.clip((height - rival) / height, 0, 1) * np.clip(height / 0.25, 0, 1))
    return {"angle_deg": angle, "strength": height, "confidence": conf}


# ---------------------------------------------------------------------- main

def _to_image(H, pts):
    p = cv2.perspectiveTransform(np.array(pts, np.float64).reshape(-1, 1, 2), np.linalg.inv(H))
    return [[round(float(x), 1), round(float(y), 1)] for x, y in p.reshape(-1, 2)]


def _on_circle(deg, radius):
    a = math.radians(deg)
    return [C + radius * math.cos(a), C + radius * math.sin(a)]


def read_gauge(img, dial: dict, spec: dict | None = None, projective: bool = True) -> dict:
    """Read a dial previously found by :func:`locate_dial`.

    ``spec`` may carry ``vmin``, ``vmax``, ``units`` and ``sweep_deg``; without ``vmin``/``vmax``
    only the needle position as a fraction of the scale is returned.
    """
    spec = spec or {}
    expected_sweep = float(spec.get("sweep_deg", 270.0))
    flags = []
    H = _affine_h(dial)
    rect = _warp(img, H)
    mono = rect.min(axis=2)
    light, _ = _light_face(mono)
    pivot_offset, pivot_penalty = 0.0, 1.0
    pivot = _find_pivot(mono if light else rect.max(axis=2), light)
    if pivot is None:
        flags.append("pivot_not_found")
    elif projective:
        pivot_offset = math.hypot(pivot[0] - C, pivot[1] - C) / R
        if not _pivot_plausible(dial, pivot, pivot_offset):
            # e.g. glare on the hub: trust the rim centre instead and say so
            flags.append("pivot_inconsistent")
            pivot_penalty = 0.6
        elif pivot_offset > 0.012:
            H = _disc_translation(pivot) @ H
            rect = _warp(img, H)
    polar = _polar(_ink(rect, light))

    scale = _find_scale(polar, expected_sweep)
    if scale is None:
        flags.append("scale_not_found")
        ring_in = int(0.72 * R)
        start, sweep, scale_conf = float(spec.get("start_deg", 135.0)), expected_sweep, 0.3
    else:
        flags += scale["flags"]
        ring_in, start, sweep, scale_conf = scale["ring_in"], scale["start_deg"], scale["sweep_deg"], scale["confidence"]

    needle = _find_needle(polar, ring_in, (start, sweep) if scale else None)
    result = {"ok": False, "value": None, "units": spec.get("units"), "fraction": None, "confidence": 0.0,
              "flags": flags, "components": {"locate": dial.get("score", 0.0), "scale": round(scale_conf, 3),
                                             "needle": 0.0},
              "scale": None if scale is None else {k: round(float(scale[k]), 2) for k in
                                                   ("start_deg", "end_deg", "sweep_deg")},
              "pivot_offset": round(pivot_offset, 4), "light_face": bool(light), "overlay": None}
    if needle is None:
        flags.append("needle_not_found")
        return result

    rel = (needle["angle_deg"] - start) % 360.0
    if rel > sweep + (360.0 - sweep) / 2.0:
        rel -= 360.0  # just before the start of the scale
    fraction = rel / sweep
    if fraction < -0.03 or fraction > 1.03:
        flags.append("needle_outside_scale")
    confidence = needle["confidence"] * pivot_penalty * scale_conf * (0.6 + 0.4 * float(dial.get("score", 0.5)))
    result.update(ok=True, fraction=round(fraction, 4), confidence=round(float(confidence), 3),
                  needle_deg=round(needle["angle_deg"], 2))
    result["components"]["needle"] = round(needle["confidence"], 3)
    if "vmin" in spec and "vmax" in spec:
        result["value"] = round(spec["vmin"] + fraction * (spec["vmax"] - spec["vmin"]), 4)
    rim = [_on_circle(a, R) for a in range(0, 360, 10)]
    pts = _to_image(H, [[C, C], _on_circle(needle["angle_deg"], 0.95 * ring_in),
                        _on_circle(start, ring_in), _on_circle(start + sweep, ring_in)] + rim)
    result["overlay"] = {"pivot": pts[0], "tip": pts[1], "scale_start": pts[2], "scale_end": pts[3], "rim": pts[4:]}
    return result
