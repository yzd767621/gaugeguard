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
CENTRE_SUPPORT_MIN = 0.25  # share of radial segment length that must agree on the voted centre
STANDARD_ENOUGH = 0.6  # confidence at which the standard pass is trusted without trying the fallbacks


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


def _vote_centre(rect):
    """Dial centre from the scale itself: tick marks are radial, so their lines meet at the needle pivot.

    Line segments (LSD) in the outer annulus that point roughly at the rectified centre each vote along their
    own line; the peak of the votes is the centre. Returns ((x, y), support) or None. ``support`` is the share of
    the segments' length that passes within 2% of R of the peak.
    """
    gray = cv2.cvtColor(rect, cv2.COLOR_BGR2GRAY)
    segs = cv2.createLineSegmentDetector().detect(gray)[0]
    if segs is None or len(segs) < 8:
        return None
    x1, y1, x2, y2 = segs.reshape(-1, 4).T.astype(np.float64)
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    length = np.hypot(x2 - x1, y2 - y1)
    dx, dy = (x2 - x1) / np.maximum(length, 1e-6), (y2 - y1) / np.maximum(length, 1e-6)
    rr = np.hypot(mx - C, my - C)
    radial = np.abs(dx * (mx - C) + dy * (my - C)) / np.maximum(rr, 1e-6)  # |cos| to the radial direction
    keep = (length > 0.025 * R) & (length < 0.3 * R) & (rr > 0.4 * R) & (rr < 1.02 * R) & (radial > 0.8)
    if keep.sum() < 8:
        return None
    mx, my, dx, dy, length = mx[keep], my[keep], dx[keep], dy[keep], length[keep]
    grid = np.arange(C - 0.35 * R, C + 0.35 * R + 1, 2.0)
    gx, gy = np.meshgrid(grid, grid)
    dist = np.abs((gx[..., None] - mx) * dy - (gy[..., None] - my) * dx)  # point-to-line distance
    sigma = 0.012 * R
    votes = (np.exp(-0.5 * (dist / sigma) ** 2) * length).sum(axis=2)
    iy, ix = np.unravel_index(int(votes.argmax()), votes.shape)
    cx, cy = float(gx[iy, ix]), float(gy[iy, ix])
    fine = np.arange(-2.5, 2.51, 0.25)  # refine to a quarter pixel around the coarse peak
    fx, fy = np.meshgrid(cx + fine, cy + fine)
    fd = np.abs((fx[..., None] - mx) * dy - (fy[..., None] - my) * dx)
    fv = (np.exp(-0.5 * (fd / sigma) ** 2) * length).sum(axis=2)
    jy, jx = np.unravel_index(int(fv.argmax()), fv.shape)
    cx, cy = float(fx[jy, jx]), float(fy[jy, jx])
    near = np.abs((cx - mx) * dy - (cy - my) * dx) < 0.02 * R
    support = float(length[near].sum() / length.sum())
    return (cx, cy), support


def _tick_ring_points(rect):
    """Outer ends of the radial tick segments in the densest tick ring (rectified px), or None."""
    gray = cv2.cvtColor(rect, cv2.COLOR_BGR2GRAY)
    segs = cv2.createLineSegmentDetector().detect(gray)[0]
    if segs is None:
        return None
    s = segs.reshape(-1, 4).astype(np.float64)
    p1, p2 = s[:, :2], s[:, 2:]
    r1, r2 = np.hypot(*(p1 - C).T), np.hypot(*(p2 - C).T)
    outer = np.where((r2 > r1)[:, None], p2, p1)
    inner_r, outer_r = np.minimum(r1, r2), np.maximum(r1, r2)
    d = (p2 - p1) / np.maximum(np.hypot(*(p2 - p1).T), 1e-6)[:, None]
    m = (p1 + p2) / 2 - C
    radial = np.abs((d * m).sum(axis=1)) / np.maximum(np.hypot(*m.T), 1e-6)
    keep = ((radial > 0.9) & (outer_r - inner_r > 0.02 * R) & (outer_r - inner_r < 0.25 * R)
            & (outer_r > 0.45 * R) & (outer_r < 1.05 * R))
    if keep.sum() < 12:
        return None
    pts, rad = outer[keep], outer_r[keep]
    hist, edges = np.histogram(rad, bins=np.arange(0.45 * R, 1.08 * R, 0.02 * R))
    ring = edges[int(np.convolve(hist, [1, 2, 1], mode="same").argmax())] + 0.01 * R
    sel = np.abs(rad - ring) < 0.07 * R
    return pts[sel].astype(np.float32) if sel.sum() >= 12 else None


def _fit_ellipse_robust(pts):
    """cv2.fitEllipse with two rounds of outlier removal. Returns ((cx, cy), (w, h), angle) or None."""
    pts = np.asarray(pts, np.float32)
    if len(pts) < 12:
        return None
    for _ in range(3):
        (ex, ey), (ea, eb), eang = cv2.fitEllipse(pts)
        t = math.radians(eang)
        u = (pts[:, 0] - ex) * math.cos(t) + (pts[:, 1] - ey) * math.sin(t)
        v = -(pts[:, 0] - ex) * math.sin(t) + (pts[:, 1] - ey) * math.cos(t)
        res = np.abs(np.hypot(u / max(ea / 2, 1e-6), v / max(eb / 2, 1e-6)) - 1)
        good = res < max(0.03, 3 * float(np.median(res)))
        if good.sum() < 12 or good.all():
            break
        pts = pts[good]
    return (ex, ey), (ea, eb), eang


TICK_RING_AT = 0.88  # where the refitted tick ring lands in the rectified image, as a fraction of R


def _refit_dial(img, H, rect):
    """A dial ellipse (image coordinates) fitted to the tick ring, when the ring found under ``H`` is not
    already a centred circle. The rim found by the locator can be a bezel or case edge whose outline is not
    the dial plane's; the tick ring is. Returns a dial dict or None."""
    pts = _tick_ring_points(rect)
    if pts is None:
        return None
    fit = _fit_ellipse_robust(pts)
    if fit is None:
        return None
    (ex, ey), (ea, eb), _ = fit
    ratio, off = min(ea, eb) / max(ea, eb), math.hypot(ex - C, ey - C) / R
    if ratio > 0.97 and off < 0.015:
        return None  # already a centred circle: nothing to fix
    img_pts = cv2.perspectiveTransform(pts.reshape(-1, 1, 2).astype(np.float64), np.linalg.inv(H)).reshape(-1, 2)
    fit = _fit_ellipse_robust(img_pts)
    if fit is None:
        return None
    (cx, cy), (w, h), ang = fit
    if min(w, h) / max(w, h) < 0.45:
        return None
    return {"center": [float(cx), float(cy)], "axes": [float(w / 2 / TICK_RING_AT), float(h / 2 / TICK_RING_AT)],
            "angle": float(ang), "score": None}


def _scale_from_segments(rect, expected_sweep):
    """Scale start/end from the tick marks themselves: segments that point exactly at the centre.

    Digits, logos and reflections rarely produce segments aimed at the pivot, so the angular extent of these
    segments in their densest ring is the printed scale, and the largest empty arc is the gap.
    """
    gray = cv2.cvtColor(rect, cv2.COLOR_BGR2GRAY)
    segs = cv2.createLineSegmentDetector().detect(gray)[0]
    if segs is None:
        return None
    s = segs.reshape(-1, 4).astype(np.float64)
    p1, p2 = s[:, :2] - C, s[:, 2:] - C
    r1, r2 = np.hypot(*p1.T), np.hypot(*p2.T)
    d = p2 - p1
    length = np.hypot(*d.T)
    miss = np.abs(p1[:, 0] * d[:, 1] - p1[:, 1] * d[:, 0]) / np.maximum(length, 1e-6)  # line's distance to centre
    inner, outer = np.minimum(r1, r2), np.maximum(r1, r2)
    keep = (miss < 0.02 * R) & (length > 0.02 * R) & (length < 0.25 * R) & (inner > 0.4 * R) & (outer < 1.05 * R)
    if keep.sum() < 15:
        return None
    mid = (p1 + p2)[keep] / 2
    ang = np.degrees(np.arctan2(mid[:, 1], mid[:, 0])) % 360.0
    inner, outer = inner[keep], outer[keep]
    hist, edges = np.histogram(outer, bins=np.arange(0.4 * R, 1.08 * R, 0.02 * R))
    ring = edges[int(np.convolve(hist, [1, 2, 1], mode="same").argmax())] + 0.01 * R
    sel = np.abs(outer - ring) < 0.05 * R
    if sel.sum() < 15:
        return None
    a = np.sort(ang[sel])
    for _ in range(2):  # drop isolated segments (screws, digits) that are not part of the regular tick run
        gaps = np.diff(np.concatenate([a, a[:1] + 360.0]))
        typical = float(np.median(gaps))
        before = np.roll(gaps, 1)
        isolated = np.minimum(before, gaps) > max(2.5 * typical, 2.0)
        if not isolated.any() or (~isolated).sum() < 15:
            break
        a = a[~isolated]
    gaps = np.diff(np.concatenate([a, a[:1] + 360.0]))
    order = np.argsort(gaps)[::-1]
    gi = int(order[0])
    gap = float(gaps[gi])
    # walk the arc from the gap: start .. end in increasing angle
    arc = np.concatenate([a[gi + 1:], a[:gi + 1] + 360.0])
    typical = float(np.median(np.diff(arc)))
    for _ in range(4):  # trim stray marks at either end (screws, the stop pin) that make the sweep too long
        sweep = arc[-1] - arc[0]
        if len(arc) < 16 or sweep - expected_sweep < 6.0:
            break
        head, tail = arc[1] - arc[0], arc[-1] - arc[-2]
        if max(head, tail) < 1.5 * typical:
            break
        trimmed = arc[1:] if head >= tail else arc[:-1]
        if trimmed[-1] - trimmed[0] < expected_sweep - 6.0:
            break  # would cut into the scale itself
        arc = trimmed
    start, end = float(arc[0] % 360.0), float(arc[-1] % 360.0)
    sweep = (end - start) % 360.0
    mismatch = abs(sweep - expected_sweep)
    conf = float(np.clip(1.0 - (mismatch - 6.0) / 24.0, 0.0, 1.0))
    # the gap must clearly stand out from the spacing between ticks
    second = float(gaps[order[1]]) if len(gaps) > 1 else 0.0
    if second > 0.5 * gap:
        conf *= 0.7
    ring_in = int(np.median(inner[sel]))
    return {"start_deg": start, "end_deg": end, "sweep_deg": sweep, "ring_in": ring_in,
            "ring_out": int(ring + 0.01 * R), "confidence": conf, "flags": ["scale_from_tick_segments"]}


def _shift(p):
    """Translation that moves rectified point ``p`` to the centre."""
    return np.array([[1, 0, C - p[0]], [0, 1, C - p[1]], [0, 0, 1]], np.float64)


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

def _reach(polar, ring_in):
    """Per direction, how far (as a fraction of ``ring_in``) ink runs unbroken outward from the hub.

    Measured out to the rim, so a pointer that runs into the tick ring scores above 1. A counterweight tail
    stops short and printed digits are not attached to the hub.
    """
    band = polar[:, :int(0.97 * R)]
    inner = polar[:, :max(int(ring_in), 2)]
    on = band > max(0.08, 0.35 * float(np.percentile(inner, 99.5)))
    on = on | np.roll(on, 1, axis=0) | np.roll(on, -1, axis=0)  # a thin needle can fall between samples
    start = int(0.12 * ring_in)
    off = ~on[:, start:]
    g = max(3, int(0.03 * ring_in))  # a break this long ends the run
    if off.shape[1] <= g:
        return np.zeros(len(polar))
    cs = np.cumsum(np.pad(off.astype(np.int32), ((0, 0), (1, 0))), axis=1)
    gap = (cs[:, g:] - cs[:, :-g]) >= g
    first = np.where(gap.any(axis=1), gap.argmax(axis=1), off.shape[1])
    reach = (start + first) / float(ring_in)
    return np.maximum.reduce([np.roll(reach, k) for k in range(-int(DEG), int(DEG) + 1)])


def _ang_dist(i, j):
    return np.abs((np.asarray(i) - j + N_ANG // 2) % N_ANG - N_ANG // 2)


def _find_needle(polar, ring_in, allowed=None):
    r0, r1 = int(0.28 * ring_in), int(0.62 * ring_in)
    prof = _circ_smooth(np.median(polar[:, r0:r1], axis=1), 1.0 * DEG)
    reach = _reach(polar, ring_in)
    score = prof * np.clip(reach / 0.6, 0, 1) ** 2  # strength counts only where ink reaches out from the hub
    search = score.copy()
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
    w = np.clip(prof.take(idx, mode="wrap") - 0.3 * float(prof[peak]), 0, None)
    angle = float((w * idx).sum() / max(w.sum(), 1e-9)) / DEG % 360.0
    k = np.arange(N_ANG)
    # The far end of the same needle (counterweight) is not a rival direction; it only raises the question of
    # which end is the pointer, which the reach answers.
    opposite = _ang_dist(k, peak + N_ANG // 2) <= 25 * DEG
    end_conf = float(np.clip((reach[peak] - reach[opposite].max()) / 0.2, 0, 1))
    others = (_ang_dist(k, peak) > 12 * DEG) & ~opposite
    rival = float(search[others].max()) if others.any() else 0.0
    conf = float(np.clip((height - rival) / height, 0, 1) * np.clip(float(prof[peak]) / 0.25, 0, 1) * end_conf)
    return {"angle_deg": angle, "strength": height, "confidence": conf,
            "reach": round(float(reach[peak]), 3), "end_margin": round(float(end_conf), 3)}


# ---------------------------------------------------------------------- main

def _to_image(H, pts):
    p = cv2.perspectiveTransform(np.array(pts, np.float64).reshape(-1, 1, 2), np.linalg.inv(H))
    return [[round(float(x), 1), round(float(y), 1)] for x, y in p.reshape(-1, 2)]


def _on_circle(deg, radius):
    a = math.radians(deg)
    return [C + radius * math.cos(a), C + radius * math.sin(a)]


def _read(img, dial, spec, projective, debug, fallback):
    """One reading pass. ``fallback`` enables the paths for real dials whose rim outline is not the scale's
    (tick-ring refit, scale centred on the voted pivot, scale ends from tick segments)."""
    expected_sweep = float(spec.get("sweep_deg", 270.0))
    flags = []
    H = _affine_h(dial)
    rect = _warp(img, H)
    mono = rect.min(axis=2)
    light, _ = _light_face(mono)
    if fallback and projective and _vote_centre(rect) is not None:
        refit = _refit_dial(img, H, rect)
        if refit is not None:
            refit["score"] = dial.get("score", 0.5)
            dial = refit  # rectify with the tick ring from here on
            flags.append("dial_refit_to_ticks")
            H = _affine_h(dial)
            rect = _warp(img, H)
    pivot_offset, pivot_penalty = 0.0, 1.0
    # Projective maps keep lines concurrent, so the tick lines meet at the pivot's image under any tilt.
    blob = _find_pivot(mono if light else rect.max(axis=2), light)
    voted = _vote_centre(rect)
    from_ticks = voted is not None and voted[1] >= CENTRE_SUPPORT_MIN
    pivot = blob
    if fallback and from_ticks:
        pivot = voted[0]
    elif from_ticks and (blob is None or not _pivot_plausible(dial, blob, math.hypot(blob[0] - C, blob[1] - C) / R)):
        pivot = voted[0]  # hub hidden or glared: the ticks still say where the pivot is
    if pivot is None:
        flags.append("pivot_not_found")
    elif projective:
        pivot_offset = math.hypot(pivot[0] - C, pivot[1] - C) / R
        if pivot_offset <= 0.012:
            pass
        elif _pivot_plausible(dial, pivot, pivot_offset):
            H = _disc_translation(pivot) @ H  # camera tilt: perspective correction about the pivot
            rect = _warp(img, H)
        elif fallback and from_ticks and pivot_offset < 0.3:
            # the rim found is not concentric with the scale (a dial face recessed behind its bezel, seen at an
            # angle, or a case edge taken as the rim): centre the scale itself
            flags.append("scale_offset_from_rim")
            H = _shift(pivot) @ H
            rect = _warp(img, H)
        else:
            # e.g. glare on the hub: trust the rim centre instead and say so
            flags.append("pivot_inconsistent")
            pivot_penalty = 0.6
    polar = _polar(_ink(rect, light))

    scale = _find_scale(polar, expected_sweep)
    if fallback and (scale is None or scale["confidence"] < 0.85):
        alt = _scale_from_segments(rect, expected_sweep)
        if alt is not None and alt["confidence"] > (scale["confidence"] if scale else 0.0):
            scale = alt
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
    if debug:
        result["debug"] = {"rect": rect, "polar": polar, "ring_in": ring_in}
    if needle is None:
        flags.append("needle_not_found")
        return result

    rel = (needle["angle_deg"] - start) % 360.0
    if rel > sweep + (360.0 - sweep) / 2.0:
        rel -= 360.0  # just before the start of the scale
    fraction = rel / sweep
    if -0.06 <= fraction < 0.0:
        fraction = 0.0  # resting on the zero stop pin, just below the first mark: reads as the minimum
        flags.append("needle_at_zero_stop")
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


def read_gauge(img, dial: dict, spec: dict | None = None, projective: bool = True, debug: bool = False) -> dict:
    """Read a dial previously found by :func:`locate_dial`.

    ``spec`` may carry ``vmin``, ``vmax``, ``units`` and ``sweep_deg``; without ``vmin``/``vmax``
    only the needle position as a fraction of the scale is returned.

    The standard pass is tried first. Only when it is unsure are the fallbacks for real dials tried; if both
    passes give a value and they disagree, the better one is kept with its confidence halved and a flag.
    """
    spec = spec or {}
    std = _read(img, dial, spec, projective, debug, fallback=False)
    if std["confidence"] >= STANDARD_ENOUGH or not projective:
        return std
    alt = _read(img, dial, spec, projective, debug, fallback=True)
    best, other = (alt, std) if alt["confidence"] > std["confidence"] else (std, alt)
    if (best["fraction"] is not None and other["fraction"] is not None
            and abs(best["fraction"] - other["fraction"]) > 0.02 and other["confidence"] >= 0.1):
        best["confidence"] = round(best["confidence"] * 0.5, 3)
        best["flags"].append("reader_paths_disagree")
    return best
