"""Synthetic analog-gauge renderer with exact ground truth.

Every image used in the evaluation is generated here, so the project ships no
third-party media and every reading has a known true value.

Angle convention used across the project: degrees, measured clockwise on
screen from the +x axis (image y points down), so a point on the dial is
``centre + r * (cos(theta), sin(theta))``.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import cv2
import numpy as np

SS = 2  # supersampling factor for anti-aliased rendering


@dataclass
class GaugeSpec:
    vmin: float = 0.0
    vmax: float = 10.0
    units: str = "bar"
    start_deg: float = 135.0  # angle of vmin
    sweep_deg: float = 270.0  # clockwise sweep from vmin to vmax
    major: int = 10  # number of major intervals
    minor: int = 5  # subdivisions per major interval
    needle_bgr: tuple = (20, 20, 20)
    face_bgr: tuple = (238, 240, 240)
    zone_from: float | None = None  # red zone from this value to vmax

    def angle_of(self, value: float) -> float:
        frac = (value - self.vmin) / (self.vmax - self.vmin)
        return self.start_deg + frac * self.sweep_deg


@dataclass
class Truth:
    value: float
    needle_deg: float
    center: tuple
    radius: float
    spec: dict
    view: dict = field(default_factory=dict)
    degradations: list = field(default_factory=list)


def _pt(c, r, deg):
    a = math.radians(deg)
    return (int(round(c[0] + r * math.cos(a))), int(round(c[1] + r * math.sin(a))))


def _background(size, rng):
    base = rng.integers(70, 190, size=3).astype(np.float32)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) / size
    gx, gy = rng.uniform(-40, 40, size=2)
    img = base[None, None, :] + (gx * xx + gy * yy)[..., None]
    # a few pipe / panel shapes so the dial is not the only structure in view
    img = np.clip(img, 0, 255).astype(np.uint8)
    for _ in range(int(rng.integers(2, 5))):
        col = tuple(int(v) for v in rng.integers(40, 210, size=3))
        if rng.random() < 0.5:
            x = int(rng.integers(0, size))
            cv2.rectangle(img, (x, 0), (x + int(rng.integers(12, 40)), size), col, -1)
        else:
            y = int(rng.integers(0, size))
            cv2.rectangle(img, (0, y), (size, y + int(rng.integers(12, 40))), col, -1)
    noise = rng.normal(0, 4, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def render_gauge(value, spec: GaugeSpec | None = None, size=640, rng=None, radius_frac=None):
    """Render a front-on gauge. Returns (bgr image, Truth)."""
    spec = spec or GaugeSpec()
    rng = rng or np.random.default_rng(0)
    S = size * SS
    img = _background(S, rng)
    rb = (radius_frac or rng.uniform(0.30, 0.42)) * S  # bezel radius
    margin = rb + 0.03 * S
    c = (float(rng.uniform(margin, S - margin)), float(rng.uniform(margin, S - margin)))
    ci = (int(round(c[0])), int(round(c[1])))
    rf = 0.88 * rb  # face radius

    cv2.circle(img, ci, int(rb), (38, 38, 42), -1, cv2.LINE_AA)
    cv2.circle(img, ci, int(rb * 0.97), (70, 70, 76), max(1, int(rb * 0.02)), cv2.LINE_AA)
    cv2.circle(img, ci, int(rf), spec.face_bgr, -1, cv2.LINE_AA)

    if spec.zone_from is not None:
        a0, a1 = spec.angle_of(spec.zone_from), spec.angle_of(spec.vmax)
        cv2.ellipse(img, ci, (int(rf * 0.955), int(rf * 0.955)), 0, a0, a1, (40, 40, 215),
                    max(2, int(rf * 0.035)), cv2.LINE_AA)

    n_ticks = spec.major * spec.minor
    for i in range(n_ticks + 1):
        deg = spec.start_deg + spec.sweep_deg * i / n_ticks
        is_major = i % spec.minor == 0
        r0 = rf * (0.78 if is_major else 0.85)
        cv2.line(img, _pt(c, r0, deg), _pt(c, rf * 0.93, deg), (25, 25, 25),
                 max(1, int(rf * (0.016 if is_major else 0.007))), cv2.LINE_AA)
    for i in range(spec.major + 1):
        deg = spec.start_deg + spec.sweep_deg * i / spec.major
        val = spec.vmin + (spec.vmax - spec.vmin) * i / spec.major
        text = f"{val:g}"
        scale = rf / 260.0
        thick = max(1, int(rf / 130))
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        p = _pt(c, rf * 0.64, deg)
        cv2.putText(img, text, (p[0] - tw // 2, p[1] + th // 2), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (30, 30, 30), thick, cv2.LINE_AA)
    (tw, th), _ = cv2.getTextSize(spec.units, cv2.FONT_HERSHEY_SIMPLEX, rf / 300.0, max(1, int(rf / 150)))
    cv2.putText(img, spec.units, (ci[0] - tw // 2, int(c[1] + rf * 0.42)), cv2.FONT_HERSHEY_SIMPLEX,
                rf / 300.0, (60, 60, 60), max(1, int(rf / 150)), cv2.LINE_AA)

    needle_deg = spec.angle_of(value)
    tip = _pt(c, rf * 0.76, needle_deg)
    tail = _pt(c, rf * 0.17, needle_deg + 180)
    w = rf * 0.028
    left, right = _pt(c, w, needle_deg + 90), _pt(c, w, needle_deg - 90)
    cv2.fillConvexPoly(img, np.array([tip, left, tail, right], np.int32), spec.needle_bgr, cv2.LINE_AA)
    cv2.circle(img, ci, int(rf * 0.075), (30, 30, 30), -1, cv2.LINE_AA)

    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    truth = Truth(value=float(value), needle_deg=float(needle_deg), center=(c[0] / SS, c[1] / SS),
                  radius=rf / SS, spec=asdict(spec))
    return img, truth


# ---------------------------------------------------------------- viewpoint

def apply_view(img, truth: Truth, roll_deg=0.0, tilt_deg=0.0, tilt_dir_deg=0.0, rng=None):
    """Re-photograph the scene with camera roll and an out-of-plane tilt (true homography)."""
    h, w = img.shape[:2]
    cx, cy = truth.center
    f = 1.6 * w  # focal length in pixels; viewing distance equals f so scale is preserved
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], np.float64)
    t, d, r = (math.radians(a) for a in (tilt_deg, tilt_dir_deg, roll_deg))
    axis = np.array([math.cos(d), math.sin(d), 0.0])
    Kx = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    R_tilt = np.eye(3) + math.sin(t) * Kx + (1 - math.cos(t)) * (Kx @ Kx)
    R_roll = np.array([[math.cos(r), -math.sin(r), 0], [math.sin(r), math.cos(r), 0], [0, 0, 1]])
    R = R_roll @ R_tilt
    # plane point X=(x-cx, y-cy, 0) -> camera R X + (0,0,f)
    A = np.column_stack([R[:, 0], R[:, 1], np.array([0, 0, f])])
    Kinv_plane = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], np.float64)
    H = K @ A @ Kinv_plane
    border = tuple(int(v) for v in img[2, 2])
    out = cv2.warpPerspective(img, H, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                              borderValue=border)
    truth.view = {"roll_deg": roll_deg, "tilt_deg": tilt_deg, "tilt_dir_deg": tilt_dir_deg}
    return out, truth


# ------------------------------------------------------------- degradations

def _blur(img, sev, rng):
    return cv2.GaussianBlur(img, (0, 0), 1.5 + 4.0 * sev)


def _motion(img, sev, rng):
    k = int(7 + 22 * sev) | 1
    kernel = np.zeros((k, k), np.float32)
    ang = rng.uniform(0, 180)
    c = k // 2
    dx, dy = math.cos(math.radians(ang)) * c, math.sin(math.radians(ang)) * c
    cv2.line(kernel, (int(round(c - dx)), int(round(c - dy))), (int(round(c + dx)), int(round(c + dy))), 1.0, 1)
    return cv2.filter2D(img, -1, kernel / kernel.sum())


def _low_light(img, sev, rng):
    gain = 0.30 - 0.20 * sev
    dark = img.astype(np.float32) * gain
    dark += rng.normal(0, 3 + 6 * sev, img.shape)
    return np.clip(dark, 0, 255).astype(np.uint8)


def _noise(img, sev, rng):
    return np.clip(img.astype(np.float32) + rng.normal(0, 10 + 30 * sev, img.shape), 0, 255).astype(np.uint8)


def _glare(img, sev, rng, truth=None):
    h, w = img.shape[:2]
    cx, cy, r = (truth.center[0], truth.center[1], truth.radius) if truth else (w / 2, h / 2, w / 4)
    ang, dist = rng.uniform(0, 2 * math.pi), rng.uniform(0.2, 0.75) * r
    gx, gy = cx + dist * math.cos(ang), cy + dist * math.sin(ang)
    mask = np.zeros((h, w), np.float32)
    axes = (int(r * (0.25 + 0.35 * sev)), int(r * (0.15 + 0.25 * sev)))
    cv2.ellipse(mask, (int(gx), int(gy)), axes, rng.uniform(0, 180), 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), r * 0.08)[..., None]
    return np.clip(img.astype(np.float32) * (1 - mask) + 255.0 * mask, 0, 255).astype(np.uint8)


def _occlusion(img, sev, rng, truth=None):
    h, w = img.shape[:2]
    cx, cy, r = (truth.center[0], truth.center[1], truth.radius) if truth else (w / 2, h / 2, w / 4)
    out = img.copy()
    ang = rng.uniform(0, 360)
    off = r * rng.uniform(0.35, 0.8)
    p = np.array([cx + off * math.cos(math.radians(ang)), cy + off * math.sin(math.radians(ang))])
    d = np.array([-math.sin(math.radians(ang)), math.cos(math.radians(ang))])
    half = r * (0.06 + 0.10 * sev)
    n = np.array([math.cos(math.radians(ang)), math.sin(math.radians(ang))])
    quad = np.array([p - d * 3 * r - n * half, p + d * 3 * r - n * half,
                     p + d * 3 * r + n * half, p - d * 3 * r + n * half], np.int32)
    cv2.fillConvexPoly(out, quad, tuple(int(v) for v in rng.integers(30, 120, size=3)), cv2.LINE_AA)
    return out


def _jpeg(img, sev, rng):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(28 - 20 * sev)])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


DEGRADATIONS = {"blur": _blur, "motion": _motion, "low_light": _low_light, "noise": _noise,
                "glare": _glare, "occlusion": _occlusion, "jpeg": _jpeg}


def degrade(img, truth: Truth, kind: str, severity: float, rng):
    fn = DEGRADATIONS[kind]
    out = fn(img, severity, rng, truth) if kind in ("glare", "occlusion") else fn(img, severity, rng)
    truth.degradations.append({"kind": kind, "severity": round(float(severity), 3)})
    return out, truth


# ------------------------------------------------------------------ dataset

CONDITIONS = ["clean", "roll", "tilt", "blur", "motion", "low_light", "noise", "glare", "occlusion",
              "jpeg", "combined"]

SPECS = [
    GaugeSpec(0, 10, "bar"),
    GaugeSpec(0, 160, "psi", major=8, minor=5, needle_bgr=(30, 30, 200), zone_from=120),
    GaugeSpec(0, 100, "C", major=10, minor=2, face_bgr=(225, 235, 240)),
    GaugeSpec(0, 6, "bar", major=6, minor=10, zone_from=5),
    GaugeSpec(-1, 3, "bar", major=8, minor=5, needle_bgr=(25, 25, 170)),
]


def make_sample(seed: int, condition: str = "clean", size=640):
    """Deterministic sample for a seed and condition. Returns (image, Truth)."""
    rng = np.random.default_rng(seed)
    spec = SPECS[int(rng.integers(0, len(SPECS)))]
    value = float(rng.uniform(spec.vmin, spec.vmax))
    img, truth = render_gauge(value, spec, size=size, rng=rng)
    if condition == "roll":
        img, truth = apply_view(img, truth, roll_deg=float(rng.uniform(-35, 35)))
    elif condition == "tilt":
        img, truth = apply_view(img, truth, roll_deg=float(rng.uniform(-10, 10)),
                                tilt_deg=float(rng.uniform(15, 40)), tilt_dir_deg=float(rng.uniform(0, 360)))
    elif condition == "combined":
        img, truth = apply_view(img, truth, roll_deg=float(rng.uniform(-20, 20)),
                                tilt_deg=float(rng.uniform(0, 25)), tilt_dir_deg=float(rng.uniform(0, 360)))
        kinds = rng.choice(["blur", "low_light", "noise", "glare", "jpeg"], size=2, replace=False)
        for k in kinds:
            img, truth = degrade(img, truth, str(k), float(rng.uniform(0.1, 0.7)), rng)
    elif condition != "clean":
        img, truth = degrade(img, truth, condition, float(rng.uniform(0.2, 1.0)), rng)
    return img, truth
