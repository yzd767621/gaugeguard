import cv2
import numpy as np
import pytest

from gaugeguard.synth import apply_view, degrade, make_sample
from gaugeguard.vision import ENHANCE_OPS, annotate, assess_quality, enhance, locate_dial, read_gauge


def _read(img, spec):
    loc = locate_dial(img)
    assert loc["found"]
    return read_gauge(img, loc["dial"], spec)


def test_runs_on_opencv_5():
    assert int(cv2.__version__.split(".")[0]) >= 5


@pytest.mark.parametrize("condition", ["clean", "roll", "tilt"])
def test_reads_within_half_percent_of_full_scale(condition):
    for seed in range(8):
        img, truth = make_sample(seed, condition)
        r = _read(img, truth.spec)
        full_scale = truth.spec["vmax"] - truth.spec["vmin"]
        assert abs(r["value"] - truth.value) / full_scale < 0.005, (condition, seed)
        assert r["confidence"] > 0.8


def test_perspective_correction_matters_on_a_tilted_dial(gauge):
    img, truth = gauge("PG-101", 3.2)
    img, truth = apply_view(img, truth, tilt_deg=38, tilt_dir_deg=20)
    loc = locate_dial(img)
    with_h = read_gauge(img, loc["dial"], truth.spec, projective=True)
    without = read_gauge(img, loc["dial"], truth.spec, projective=False)
    assert abs(with_h["value"] - 3.2) < 0.05
    assert abs(with_h["value"] - 3.2) < abs(without["value"] - 3.2)


def test_quality_flags_the_right_defect(gauge):
    img, truth = gauge("PG-101", 5.0)
    rng = np.random.default_rng(1)
    assert assess_quality(img)["issues"] == []
    for kind, issue in (("blur", "blurry"), ("low_light", "dark"), ("noise", "noisy")):
        bad, _ = degrade(img, truth, kind, 0.9, rng)
        assert issue in assess_quality(bad)["issues"], kind


def test_every_enhancement_keeps_shape_and_type(gauge):
    img, _ = gauge("PG-101", 5.0)
    for op in ENHANCE_OPS:
        out = enhance(img, op)
        assert out.shape == img.shape and out.dtype == np.uint8
    with pytest.raises(ValueError):
        enhance(img, "nope")


def test_blank_image_has_no_dial():
    blank = np.full((480, 640, 3), 127, np.uint8)
    assert not locate_dial(blank)["found"]
    assert not locate_dial(blank, "hough")["found"]


def test_low_confidence_when_the_needle_is_hidden(gauge):
    img, truth = gauge("PG-101", 5.0)
    c = tuple(int(v) for v in truth.center)
    cv2.circle(img, c, int(truth.radius * 0.7), (240, 240, 240), -1)  # paint over needle and digits
    assert _read(img, truth.spec)["confidence"] < 0.6


def test_annotate_draws_on_a_copy(gauge):
    img, truth = gauge("PG-101", 5.0)
    out = annotate(img, _read(img, truth.spec), "PG-101 5.00 bar", "ok")
    assert out.shape == img.shape and not np.array_equal(out, img)


def test_tick_marks_vote_for_the_pivot(gauge):
    from gaugeguard.vision import reader as rd
    img, truth = gauge("PG-101", 4.0)
    rect = rd._warp(img, rd._affine_h(locate_dial(img)["dial"]))
    (cx, cy), support = rd._vote_centre(rect)
    assert support > 0.3 and np.hypot(cx - rd.C, cy - rd.C) < 2.5  # ~1% of R: a fallback, the hub blob is finer


@pytest.mark.parametrize("name, truth, full_scale", [("soderstroms_0-15kgcm2.jpg", 0.0, 15),
                                                     ("astralpool_0-3bar.jpg", 0.0, 3)])
def test_real_dial_with_a_long_counterweight_reads_the_pointer_end(name, truth, full_scale):
    import json
    import os
    from gaugeguard.agent.loop import fit_image
    here = os.path.join(os.path.dirname(__file__), "..", "eval", "real")
    label = next(l for l in json.load(open(os.path.join(here, "labels.json"), encoding="utf-8")) if l["file"] == name)
    r = _read(fit_image(cv2.imread(os.path.join(here, "images", name))), label)
    assert abs(r["value"] - truth) / full_scale < 0.02
