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
