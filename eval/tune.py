"""Quick tuning harness: single-pass reading on development data only.

Synthetic: 7 per condition from development seeds (below 2000; the evaluation uses seeds from 5000).
Real: the tuning photos in eval/real/labels.json (the held-out test set lives in eval/real/test_labels.json).

    python eval/tune.py
"""
from __future__ import annotations

import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gaugeguard import config  # noqa: E402
from gaugeguard.agent.loop import fit_image  # noqa: E402
from gaugeguard.store import load_assets  # noqa: E402
from gaugeguard.synth import CONDITIONS, make_sample  # noqa: E402
from gaugeguard.vision import locate_dial, read_gauge  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def read(img, spec):
    loc = locate_dial(img)
    if not loc["found"]:
        return None, 0.0
    r = read_gauge(img, loc["dial"], spec)
    return r["value"], r["confidence"]


def summary(rows):
    n = len(rows)
    ok = [r for r in rows if r["err"] is not None and r["err"] <= 2]
    conf_ok = [r for r in ok if r["conf"] >= config.MIN_CONFIDENCE]
    conf_bad = [r for r in rows if r["conf"] >= config.MIN_CONFIDENCE and (r["err"] is None or r["err"] > 2)]
    return f"within 2%: {len(ok)}/{n}   confident & right: {len(conf_ok)}/{n}   confident & wrong: {len(conf_bad)}"


def main():
    assets = load_assets()
    syn = []
    for cond in CONDITIONS:
        for seed in range(1, 8):
            img, truth = make_sample(seed, cond)
            sp = truth.spec
            v, c = read(img, sp)
            err = None if v is None else 100 * abs(v - truth.value) / (sp["vmax"] - sp["vmin"])
            syn.append({"cond": cond, "seed": seed, "err": err, "conf": c})
    print("synthetic", summary(syn))
    bad = [f"{r['cond']}:{r['seed']}({'-' if r['err'] is None else round(r['err'], 1)})" for r in syn if r["err"] is None or r["err"] > 2]
    print("  misses:", " ".join(bad))

    labels = [l for l in json.load(open(os.path.join(HERE, "real", "labels.json"), encoding="utf-8")) if l["truth"] is not None]
    real = []
    for l in labels:
        img = fit_image(cv2.imread(os.path.join(HERE, "real", "images", l["file"])))
        v, c = read(img, l)
        err = None if v is None else 100 * abs(v - l["truth"]) / (l["vmax"] - l["vmin"])
        real.append({"err": err, "conf": c})
        print(f"  {l['file'][:26]:26} truth {l['truth']:>7} read {v if v is None else round(v, 2):>9} err {'-' if err is None else round(err, 1):>5} conf {c:.2f}")
    print("real", summary(real))


if __name__ == "__main__":
    main()
