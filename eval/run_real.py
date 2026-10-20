"""Evaluation on real photographs (openly licensed, from Wikimedia Commons; see eval/real/SOURCES.md).

Ground truth was read by eye and is approximate (``truth_tol`` in labels.json). Photos whose value a person
cannot read have ``truth: null``; for those the only correct outcome is to withhold the value.

Usage:  python eval/run_real.py [--out eval/results]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gaugeguard.agent import RulePlanner, run_inspection  # noqa: E402
from gaugeguard.agent.loop import fit_image  # noqa: E402
from gaugeguard.store import MemoryStore  # noqa: E402
from gaugeguard.vision import annotate, locate_dial, read_gauge  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TOL = 2.0  # % of full scale counted as a correct reading, as in run_eval.py


def asset_for(label):
    return {"asset_id": os.path.splitext(label["file"])[0], "name": label["file"], "vmin": label["vmin"],
            "vmax": label["vmax"], "units": label["units"], "sweep_deg": label["sweep_deg"], "warn_high": None,
            "crit_high": None, "warn_low": None, "crit_low": None, "max_step": label["vmax"] - label["vmin"]}


def err_pct(value, label):
    if value is None or label["truth"] is None:
        return None
    return round(100 * abs(value - label["truth"]) / (label["vmax"] - label["vmin"]), 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    ap.add_argument("--set", choices=["tuning", "test"], default="test",
                    help="tuning: photos used while improving the reader; test: held-out photos, labelled before the "
                         "improved reader was run on them")
    args = ap.parse_args()
    label_file, image_dir = {"tuning": ("labels.json", "images"), "test": ("test_labels.json", "test_images")}[args.set]
    labels = json.load(open(os.path.join(HERE, "real", label_file), encoding="utf-8"))

    rows, tiles = [], []
    for label in labels:
        img = fit_image(cv2.imread(os.path.join(HERE, "real", image_dir, label["file"])))
        asset = asset_for(label)

        loc = locate_dial(img)
        single = read_gauge(img, loc["dial"], asset) if loc["found"] else None
        run = run_inspection(img, asset["asset_id"], MemoryStore({asset["asset_id"]: asset}), RulePlanner(),
                             save_images=False)
        reading = run.get("reading") or {}
        recorded = run["outcome"]["status"] in ("recorded", "alert_raised", "alert_pending_approval")
        agent_value = reading.get("value") if recorded else None

        row = {"file": label["file"], "truth": label["truth"], "units": label["units"],
               "single_value": single and round(single["value"], 3), "single_conf": single and single["confidence"],
               "single_err_pct_fs": err_pct(single and single["value"], label),
               "agent_outcome": run["outcome"]["status"], "agent_value": agent_value and round(agent_value, 3),
               "agent_err_pct_fs": err_pct(agent_value, label), "agent_steps": run["steps"],
               "agent_reason": run["outcome"].get("reason") or run["outcome"].get("message")}
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False))

        shown = single or {}
        text = f"truth {label['truth']} | single {row['single_value']} | agent {agent_value and round(agent_value, 2)}"
        tile = annotate(img, shown, text, "ok" if recorded else "warn")
        h, w = tile.shape[:2]
        s = 480 / max(h, w)
        tile = cv2.resize(tile, (int(w * s), int(h * s)))
        tiles.append(cv2.copyMakeBorder(tile, 0, 480 - tile.shape[0], 0, 480 - tile.shape[1],
                                        cv2.BORDER_CONSTANT, value=(255, 255, 255)))

    readable = [r for r in rows if r["truth"] is not None]
    unreadable = [r for r in rows if r["truth"] is None]

    def stats(key_val, key_err):
        reported = [r for r in readable if r[key_val] is not None]
        correct = [r for r in reported if r[key_err] <= TOL]
        return {"readable_photos": len(readable), "reported": len(reported), "correct": len(correct),
                "wrong": len(reported) - len(correct),
                "withheld_on_unreadable": sum(1 for r in unreadable if r[key_val] is None),
                "reported_on_unreadable": sum(1 for r in unreadable if r[key_val] is not None)}

    summary = {"single_pass": stats("single_value", "single_err_pct_fs"),
               "agent": stats("agent_value", "agent_err_pct_fs"), "unreadable_photos": len(unreadable)}
    print(json.dumps(summary, indent=1))

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f"real_{args.set}_results.json"), "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "rows": rows}, f, indent=1, ensure_ascii=False)
    while len(tiles) % 4:
        tiles.append(np.full_like(tiles[0], 255))
    sheet = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)])
    cv2.imwrite(os.path.join(args.out, f"real_{args.set}_overlays.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])


if __name__ == "__main__":
    main()
