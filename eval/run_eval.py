"""Reproducible evaluation on synthetic gauges with exact ground truth.

Compares three ways of using the same OpenCV 5 reader:
  single_pass  locate once, read once, always report the value
  gated        the same, but withhold the value when confidence is below the minimum
  agent        the full perception-decision-action loop (rule planner, so it runs offline)

Usage:  python eval/run_eval.py [--n 60] [--out eval/results]
Seeds below 2000 were used while developing the reader and the planner; the evaluation uses unseen seeds from 5000.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gaugeguard import config  # noqa: E402
from gaugeguard.agent import RulePlanner, limit_status, run_inspection  # noqa: E402
from gaugeguard.store import MemoryStore, load_assets  # noqa: E402
from gaugeguard.synth import CONDITIONS, make_sample  # noqa: E402
from gaugeguard.vision import locate_dial, read_gauge  # noqa: E402

TOL = 2.0  # % of full scale counted as a correct reading
FIRST_SEED = 5000


def asset_for(truth, assets):
    sp = truth.spec
    return next(a for a in assets.values() if (a["vmin"], a["vmax"], a["units"]) == (sp["vmin"], sp["vmax"], sp["units"]))


def single_pass(img, asset):
    loc = locate_dial(img)
    if not loc["found"]:
        return None, 0.0
    r = read_gauge(img, loc["dial"], asset)
    return r["value"], r["confidence"]


def summarise(rows, key):
    n = len(rows)
    reported = [r for r in rows if r[key]["value"] is not None]
    errs = np.array([r[key]["err"] for r in reported]) if reported else np.array([])
    correct = int((errs <= TOL).sum())
    truth_out = [r for r in rows if r["truth_status"] != "normal"]
    alerts = [r for r in rows if r[key].get("alert")]
    return {
        "n": n,
        "coverage_pct": round(100 * len(reported) / n, 1),
        "correct_pct": round(100 * correct / n, 1),
        "silent_error_pct": round(100 * (len(reported) - correct) / n, 1),
        "mae_pct_fs": round(float(errs.mean()), 2) if len(errs) else None,
        "median_err_pct_fs": round(float(np.median(errs)), 2) if len(errs) else None,
        "alert_recall_pct": (round(100 * sum(1 for r in truth_out if r[key].get("alert")) / len(truth_out), 1)
                             if truth_out else None),
        "false_alerts": sum(1 for r in alerts if r["truth_status"] == "normal"),
        "out_of_limit_samples": len(truth_out),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60, help="samples per condition")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    assets = load_assets()
    rows, traces, failures = [], {}, []
    t_all = time.time()

    for cond in CONDITIONS:
        for seed in range(FIRST_SEED, FIRST_SEED + args.n):
            img, truth = make_sample(seed, cond)
            asset = asset_for(truth, assets)
            fs = asset["vmax"] - asset["vmin"]
            truth_status = limit_status(asset, truth.value)

            def score(value):
                return None if value is None else abs(value - truth.value) / fs * 100

            v, c = single_pass(img, asset)
            sp = {"value": v, "err": score(v), "alert": v is not None and limit_status(asset, v) != "normal"}
            gv = v if c >= config.MIN_CONFIDENCE else None
            gated = {"value": gv, "err": score(gv), "alert": gv is not None and limit_status(asset, gv) != "normal"}

            t0 = time.perf_counter()
            run = run_inspection(img, asset["asset_id"], MemoryStore(assets), RulePlanner(), save_images=False)
            ms = (time.perf_counter() - t0) * 1000
            av = run["recorded"]["value"] if run["recorded"] else None
            agent = {"value": av, "err": score(av), "alert": run["alert"] is not None,
                     "status": run["outcome"]["status"], "steps": run["steps"], "ms": ms}
            rows.append({"condition": cond, "seed": seed, "truth": truth.value, "truth_status": truth_status,
                         "single_pass": sp, "gated": gated, "agent": agent})
            if cond not in traces and run["steps"] > 5:
                traces[cond] = {"seed": seed, "truth": truth.value, "outcome": run["outcome"],
                                "recorded": run["recorded"], "trace": run["trace"]}
            if agent["err"] is not None and agent["err"] > TOL and len(failures) < 8:
                failures.append((cond, seed, truth.value, av, img))
        print(f"{cond:10s} done", flush=True)

    summary = {"settings": {"n_per_condition": args.n, "first_seed": FIRST_SEED, "tolerance_pct_fs": TOL,
                            "min_confidence": config.MIN_CONFIDENCE, "opencv": cv2.__version__,
                            "seconds": round(time.time() - t_all, 1)},
               "overall": {}, "by_condition": {}, "agent_outcomes": {}, "agent_cost": {}}
    for key in ("single_pass", "gated", "agent"):
        summary["overall"][key] = summarise(rows, key)
    for cond in CONDITIONS:
        sub = [r for r in rows if r["condition"] == cond]
        summary["by_condition"][cond] = {k: summarise(sub, k) for k in ("single_pass", "gated", "agent")}
    for r in rows:
        st = r["agent"]["status"]
        summary["agent_outcomes"][st] = summary["agent_outcomes"].get(st, 0) + 1
    summary["agent_cost"] = {"mean_steps": round(float(np.mean([r["agent"]["steps"] for r in rows])), 2),
                             "mean_ms": round(float(np.mean([r["agent"]["ms"] for r in rows])), 1),
                             "p95_ms": round(float(np.percentile([r["agent"]["ms"] for r in rows], 95)), 1)}

    with open(os.path.join(args.out, "results.json"), "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "rows": rows}, f, indent=1)
    with open(os.path.join(args.out, "example_traces.json"), "w", encoding="utf-8") as f:
        json.dump(traces, f, indent=1)
    if failures:
        tiles = []
        for cond, seed, tv, av, img in failures:
            tile = cv2.resize(img, (320, 320))
            cv2.rectangle(tile, (0, 0), (320, 26), (20, 20, 20), -1)
            cv2.putText(tile, f"{cond} #{seed}: read {av:.3g}, true {tv:.3g}", (6, 18), (255, 255, 255),
                        cv2.FontFace("sans"), 15, 500)
            tiles.append(tile)
        while len(tiles) % 4:
            tiles.append(np.zeros_like(tiles[0]))
        grid = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)])
        cv2.imwrite(os.path.join(args.out, "agent_failures.jpg"), grid)

    def line(name, m):
        return (f"| {name} | {m['coverage_pct']} | {m['correct_pct']} | {m['silent_error_pct']} | "
                f"{m['mae_pct_fs']} | {m['alert_recall_pct']} | {m['false_alerts']} |")

    head = ("| | Reported % | Correct % | Wrong value reported % | MAE of reported (% FS) | Alert recall % | "
            "False alerts |\n|---|---|---|---|---|---|---|")
    md = [f"# Evaluation results\n\nOpenCV {cv2.__version__}, {args.n} synthetic images per condition "
          f"({len(rows)} total), seeds from {FIRST_SEED}. A reading is correct when it is within {TOL}% of full "
          "scale. All percentages are of all images in the group.\n", "## Overall\n", head]
    md += [line(k, summary["overall"][k]) for k in ("single_pass", "gated", "agent")]
    md.append("\n## By condition\n")
    for cond in CONDITIONS:
        md += [f"### {cond}\n", head] + [line(k, summary["by_condition"][cond][k])
                                           for k in ("single_pass", "gated", "agent")] + [""]
    md.append("## Agent outcomes\n")
    md += [f"- {k}: {v}" for k, v in sorted(summary["agent_outcomes"].items())]
    c = summary["agent_cost"]
    md.append(f"\nAgent cost on this machine: mean {c['mean_steps']} tool calls and {c['mean_ms']} ms per image "
              f"(p95 {c['p95_ms']} ms).\n")
    with open(os.path.join(args.out, "results.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
