"""Compare model planners (Bedrock Converse API) with the rule planner on the same synthetic gauges.

Makes live Bedrock calls (about 15k tokens per run). Usage:
  python eval/compare_planners.py zai.glm-5,amazon.nova-pro-v1:0[@region],...
Samples: one per condition plus out-of-limit values, seeds from 8000 (unseen during development).
"""
import collections
import json
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gaugeguard.agent import RulePlanner, limit_status, run_inspection  # noqa: E402
from gaugeguard.agent.converse import ConversePlanner  # noqa: E402
from gaugeguard.store import MemoryStore, load_assets  # noqa: E402
from gaugeguard.synth import make_sample  # noqa: E402

models = sys.argv[1].split(",")
conds = ["clean", "low_light", "glare", "noise", "combined", "motion", "occlusion", "tilt", "blur", "jpeg"]
assets = load_assets()
cases = []
seed = 8000
while len(cases) < 14:  # all conditions once, plus out-of-limit values to test verification and alerting
    img, truth = make_sample(seed, conds[len(cases) % len(conds)])
    sp = truth.spec
    a = next(x for x in assets.values() if (x["vmin"], x["vmax"], x["units"]) == (sp["vmin"], sp["vmax"], sp["units"]))
    status = limit_status(a, truth.value)
    if len(cases) < len(conds) or status != "normal":
        cases.append((seed, img, truth, a, status))
    seed += 1

score = collections.defaultdict(collections.Counter)
for seed, img, truth, a, status in cases:
    fs = a["vmax"] - a["vmin"]
    for name in ["rules"] + models:
        mid, _, reg = name.partition("@")
        planner = RulePlanner() if name == "rules" else ConversePlanner(model=mid, region=reg or "ap-southeast-2")
        t = time.time()
        run = run_inspection(img, a["asset_id"], MemoryStore(assets), planner, save_images=False)
        rec = run.get("recorded")
        err = None if not rec else 100 * abs(rec["value"] - truth.value) / fs
        sc = score[name]
        sc["runs"] += 1
        sc["fallback"] += "+rules" in run["planner"]
        sc["correct"] += err is not None and err <= 2
        sc["wrong"] += err is not None and err > 2
        sc["alerted_when_needed"] += status != "normal" and run.get("alert") is not None
        sc["needed_alert"] += status != "normal"
        sc["steps"] += run["steps"]
        sc["seconds"] += time.time() - t
        sc["tokens"] += sum(getattr(planner, "usage", {}).values())
        sc["empty_notes"] += sum(1 for s in run["trace"] if s["tool"] != "planner_fallback" and not (s.get("reason") or "").strip())
        print(f"{seed} {status:9} {name[:30]:30} "
              f"{run['outcome']['status']:22} steps={run['steps']:2} err={None if err is None else round(err, 2)} {run['planner']}",
              flush=True)

print("\nSUMMARY")
for name, sc in score.items():
    n = sc["runs"]
    print(f"{name[:34]:34} correct {sc['correct']}/{n}  wrong {sc['wrong']}  alerts {sc['alerted_when_needed']}/{sc['needed_alert']}"
          f"  fallback {sc['fallback']}  mean steps {sc['steps']/n:.1f}  mean {sc['seconds']/n:.1f}s"
          f"  tokens/run {sc['tokens']/n:.0f}  empty notes {sc['empty_notes']}")
