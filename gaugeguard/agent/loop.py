"""Run one inspection: planner chooses, tools execute, everything is traced."""
from __future__ import annotations

import json
import time
import uuid

import cv2
import numpy as np

from .. import config
from ..vision import annotate
from .bedrock import BedrockPlanner, PlannerUnavailable
from .converse import ConversePlanner
from .policy import RulePlanner
from .tools import Session, call_tool, limit_status


def make_planner(name: str | None = None):
    name = name or config.PLANNER
    if name == "bedrock":
        return ConversePlanner()
    if name == "claude":
        return BedrockPlanner()
    if name == "rules":
        return RulePlanner()
    raise ValueError(f"unknown planner {name!r}")


def decode_image(data: bytes) -> np.ndarray:
    if len(data) > config.MAX_IMAGE_BYTES:
        raise ValueError("image is too large")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("could not decode image")
    return fit_image(img)


def fit_image(img: np.ndarray) -> np.ndarray:
    side = max(img.shape[:2])
    if side > config.MAX_IMAGE_SIDE:
        f = config.MAX_IMAGE_SIDE / side
        img = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    return img


def _compact(result: dict) -> dict:
    """Trace-friendly copy of a tool result (no large nested blobs)."""
    return json.loads(json.dumps({k: v for k, v in result.items() if k != "raw"}, default=str))


def _finalise(s: Session, stopped_early: str | None) -> dict:
    if s.outcome:
        return dict(s.outcome)
    if s.recorded:
        status = limit_status(s.asset, s.recorded["value"])
        if status != "normal" and s.alert is None:
            # An out-of-limit recorded value always produces an alert, whatever the planner did.
            call_tool(s, "raise_alert", {"message": f"{s.asset['name']} recorded {s.recorded['value']:.3g} "
                                         f"{s.asset['units']} ({status})."})
        if s.alert:
            state = "alert_pending_approval" if s.alert["state"] == "pending_approval" else "alert_raised"
            return {"status": state, "severity": s.alert["severity"]}
        return {"status": "recorded"}
    return {"status": "escalated", "reason": stopped_early or "The planner stopped without reaching a decision."}


def run_inspection(image: np.ndarray, asset_id: str, store, planner=None, run_id: str | None = None,
                   max_steps: int | None = None, save_images: bool = True) -> dict:
    asset = store.get_asset(asset_id)
    if asset is None:
        raise KeyError(f"unknown asset {asset_id!r}")
    run_id = run_id or uuid.uuid4().hex[:12]
    planner = planner or make_planner()
    s = Session(fit_image(image), asset, store, run_id)
    trace, t_start, last, stopped, planners = [], time.perf_counter(), None, None, [planner.name]

    for step_no in range(1, (max_steps or config.MAX_STEPS) + 1):
        try:
            action = planner.step(s, last)
        except PlannerUnavailable as e:
            # The rule planner reads only session state, so it can continue from here.
            trace.append({"step": step_no, "tool": "planner_fallback", "args": {}, "reason": str(e)[:300],
                          "result": {"from": planner.name, "to": "rules"}, "ms": 0})
            planner, last = RulePlanner(), None
            planners.append("rules")
            action = planner.step(s, None)
        if action is None:
            break
        t0 = time.perf_counter()
        result = call_tool(s, action.tool, action.args)
        trace.append({"step": step_no, "tool": action.tool, "args": action.args, "reason": action.reason,
                      "result": _compact(result), "ms": round((time.perf_counter() - t0) * 1000, 1)})
        last = (action, result)
        if s.done:
            break
    else:
        stopped = "Step budget exhausted before a decision was reached."

    outcome = _finalise(s, stopped)
    chosen = s.reading(s.recorded["reading_id"]) if s.recorded else s.best_reading()
    view = {k: v for k, v in chosen.items() if k != "raw"} if chosen else None
    label_status = {"recorded": "ok", "alert_raised": "alert", "alert_pending_approval": "alert"}.get(
        outcome["status"], "warn")
    if s.recorded:
        label = f"{asset['asset_id']}  {s.recorded['value']:.3g} {asset['units']}  conf {s.recorded['confidence']:.2f}"
    else:
        label = f"{asset['asset_id']}  no reading recorded: {outcome['status'].replace('_', ' ')}"
    annotated = annotate(s.images[chosen["image"]] if chosen else s.images["original"],
                         chosen["raw"] if chosen else None, label, label_status)

    run = {"run_id": run_id, "asset_id": asset_id, "ts": time.time(), "planner": "+".join(planners),
           "outcome": outcome, "recorded": s.recorded, "alert": s.alert, "reading": view,
           "quality": s.quality.get("original"), "trace": trace, "steps": len(trace),
           "duration_ms": round((time.perf_counter() - t_start) * 1000, 1), "opencv": cv2.__version__,
           "usage": getattr(planner, "usage", None), "images": {}}
    if save_images:
        for name, im in (("original.jpg", s.images["original"]), ("annotated.jpg", annotated)):
            ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 88])
            run["images"][name.split(".")[0]] = store.save_image(run_id, name, buf.tobytes())
    store.put_run(run)
    return run


def decide_alert(store, run_id: str, approve: bool, operator: str, note: str = "") -> dict:
    """Human decision on a critical alert that was held for approval."""
    run = store.get_run(run_id)
    if run is None:
        raise KeyError(run_id)
    alert = run.get("alert")
    if not alert or alert.get("state") != "pending_approval":
        raise ValueError("this run has no alert awaiting approval")
    alert["decision"] = {"approved": bool(approve), "operator": operator[:80], "note": note[:300], "ts": time.time()}
    if approve:
        alert["state"] = "dispatched"
        alert["delivery"] = store.send_alert(alert)
        run["outcome"] = {"status": "alert_raised", "severity": alert["severity"], "approved_by": operator[:80]}
    else:
        alert["state"] = "rejected"
        run["outcome"] = {"status": "alert_rejected", "severity": alert["severity"], "rejected_by": operator[:80]}
    store.put_run(run)
    return run
