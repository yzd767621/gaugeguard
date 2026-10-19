"""The agent's tools and the guardrails around them.

Every rule that matters for safety lives here, in code, and applies to whichever
planner is driving (the rule planner or the Bedrock model). A planner can ask for
anything; a tool refuses when the evidence does not support it.
"""
from __future__ import annotations

import time

import numpy as np

from .. import config
from ..vision import ENHANCE_OPS, assess_quality, enhance, locate_dial, read_gauge

MAX_ENHANCED_VERSIONS = 5
FATAL_FLAGS = ("needle_outside_scale",)


class ToolError(Exception):
    """A refused or invalid tool call. The message is returned to the planner."""


def limit_status(asset: dict, value: float) -> str:
    for key, test in (("crit_high", value.__ge__), ("crit_low", value.__le__),
                      ("warn_high", value.__ge__), ("warn_low", value.__le__)):
        if asset.get(key) is not None and test(asset[key]):
            return key
    return "normal"


class Session:
    """All state for one inspection run."""

    def __init__(self, image: np.ndarray, asset: dict, store, run_id: str):
        self.images = {"original": image}
        self.current = "original"
        self.asset, self.store, self.run_id = asset, store, run_id
        self.history = store.recent_readings(asset["asset_id"], 5)
        self.context_loaded = False
        self.quality: dict = {}
        self.locates: dict = {}  # image id -> latest locate result
        self.readings: list = []
        self.recorded = None
        self.alert = None
        self.outcome = None  # set by a terminal tool

    # -- helpers used by tools and planners
    @property
    def full_scale(self):
        return self.asset["vmax"] - self.asset["vmin"]

    def reading(self, reading_id):
        return next((r for r in self.readings if r["reading_id"] == reading_id), None)

    def latest_reading_on(self, image_id, method):
        return next((r for r in reversed(self.readings) if r["image"] == image_id and r["method"] == method), None)

    def best_reading(self):
        usable = [r for r in self.readings if r["value"] is not None]
        return max(usable, key=lambda r: r["confidence"]) if usable else None

    def acceptable(self, r):
        return (r is not None and r["value"] is not None and r["confidence"] >= config.MIN_CONFIDENCE
                and not any(f in r["flags"] for f in FATAL_FLAGS))

    def needs_verification(self, r) -> str | None:
        """Out-of-limit values and implausible jumps must be confirmed by a second, independent reading."""
        status = limit_status(self.asset, r["value"])
        if status != "normal":
            return f"value is in the {status} band"
        if self.history and self.asset.get("max_step") is not None:
            jump = abs(r["value"] - self.history[-1]["value"])
            if jump > self.asset["max_step"]:
                return f"value moved {jump:.3g} {self.asset['units']} since the last recorded reading"
        return None

    def agreeing_reading(self, r):
        tol = config.AGREE_FRACTION * self.full_scale
        for other in self.readings:
            independent = (other["image"], other["method"]) != (r["image"], r["method"])
            if (other is not r and independent and other["value"] is not None
                    and other["confidence"] >= 0.67 * config.MIN_CONFIDENCE
                    and abs(other["value"] - r["value"]) <= tol):
                return other
        return None

    @property
    def done(self):
        if self.outcome:
            return True
        if self.recorded:
            return self.alert is not None or limit_status(self.asset, self.recorded["value"]) == "normal"
        return False


# ------------------------------------------------------------------- tools

def get_asset_context(s: Session):
    s.context_loaded = True
    return {"asset": s.asset, "recent_readings": [{"value": h["value"], "ts": h["ts"]} for h in s.history],
            "policy": {"min_confidence": config.MIN_CONFIDENCE, "agreement_tolerance_fraction": config.AGREE_FRACTION,
                       "critical_alerts_need_human_approval": True}}


def tool_assess_quality(s: Session):
    loc = s.locates.get(s.current)
    q = assess_quality(s.images[s.current], loc["dial"] if loc and loc["found"] else None)
    s.quality[s.current] = q
    return {"image": s.current, **q}


def enhance_image(s: Session, op: str):
    if op not in ENHANCE_OPS:
        raise ToolError(f"unknown op {op!r}; choose one of {sorted(ENHANCE_OPS)}")
    if len(s.images) > MAX_ENHANCED_VERSIONS:
        raise ToolError("enhancement budget used up; decide with the evidence you have")
    new_id = f"v{len(s.images)}:{op}"
    s.images[new_id] = enhance(s.images[s.current], op)
    source, s.current = s.current, new_id
    return {"image": new_id, "derived_from": source, "note": "now the current image; locate and read it again"}


def tool_locate_dial(s: Session, method: str = "contour"):
    if method not in ("contour", "hough"):
        raise ToolError("method must be 'contour' or 'hough'")
    loc = locate_dial(s.images[s.current], method)
    s.locates[s.current] = loc
    out = {"image": s.current, "method": method, "found": loc["found"], "candidates": len(loc["candidates"])}
    if loc["found"]:
        d = loc["dial"]
        out.update(score=d["score"], centre=[round(v) for v in d["center"]],
                   axis_ratio=round(min(d["axes"]) / max(d["axes"]), 3))
    return out


def tool_read_gauge(s: Session, candidate: int = 0):
    loc = s.locates.get(s.current)
    if not loc or not loc["found"]:
        raise ToolError("no dial located on the current image; call locate_dial first")
    if not 0 <= candidate < len(loc["candidates"]):
        raise ToolError(f"candidate must be between 0 and {len(loc['candidates']) - 1}")
    raw = read_gauge(s.images[s.current], loc["candidates"][candidate], s.asset)
    r = {"reading_id": f"r{len(s.readings) + 1}", "image": s.current, "method": f"{loc['method']}#{candidate}",
         "value": raw["value"], "units": raw["units"], "confidence": raw["confidence"], "flags": raw["flags"],
         "components": raw["components"], "raw": raw}
    s.readings.append(r)
    out = {k: v for k, v in r.items() if k != "raw"}
    out["acceptable"] = s.acceptable(r)
    if r["value"] is not None:
        out["limit_status"] = limit_status(s.asset, r["value"])
    return out


def record_reading(s: Session, reading_id: str):
    r = s.reading(reading_id)
    if r is None or r["value"] is None:
        raise ToolError(f"{reading_id!r} is not a reading with a value")
    if s.recorded:
        raise ToolError("a reading has already been recorded for this run")
    if any(f in r["flags"] for f in FATAL_FLAGS):
        raise ToolError("the needle is outside the printed scale; escalate to a human instead of recording")
    if r["confidence"] < config.MIN_CONFIDENCE:
        raise ToolError(f"confidence {r['confidence']} is below the minimum {config.MIN_CONFIDENCE}; improve the "
                        "evidence, request a recapture, or escalate")
    why = s.needs_verification(r)
    partner = s.agreeing_reading(r) if why else None
    if why and partner is None:
        raise ToolError(f"verification required because the {why}: obtain a second reading from a different "
                        "image version or locate method that agrees within tolerance, or escalate")
    status = limit_status(s.asset, r["value"])
    s.recorded = {"asset_id": s.asset["asset_id"], "value": r["value"], "units": r["units"],
                  "confidence": r["confidence"], "limit_status": status, "reading_id": reading_id,
                  "verified_by": partner["reading_id"] if partner else None, "run_id": s.run_id, "ts": time.time()}
    s.store.put_reading(s.recorded)
    return {"recorded": True, "value": r["value"], "units": r["units"], "limit_status": status,
            "verified_by": s.recorded["verified_by"],
            "next": "raise_alert" if status != "normal" else "finished"}


def raise_alert(s: Session, message: str):
    if not s.recorded:
        raise ToolError("record a reading before raising an alert")
    status = s.recorded["limit_status"]
    if status == "normal":
        raise ToolError("the recorded value is within limits; no alert is warranted")
    if s.alert:
        raise ToolError("an alert already exists for this run")
    severity = "critical" if status.startswith("crit") else "warning"
    s.alert = {"asset_id": s.asset["asset_id"], "asset_name": s.asset["name"], "severity": severity,
               "limit_status": status, "value": s.recorded["value"], "units": s.recorded["units"],
               "message": message[:500], "run_id": s.run_id, "ts": time.time()}
    if severity == "critical":
        # Critical alerts can trigger costly responses, so a person confirms them first.
        s.alert["state"] = "pending_approval"
        return {"alert": "held", "severity": severity, "state": "pending_approval",
                "note": "critical alerts are dispatched only after an operator approves them"}
    s.alert["state"] = "dispatched"
    s.alert["delivery"] = s.store.send_alert(s.alert)
    return {"alert": "dispatched", "severity": severity, "delivery": s.alert["delivery"]}


def request_recapture(s: Session, reason: str, guidance: str):
    s.outcome = {"status": "recapture_requested", "reason": reason[:300], "guidance": guidance[:300]}
    return {"ok": True}


def escalate_to_human(s: Session, reason: str):
    s.outcome = {"status": "escalated", "reason": reason[:300]}
    return {"ok": True}


def _schema(props=None, required=None):
    return {"type": "object", "properties": props or {}, "required": required or [], "additionalProperties": False}


TOOLS = {
    "get_asset_context": (get_asset_context, "Get the gauge's range, units, alarm limits, recent recorded "
                          "readings and the recording policy. Call this first.", _schema()),
    "assess_quality": (tool_assess_quality, "Measure sharpness, brightness, contrast, noise and glare of the current "
                       "image with OpenCV (restricted to the dial once one is located). Returns a list of issues.",
                       _schema()),
    "enhance_image": (enhance_image, "Create an enhanced copy of the current image and make it current. Ops: "
                      "brighten (dark), denoise (noisy), clahe (low contrast), sharpen (blurry), deglare (specular "
                      "highlights). Choose the op from the measured issues.",
                      _schema({"op": {"type": "string", "enum": sorted(ENHANCE_OPS)}}, ["op"])),
    "locate_dial": (tool_locate_dial, "Find the dial rim in the current image. 'contour' fits an ellipse and handles "
                    "tilted views; 'hough' votes for a circle and tolerates a partly hidden rim.",
                    _schema({"method": {"type": "string", "enum": ["contour", "hough"]}}, ["method"])),
    "read_gauge": (tool_read_gauge, "Rectify the located dial, find the scale and the needle, and return a value "
                   "with a confidence and diagnostic flags. 'candidate' selects among the located rims (0 = best).",
                   _schema({"candidate": {"type": "integer"}}, ["candidate"])),
    "record_reading": (record_reading, "Record a reading as the authoritative value. Refused when confidence is "
                       "below the minimum, or when an out-of-limit or implausible value lacks an agreeing second "
                       "reading.", _schema({"reading_id": {"type": "string"}}, ["reading_id"])),
    "raise_alert": (raise_alert, "Raise an alert for a recorded out-of-limit value. Severity is set from the limits; "
                    "critical alerts are held for operator approval.",
                    _schema({"message": {"type": "string"}}, ["message"])),
    "request_recapture": (request_recapture, "End the run by asking the operator for a new photo, with specific "
                          "guidance on what to change.",
                          _schema({"reason": {"type": "string"}, "guidance": {"type": "string"}},
                                  ["reason", "guidance"])),
    "escalate_to_human": (escalate_to_human, "End the run by handing the evidence to a human reviewer, for example "
                          "when readings disagree or the needle is off the scale.",
                          _schema({"reason": {"type": "string"}}, ["reason"])),
}


def tool_specs():
    """Tool definitions in the Anthropic Messages API format."""
    return [{"name": name, "description": desc, "input_schema": schema, "strict": True}
            for name, (_, desc, schema) in TOOLS.items()]


def call_tool(s: Session, name: str, args: dict) -> dict:
    if name not in TOOLS:
        return {"error": f"unknown tool {name!r}"}
    try:
        return TOOLS[name][0](s, **(args or {}))
    except ToolError as e:
        return {"error": str(e)}
    except TypeError as e:
        return {"error": f"bad arguments for {name}: {e}"}
