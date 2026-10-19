"""Deterministic planner.

It makes the same kind of decisions as the model-driven planner (each step is
chosen from the latest OpenCV evidence) but needs no network, so it is used for
tests, for the reproducible evaluation, and as the fallback when Bedrock is
unavailable. It reads only the session state, so it can take over mid-run.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .tools import FATAL_FLAGS, Session, limit_status

MAX_REMEDIES = 4
# Remedies per measured issue, in the order to try them. Denoising comes first because, on the development
# images, noise is what most often hides thin tick marks, and sharpening or CLAHE amplify it.
ISSUE_TO_OPS = {"noisy": ["denoise"], "dark": ["denoise", "brighten"], "blurry": ["sharpen"],
                "glare": ["deglare"], "low_contrast": ["clahe"]}
FALLBACK_OPS = ["denoise", "sharpen"]
GUIDANCE = {
    "glare": "A reflection hides part of the dial. Step to one side or shade the glass, then retake.",
    "dark": "The dial is too dark. Add light or use the torch, then retake.",
    "blurry": "The photo is blurred. Hold the camera steady, tap to focus on the dial, then retake.",
    "noisy": "The photo is very grainy. Add light so the camera can use a lower gain, then retake.",
    "scale_partially_hidden": "Part of the scale is hidden. Clear whatever is in front of the dial, then retake.",
    "scale_not_found": "The scale markings are not readable. Move closer and face the dial squarely, then retake.",
    "dial_not_found": "No dial was found. Centre the whole gauge in the frame, then retake.",
}


@dataclass
class Action:
    tool: str
    args: dict = field(default_factory=dict)
    reason: str = ""
    id: str | None = None


class RulePlanner:
    name = "rules"

    def __init__(self):
        self.tried: set = set()
        self.verifying = False

    # -- remedies ----------------------------------------------------------
    def _remedy(self, s: Session, problem_flags) -> Action | None:
        if len(self.tried) >= MAX_REMEDIES:
            return None
        issues = list(s.quality.get("original", {}).get("issues", []))
        for issue in issues:
            for op in ISSUE_TO_OPS[issue]:
                if op not in self.tried:
                    self.tried.add(op)
                    return Action("enhance_image", {"op": op}, f"Quality check reported '{issue}', so apply {op} "
                                  "and look again.")
        loc = s.locates.get(s.current)
        geometric = {"scale_not_found", "scale_sweep_mismatch", "pivot_inconsistent", "pivot_not_found",
                     "dial_not_found"} & set(problem_flags)
        if geometric and "hough" not in self.tried:
            self.tried.add("hough")
            return Action("locate_dial", {"method": "hough"}, f"Geometry looks wrong ({', '.join(sorted(geometric))})"
                          ", so re-locate the rim by circle voting.")
        if loc and loc["found"] and len(loc["candidates"]) > 1 and "candidate1" not in self.tried:
            self.tried.add("candidate1")
            return Action("read_gauge", {"candidate": 1}, "Try the second rim candidate; the first may be the bezel "
                          "or a reflection.")
        for op in FALLBACK_OPS:
            if op not in self.tried:
                self.tried.add(op)
                return Action("enhance_image", {"op": op}, f"No specific defect explains the low confidence; "
                              f"try {op} and look again.")
        return None

    def _give_up(self, s: Session, problem_flags) -> Action:
        best = s.best_reading()
        if best and any(f in best["flags"] for f in FATAL_FLAGS) and best["confidence"] >= 0.4:
            return Action("escalate_to_human", {"reason": "The needle appears to be beyond the printed scale. This can "
                          "be a pegged gauge or a misread, so a person should look."}, "Off-scale needle.")
        issues = list(s.quality.get("original", {}).get("issues", [])) + list(problem_flags)
        key = next((k for k in ("glare", "scale_partially_hidden", "dark", "blurry", "noisy", "scale_not_found",
                                "dial_not_found") if k in issues), "scale_not_found")
        why = "no dial found" if "dial_not_found" in problem_flags else (
            f"best confidence {best['confidence']:.2f} is below the minimum" if best else "no reading obtained")
        return Action("request_recapture", {"reason": f"Could not obtain a trustworthy reading ({why}).",
                                            "guidance": GUIDANCE[key]}, "Remedies exhausted.")

    # -- main --------------------------------------------------------------
    def step(self, s: Session, last=None) -> Action | None:
        if s.done:
            return None
        if not s.context_loaded:
            return Action("get_asset_context", {}, "Load the gauge range and alarm limits first.")
        if "original" not in s.quality:
            return Action("assess_quality", {}, "Measure image quality before trusting any reading.")
        if s.recorded:
            status = limit_status(s.asset, s.recorded["value"])
            return Action("raise_alert", {"message": f"{s.asset['name']} read {s.recorded['value']:.3g} "
                          f"{s.asset['units']} ({status}), confirmed by two independent readings."},
                          "Recorded value is outside its limits.")

        loc = s.locates.get(s.current)
        if loc is None:
            return Action("locate_dial", {"method": "contour"}, "Find the dial rim on the current image.")
        if not loc["found"]:
            return self._remedy(s, ["dial_not_found"]) or self._give_up(s, ["dial_not_found"])
        if s.latest_reading_on(s.current, f"{loc['method']}#0") is None:
            return Action("read_gauge", {"candidate": 0}, "Dial located; read it.")

        best = s.best_reading()
        if s.acceptable(best):
            why = s.needs_verification(best)
            if not why:
                return Action("record_reading", {"reading_id": best["reading_id"]},
                              f"Confidence {best['confidence']:.2f} meets the minimum and the value is in range.")
            if s.agreeing_reading(best):
                return Action("record_reading", {"reading_id": best["reading_id"]},
                              "A second, independent reading agrees.")
            if not self.verifying:
                self.verifying = True
                op = next((o for o in ("denoise", "brighten", "sharpen") if o not in self.tried), "clahe")
                self.tried.add(op)
                return Action("enhance_image", {"op": op}, f"The {why}. Before acting on that, take an independent "
                              "second reading from a re-processed image.")
            return Action("escalate_to_human", {"reason": f"The {why}, but a second reading did not confirm it."},
                          "Independent readings disagree.")

        flags = best["flags"] if best else ["scale_not_found"]
        return self._remedy(s, flags) or self._give_up(s, flags)
