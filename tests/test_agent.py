import json
from types import SimpleNamespace

import numpy as np
import pytest

from gaugeguard.agent import RulePlanner, Session, call_tool, decide_alert, run_inspection
from gaugeguard.agent.bedrock import BedrockPlanner, PlannerUnavailable
from gaugeguard.synth import degrade


def _tools(run):
    return [t["tool"] for t in run["trace"]]


def test_normal_reading_is_recorded_without_extra_steps(store, gauge):
    img, _ = gauge("PG-101", 4.0)
    run = run_inspection(img, "PG-101", store, RulePlanner())
    assert run["outcome"]["status"] == "recorded"
    assert abs(run["recorded"]["value"] - 4.0) < 0.05
    assert _tools(run) == ["get_asset_context", "assess_quality", "locate_dial", "read_gauge", "record_reading"]
    assert store.recent_readings("PG-101")[-1]["value"] == run["recorded"]["value"]


def test_out_of_limit_value_is_verified_then_alerted(store, gauge):
    img, _ = gauge("PG-101", 8.0)  # warn_high is 7.5
    run = run_inspection(img, "PG-101", store, RulePlanner())
    assert run["outcome"] == {"status": "alert_raised", "severity": "warning"}
    assert run["recorded"]["verified_by"] is not None
    tools = _tools(run)
    assert tools.count("read_gauge") == 2 and tools[-1] == "raise_alert"
    assert store.alerts and store.alerts[0]["state"] == "dispatched"


def test_critical_alert_waits_for_a_human(store, gauge):
    img, _ = gauge("PG-101", 9.5)  # crit_high is 9.0
    run = run_inspection(img, "PG-101", store, RulePlanner())
    assert run["outcome"]["status"] == "alert_pending_approval"
    assert store.alerts == []  # nothing dispatched yet
    approved = decide_alert(store, run["run_id"], True, "A. Operator")
    assert approved["outcome"]["status"] == "alert_raised" and len(store.alerts) == 1


def test_rejected_critical_alert_is_never_sent(store, gauge):
    img, _ = gauge("PG-101", 9.5)
    run = run_inspection(img, "PG-101", store, RulePlanner())
    rejected = decide_alert(store, run["run_id"], False, "A. Operator", "gauge under maintenance")
    assert rejected["outcome"]["status"] == "alert_rejected" and store.alerts == []


def test_quality_evidence_changes_the_next_tool_call(store, gauge):
    img, truth = gauge("PG-101", 4.0)
    dark, _ = degrade(img, truth, "low_light", 0.9, np.random.default_rng(3))
    run = run_inspection(dark, "PG-101", store, RulePlanner())
    enh = [t for t in run["trace"] if t["tool"] == "enhance_image"]
    assert enh and enh[0]["args"]["op"] == "denoise"
    assert run["outcome"]["status"] == "recorded" and abs(run["recorded"]["value"] - 4.0) < 0.1


def test_unreadable_photo_ends_in_a_recapture_request(store):
    blank = np.full((480, 640, 3), 127, np.uint8)
    run = run_inspection(blank, "PG-101", store, RulePlanner())
    assert run["outcome"]["status"] == "recapture_requested" and run["outcome"]["guidance"]
    assert run["recorded"] is None and store.recent_readings("PG-101") == []


def test_implausible_jump_needs_a_second_reading(store, gauge):
    store.put_reading({"asset_id": "PG-101", "value": 1.5, "ts": 1.0})
    img, _ = gauge("PG-101", 6.0)  # within limits but 4.5 bar above the last reading; max_step is 3
    run = run_inspection(img, "PG-101", store, RulePlanner())
    assert run["outcome"]["status"] == "recorded" and run["recorded"]["verified_by"]


# ---------------------------------------------------------------- guardrails

def _session(store, img, asset_id="PG-101"):
    s = Session(img, store.get_asset(asset_id), store, "test")
    call_tool(s, "locate_dial", {"method": "contour"})
    return s, call_tool(s, "read_gauge", {"candidate": 0})


def test_guardrail_refuses_unverified_out_of_limit_reading(store, gauge):
    s, r = _session(store, gauge("PG-101", 8.0)[0])
    out = call_tool(s, "record_reading", {"reading_id": r["reading_id"]})
    assert "verification required" in out["error"] and s.recorded is None


def test_guardrail_refuses_low_confidence_reading(store, gauge):
    s, r = _session(store, gauge("PG-101", 4.0)[0])
    s.readings[0]["confidence"] = 0.2
    assert "below the minimum" in call_tool(s, "record_reading", {"reading_id": r["reading_id"]})["error"]


def test_guardrail_refuses_alert_for_in_range_value(store, gauge):
    s, r = _session(store, gauge("PG-101", 4.0)[0])
    assert call_tool(s, "record_reading", {"reading_id": r["reading_id"]})["recorded"]
    assert "within limits" in call_tool(s, "raise_alert", {"message": "panic"})["error"]


def test_bad_tool_calls_return_errors_not_exceptions(store, gauge):
    s, _ = _session(store, gauge("PG-101", 4.0)[0])
    assert "error" in call_tool(s, "no_such_tool", {})
    assert "error" in call_tool(s, "enhance_image", {"op": "magic"})
    assert "error" in call_tool(s, "read_gauge", {"candidate": 99})
    assert "error" in call_tool(s, "record_reading", {"reading_id": "r99"})


# ------------------------------------------------------------ model planner

class _FakeClient:
    """Stands in for the Anthropic Bedrock client and scripts the model's tool calls."""

    def __init__(self, script):
        self.script, self.calls = list(script), []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls.append({**kwargs, "messages": list(kwargs["messages"])})
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        content = [SimpleNamespace(type="text", text=step.get("say", ""))]
        content += [SimpleNamespace(type="tool_use", id=f"tu{len(self.calls)}_{i}", name=n, input=a)
                    for i, (n, a) in enumerate(step.get("calls", []))]
        return SimpleNamespace(content=content, stop_reason="tool_use" if step.get("calls") else "end_turn",
                               usage=SimpleNamespace(input_tokens=10, output_tokens=5))


def test_model_planner_drives_the_same_tools(store, gauge):
    client = _FakeClient([
        {"say": "Load context and check quality.", "calls": [("get_asset_context", {}), ("assess_quality", {})]},
        {"say": "Quality is fine; find the dial.", "calls": [("locate_dial", {"method": "contour"})]},
        {"say": "Dial found; read it.", "calls": [("read_gauge", {"candidate": 0})]},
        {"say": "Confident and in range; record.", "calls": [("record_reading", {"reading_id": "r1"})]},
    ])
    run = run_inspection(gauge("PG-101", 4.0)[0], "PG-101", store, BedrockPlanner(client=client))
    assert run["outcome"]["status"] == "recorded" and run["planner"] == "bedrock"
    assert run["trace"][2]["reason"] == "Quality is fine; find the dial."
    # both results of the parallel first turn go back in a single user message
    results = client.calls[1]["messages"][-1]["content"]
    assert [r["type"] for r in results] == ["tool_result", "tool_result"]
    assert json.loads(results[0]["content"])["asset"]["asset_id"] == "PG-101"
    assert all(t["strict"] for t in client.calls[0]["tools"])


def test_model_cannot_talk_its_way_past_a_guardrail(store, gauge):
    client = _FakeClient([
        {"calls": [("locate_dial", {"method": "contour"})]},
        {"calls": [("read_gauge", {"candidate": 0})]},
        {"say": "Looks high, recording anyway.", "calls": [("record_reading", {"reading_id": "r1"})]},
        {"say": "Refused; escalate.", "calls": [("escalate_to_human", {"reason": "could not verify high value"})]},
    ])
    run = run_inspection(gauge("PG-101", 8.0)[0], "PG-101", store, BedrockPlanner(client=client))
    assert run["outcome"]["status"] == "escalated" and run["recorded"] is None
    refused = client.calls[3]["messages"][-1]["content"][0]
    assert refused["is_error"] and "verification required" in refused["content"]


def test_rules_take_over_when_the_model_is_unavailable(store, gauge):
    client = _FakeClient([{"calls": [("get_asset_context", {})]}, RuntimeError("throttled")])
    run = run_inspection(gauge("PG-101", 4.0)[0], "PG-101", store, BedrockPlanner(client=client))
    assert run["planner"] == "bedrock+rules" and run["outcome"]["status"] == "recorded"
    assert any(t["tool"] == "planner_fallback" for t in run["trace"])


def test_model_that_stops_without_deciding_is_escalated(store, gauge):
    client = _FakeClient([{"say": "I am done."}])
    run = run_inspection(gauge("PG-101", 4.0)[0], "PG-101", store, BedrockPlanner(client=client))
    assert run["outcome"]["status"] == "escalated"


def test_refusal_makes_the_planner_unavailable():
    client = _FakeClient([])
    client.messages.create = lambda **k: SimpleNamespace(stop_reason="refusal", content=[], usage=None)
    with pytest.raises(PlannerUnavailable):
        BedrockPlanner(client=client)._ask()
