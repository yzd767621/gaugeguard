"""Model-driven planner: Claude on Amazon Bedrock chooses each tool call from the OpenCV evidence."""
from __future__ import annotations

import json

from .. import config
from .policy import Action
from .tools import Session, tool_specs

SYSTEM = """You operate GaugeGuard, an inspection agent that reads analog gauges from photos.

You never see the photo. You work only through tools that run OpenCV on it, and you decide each next step from \
what the previous tool measured. The goal of every run is exactly one of these outcomes:
- record a trustworthy reading (and raise an alert when it is outside the asset's limits),
- ask the operator for a new photo, with concrete guidance on what to change, or
- escalate to a human reviewer.

How to work:
- Start with get_asset_context, then assess_quality, locate_dial and read_gauge.
- A reading comes with a confidence and diagnostic flags. When confidence is below the policy minimum, look at the \
measured quality issues and flags and pick the remedy that addresses them: an enhancement op, the other locate \
method, or another rim candidate. Re-locate and re-read after every enhancement.
- A value outside the asset's limits, or far from the last recorded reading, matters to people who will act on \
it. Confirm it with a second reading taken from a different image version or locate method before recording it. \
If the two readings disagree, escalate rather than choose between them.
- Stop improving after a few attempts. A photo with glare over the needle or a hidden scale cannot be fixed by \
processing; request a recapture and tell the operator what to change.
- Never state a value that no read_gauge call returned. The tools enforce the recording policy and will refuse \
calls the evidence does not support; a refusal tells you what is missing.

Before each tool call, write one short sentence saying what the last result showed and why you chose the next \
step. Operators read these sentences in the audit trail."""


class PlannerUnavailable(Exception):
    """The model could not be used for this run; the caller falls back to the rule planner."""


class BedrockPlanner:
    name = "bedrock"

    def __init__(self, model: str | None = None, region: str | None = None, client=None):
        if client is None:
            try:
                from anthropic import AnthropicBedrockMantle
            except ImportError as e:  # the SDK is optional for local, rules-only use
                raise PlannerUnavailable(f"anthropic SDK not installed: {e}") from e
            client = AnthropicBedrockMantle(aws_region=region or config.BEDROCK_REGION)
        self.client, self.model = client, model or config.BEDROCK_MODEL
        self.messages: list = []
        self.queue: list = []
        self.results: list = []
        self.usage = {"input_tokens": 0, "output_tokens": 0}

    def _ask(self):
        try:
            resp = self.client.messages.create(
                model=self.model, max_tokens=16000, system=SYSTEM, tools=tool_specs(),
                thinking={"type": "adaptive"}, output_config={"effort": "medium"}, messages=self.messages)
        except Exception as e:  # credentials, throttling, model access: all handled by falling back
            raise PlannerUnavailable(f"{type(e).__name__}: {e}") from e
        if resp.stop_reason == "refusal":
            raise PlannerUnavailable("model declined the request")
        self.usage["input_tokens"] += resp.usage.input_tokens
        self.usage["output_tokens"] += resp.usage.output_tokens
        # thinking blocks must be replayed unchanged, so keep the whole content list
        self.messages.append({"role": "assistant", "content": resp.content})
        note = " ".join(b.text.strip() for b in resp.content if b.type == "text").strip()
        return [Action(b.name, dict(b.input), note, b.id) for b in resp.content if b.type == "tool_use"]

    def step(self, s: Session, last=None) -> Action | None:
        if last is not None:
            action, result = last
            self.results.append({"type": "tool_result", "tool_use_id": action.id, "content": json.dumps(result),
                                 "is_error": "error" in result})
        if not self.queue:
            if not self.messages:
                self.messages.append({"role": "user", "content": f"Inspect asset {s.asset['asset_id']} "
                                      f"({s.asset['name']}). A photo has been loaded as the current image."})
            else:
                self.messages.append({"role": "user", "content": self.results})
                self.results = []
            self.queue = self._ask()
            if not self.queue:
                return None
        return self.queue.pop(0)
