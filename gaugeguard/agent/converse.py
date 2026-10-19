"""Model-driven planner for any tool-capable Bedrock model, through the Bedrock Converse API.

Same tools, prompt and guardrails as the Claude planner; only the wire format differs. It is used with models
that need no Anthropic Marketplace agreement (Amazon Nova, GLM, Kimi, DeepSeek, Mistral and others), so it
also works on AWS Free-plan accounts, where Claude is not available.
"""
from __future__ import annotations

import re

from .. import config
from .bedrock import SYSTEM, PlannerUnavailable
from .policy import Action
from .tools import TOOLS, Session

_THINKING = re.compile(r"</?thinking>")


def converse_tools():
    """Tool definitions in the Converse API format."""
    return {"tools": [{"toolSpec": {"name": name, "description": desc, "inputSchema": {"json": schema}}}
                      for name, (_, desc, schema) in TOOLS.items()]}


class ConversePlanner:

    def __init__(self, model: str | None = None, region: str | None = None, client=None):
        if client is None:
            import boto3
            client = boto3.client("bedrock-runtime", region_name=region or config.BEDROCK_REGION)
        self.client, self.model = client, model or config.PLANNER_MODEL
        self.name = "bedrock:" + re.sub(r"^(us|eu|apac|au|jp|global)\.", "", self.model)  # drop the routing prefix
        self.messages: list = []
        self.queue: list = []
        self.results: list = []
        self.usage = {"input_tokens": 0, "output_tokens": 0}

    def _ask(self):
        try:
            resp = self.client.converse(
                modelId=self.model, system=[{"text": SYSTEM}], messages=self.messages, toolConfig=converse_tools(),
                inferenceConfig={"maxTokens": 2000, "temperature": 0})  # greedy decoding is recommended for tool use
        except Exception as e:  # credentials, throttling, model access: all handled by falling back
            raise PlannerUnavailable(f"{type(e).__name__}: {e}") from e
        if resp.get("stopReason") in ("guardrail_intervened", "content_filtered"):
            raise PlannerUnavailable(f"model stopped: {resp['stopReason']}")
        self.usage["input_tokens"] += resp["usage"]["inputTokens"]
        self.usage["output_tokens"] += resp["usage"]["outputTokens"]
        content = resp["output"]["message"]["content"]
        self.messages.append({"role": "assistant", "content": content})
        note = " ".join(_THINKING.sub("", b["text"]).strip() for b in content if "text" in b).strip()
        return [Action(b["toolUse"]["name"], dict(b["toolUse"].get("input") or {}), note, b["toolUse"]["toolUseId"])
                for b in content if "toolUse" in b]

    def step(self, s: Session, last=None) -> Action | None:
        if last is not None:
            action, result = last
            self.results.append({"toolResult": {"toolUseId": action.id, "content": [{"json": result}],
                                                "status": "error" if "error" in result else "success"}})
        if not self.queue:
            if not self.messages:
                self.messages.append({"role": "user", "content": [{"text": f"Inspect asset {s.asset['asset_id']} "
                                      f"({s.asset['name']}). A photo has been loaded as the current image."}]})
            else:
                self.messages.append({"role": "user", "content": self.results})
                self.results = []
            self.queue = self._ask()
            if not self.queue:
                return None
        return self.queue.pop(0)
