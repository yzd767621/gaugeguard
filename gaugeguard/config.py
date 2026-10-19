"""Runtime settings, read from the environment so the same code runs locally and on Lambda."""
from __future__ import annotations

import os

# A reading below this confidence is never recorded as authoritative.
MIN_CONFIDENCE = float(os.environ.get("GG_MIN_CONFIDENCE", "0.6"))
# Two readings "agree" when they differ by no more than this fraction of full scale.
AGREE_FRACTION = float(os.environ.get("GG_AGREE_FRACTION", "0.02"))
MAX_STEPS = int(os.environ.get("GG_MAX_STEPS", "20"))
MAX_IMAGE_BYTES = int(os.environ.get("GG_MAX_IMAGE_BYTES", str(6 * 1024 * 1024)))
MAX_IMAGE_SIDE = int(os.environ.get("GG_MAX_IMAGE_SIDE", "960"))

PLANNER = os.environ.get("GG_PLANNER", "rules")  # "rules", "bedrock" (Converse API) or "claude"
PLANNER_MODEL = os.environ.get("GG_PLANNER_MODEL", "zai.glm-5")  # model for the "bedrock" planner
BEDROCK_MODEL = os.environ.get("GG_BEDROCK_MODEL", "anthropic.claude-opus-5-5")  # Claude planner
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
# Bedrock may not offer the model in the region the function runs in; this sends model calls elsewhere.
BEDROCK_REGION = os.environ.get("GG_BEDROCK_REGION") or AWS_REGION

DATA_DIR = os.environ.get("GG_DATA_DIR", os.path.join(os.getcwd(), ".gaugeguard"))
TABLE = os.environ.get("GG_TABLE")  # DynamoDB table name; unset means local file storage
BUCKET = os.environ.get("GG_BUCKET")
ALERT_TOPIC = os.environ.get("GG_ALERT_TOPIC")
DEMO_TOKEN = os.environ.get("GG_DEMO_TOKEN")  # when set, write endpoints require this bearer token
