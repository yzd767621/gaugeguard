"""Persistence for runs, readings and alerts: local files for development, DynamoDB + S3 + SNS on AWS."""
from __future__ import annotations

import json
import os
import time

from . import config

_ASSETS_FILE = os.path.join(os.path.dirname(__file__), "assets.json")


def load_assets() -> dict:
    with open(_ASSETS_FILE, encoding="utf-8") as f:
        return {a["asset_id"]: a for a in json.load(f)}


class MemoryStore:
    """In-process store; also the base class that defines the interface."""

    def __init__(self, assets: dict | None = None):
        self.assets = assets if assets is not None else load_assets()
        self._readings, self._runs, self._images, self.alerts = {}, {}, {}, []

    def get_asset(self, asset_id):
        return self.assets.get(asset_id)

    def list_assets(self):
        return list(self.assets.values())

    def recent_readings(self, asset_id, limit=5):
        return self._readings.get(asset_id, [])[-limit:]

    def put_reading(self, record):
        self._readings.setdefault(record["asset_id"], []).append(record)

    def put_run(self, run):
        self._runs[run["run_id"]] = run

    def get_run(self, run_id):
        return self._runs.get(run_id)

    def save_image(self, run_id, name, data: bytes):
        self._images[f"{run_id}/{name}"] = data
        return f"{run_id}/{name}"

    def load_image(self, key):
        return self._images.get(key)

    def send_alert(self, alert):
        self.alerts.append(alert)
        return {"delivered": "memory"}


class LocalStore(MemoryStore):
    """JSON files under ``GG_DATA_DIR``."""

    def __init__(self, root=None, assets=None):
        super().__init__(assets)
        self.root = root or config.DATA_DIR
        for sub in ("runs", "readings", "images"):
            os.makedirs(os.path.join(self.root, sub), exist_ok=True)

    def _path(self, sub, name):
        safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in name)
        return os.path.join(self.root, sub, safe)

    def recent_readings(self, asset_id, limit=5):
        path = self._path("readings", asset_id + ".jsonl")
        if not os.path.exists(path):
            return []
        with open(path, encoding="utf-8") as f:
            return [json.loads(line) for line in f.readlines()[-limit:]]

    def put_reading(self, record):
        with open(self._path("readings", record["asset_id"] + ".jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def put_run(self, run):
        with open(self._path("runs", run["run_id"] + ".json"), "w", encoding="utf-8") as f:
            json.dump(run, f)

    def get_run(self, run_id):
        path = self._path("runs", run_id + ".json")
        if not os.path.exists(path):
            return None
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def save_image(self, run_id, name, data: bytes):
        key = f"{run_id}_{name}"
        with open(self._path("images", key), "wb") as f:
            f.write(data)
        return key

    def load_image(self, key):
        path = self._path("images", key)
        if not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            return f.read()

    def send_alert(self, alert):
        with open(os.path.join(self.root, "alerts.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(alert) + "\n")
        return {"delivered": "local-file"}


class AwsStore(MemoryStore):
    """DynamoDB single table (pk/sk), S3 for images, SNS for alerts."""

    def __init__(self, table=None, bucket=None, topic=None, assets=None):
        super().__init__(assets)
        import boto3

        self.table = boto3.resource("dynamodb").Table(table or config.TABLE)
        self.s3, self.bucket = boto3.client("s3"), bucket or config.BUCKET
        self.sns, self.topic = boto3.client("sns"), topic or config.ALERT_TOPIC

    def recent_readings(self, asset_id, limit=5):
        from boto3.dynamodb.conditions import Key

        resp = self.table.query(KeyConditionExpression=Key("pk").eq(f"ASSET#{asset_id}"),
                                ScanIndexForward=False, Limit=limit)
        return [json.loads(i["body"]) for i in reversed(resp.get("Items", []))]

    def put_reading(self, record):
        self.table.put_item(Item={"pk": f"ASSET#{record['asset_id']}", "sk": f"READING#{record['ts']:.3f}",
                                  "body": json.dumps(record)})

    def put_run(self, run):
        self.table.put_item(Item={"pk": f"RUN#{run['run_id']}", "sk": "RUN", "body": json.dumps(run),
                                  "expires_at": int(time.time()) + 30 * 86400})

    def get_run(self, run_id):
        item = self.table.get_item(Key={"pk": f"RUN#{run_id}", "sk": "RUN"}).get("Item")
        return json.loads(item["body"]) if item else None

    def save_image(self, run_id, name, data: bytes):
        key = f"runs/{run_id}/{name}"
        self.s3.put_object(Bucket=self.bucket, Key=key, Body=data, ContentType="image/jpeg")
        return key

    def load_image(self, key):
        try:
            return self.s3.get_object(Bucket=self.bucket, Key=key)["Body"].read()
        except self.s3.exceptions.NoSuchKey:
            return None

    def send_alert(self, alert):
        if not self.topic:
            return {"delivered": "none", "reason": "no alert topic configured"}
        subject = f"GaugeGuard {alert['severity']}: {alert['asset_id']}"[:100]
        resp = self.sns.publish(TopicArn=self.topic, Subject=subject, Message=json.dumps(alert, indent=2))
        return {"delivered": "sns", "message_id": resp["MessageId"]}


def default_store():
    return AwsStore() if config.TABLE else LocalStore()
