"""HTTP API as one transport-neutral function, shared by the local server and the Lambda handler."""
from __future__ import annotations

import base64
import hmac
import json
import os
import time
import traceback
from urllib.parse import parse_qs

import cv2

from . import __version__, config
from .agent import decide_alert, decode_image, make_planner, run_inspection
from .synth import CONDITIONS, make_sample

_WEB = os.path.join(os.path.dirname(__file__), "web", "index.html")
_SECURITY = {"X-Content-Type-Options": "nosniff", "Referrer-Policy": "no-referrer",
             "Cache-Control": "no-store"}


def _json(status, obj):
    return status, {"Content-Type": "application/json", **_SECURITY}, json.dumps(obj).encode()


def _error(status, message):
    return _json(status, {"error": message})


def _authorised(headers) -> bool:
    if not config.DEMO_TOKEN:
        return True
    supplied = headers.get("authorization", "").removeprefix("Bearer ").strip()
    return hmac.compare_digest(supplied, config.DEMO_TOKEN)


def _with_images(store, run):
    out = dict(run)
    for name, key in (run.get("images") or {}).items():
        data = store.load_image(key)
        if data:
            out[f"{name}_jpeg_b64"] = base64.b64encode(data).decode()
    return out


def _emit_metrics(run):
    """CloudWatch Embedded Metric Format: one structured log line becomes metrics, no API call needed."""
    reading = run.get("reading") or {}
    print(json.dumps({
        "_aws": {"Timestamp": int(time.time() * 1000), "CloudWatchMetrics": [{
            "Namespace": "GaugeGuard", "Dimensions": [["Outcome"]],
            "Metrics": [{"Name": "Runs", "Unit": "Count"}, {"Name": "Steps", "Unit": "Count"},
                        {"Name": "DurationMs", "Unit": "Milliseconds"}, {"Name": "Confidence", "Unit": "None"}]}]},
        "Outcome": run["outcome"]["status"], "Runs": 1, "Steps": run["steps"], "DurationMs": run["duration_ms"],
        "Confidence": reading.get("confidence") or 0.0, "run_id": run["run_id"], "asset_id": run["asset_id"],
        "planner": run["planner"]}))


def handle(method: str, path: str, query: str, headers: dict, body: bytes, store):
    """Return ``(status, headers, body_bytes)``. ``headers`` keys must be lower-case."""
    method, path = method.upper(), path.rstrip("/") or "/"
    try:
        if method == "GET" and path == "/":
            with open(_WEB, "rb") as f:
                return 200, {"Content-Type": "text/html; charset=utf-8", **_SECURITY}, f.read()
        if method == "GET" and path == "/api/health":
            return _json(200, {"ok": True, "version": __version__, "opencv": cv2.__version__,
                               "planner": config.PLANNER, "auth_required": bool(config.DEMO_TOKEN)})
        if method == "GET" and path == "/api/assets":
            return _json(200, {"assets": store.list_assets(), "conditions": CONDITIONS})
        if method == "GET" and path == "/api/sample":
            q = parse_qs(query or "")
            cond = q.get("condition", ["clean"])[0]
            if cond not in CONDITIONS:
                return _error(400, f"condition must be one of {CONDITIONS}")
            seed = int(q.get("seed", [str(int(time.time() * 1000) % 100000)])[0])
            img, truth = make_sample(seed, cond)
            asset = next(a for a in store.list_assets()
                         if (a["vmin"], a["vmax"], a["units"]) == (truth.spec["vmin"], truth.spec["vmax"],
                                                                   truth.spec["units"]))
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            return _json(200, {"seed": seed, "condition": cond, "asset_id": asset["asset_id"],
                               "true_value": round(truth.value, 4), "image_jpeg_b64": base64.b64encode(buf).decode()})
        if method == "GET" and path.startswith("/api/runs/"):
            run = store.get_run(path.split("/")[3])
            return _json(200, _with_images(store, run)) if run else _error(404, "run not found")

        if method == "POST" and path == "/api/inspect":
            if not _authorised(headers):
                return _error(401, "missing or wrong bearer token")
            req = json.loads(body or b"{}")
            if not isinstance(req.get("image_b64"), str) or not isinstance(req.get("asset_id"), str):
                return _error(400, "asset_id and image_b64 are required")
            if store.get_asset(req["asset_id"]) is None:
                return _error(404, "unknown asset")
            image = decode_image(base64.b64decode(req["image_b64"], validate=True))
            planner_name = req.get("planner") or config.PLANNER
            if planner_name not in ("rules", "bedrock"):
                return _error(400, "planner must be 'rules' or 'bedrock'")
            try:
                planner = make_planner(planner_name)
            except Exception as e:  # Bedrock client could not be built: run with the rule planner and say so
                planner = make_planner("rules")
                planner.name = f"rules (bedrock unavailable: {type(e).__name__})"
            run = run_inspection(image, req["asset_id"], store, planner)
            _emit_metrics(run)
            return _json(200, _with_images(store, run))
        if method == "POST" and path.startswith("/api/runs/") and path.endswith("/decision"):
            if not _authorised(headers):
                return _error(401, "missing or wrong bearer token")
            req = json.loads(body or b"{}")
            if not isinstance(req.get("approve"), bool) or not str(req.get("operator", "")).strip():
                return _error(400, "approve (boolean) and operator (name) are required")
            run = decide_alert(store, path.split("/")[3], req["approve"], str(req["operator"]), str(req.get("note", "")))
            return _json(200, _with_images(store, run))
        return _error(404, "not found")
    except (ValueError, json.JSONDecodeError) as e:
        return _error(400, str(e))
    except KeyError as e:
        return _error(404, f"not found: {e}")
    except Exception as e:  # never drop the connection: the caller always gets a JSON error it can show
        traceback.print_exc()
        return _error(500, f"internal error ({type(e).__name__}); see the server log")
