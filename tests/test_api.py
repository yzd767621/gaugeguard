import base64
import json

import cv2

from gaugeguard import config
from gaugeguard.api import handle
from gaugeguard.lambda_handler import handler as lambda_handler
from gaugeguard.mcp_server import Server


def _call(store, method, path, body=None, query="", headers=None):
    status, _, out = handle(method, path, query, headers or {}, json.dumps(body).encode() if body else b"", store)
    return status, (json.loads(out) if out[:1] in (b"{", b"[") else out)


def _b64(img):
    return base64.b64encode(cv2.imencode(".jpg", img)[1]).decode()


def test_health_assets_and_ui(store):
    assert _call(store, "GET", "/api/health")[1]["opencv"].startswith("5.")
    assert len(_call(store, "GET", "/api/assets")[1]["assets"]) == 5
    status, page = _call(store, "GET", "/")
    assert status == 200 and b"GaugeGuard" in page


def test_sample_then_inspect_round_trip(store):
    status, sample = _call(store, "GET", "/api/sample", query="condition=clean&seed=7")
    assert status == 200
    status, run = _call(store, "POST", "/api/inspect",
                        {"asset_id": sample["asset_id"], "image_b64": sample["image_jpeg_b64"], "planner": "rules"})
    assert status == 200 and run["recorded"]
    asset = store.get_asset(sample["asset_id"])
    assert abs(run["recorded"]["value"] - sample["true_value"]) < 0.02 * (asset["vmax"] - asset["vmin"])
    assert run["annotated_jpeg_b64"]
    assert _call(store, "GET", f"/api/runs/{run['run_id']}")[1]["run_id"] == run["run_id"]


def test_decision_endpoint_releases_a_held_alert(store, gauge):
    _, run = _call(store, "POST", "/api/inspect", {"asset_id": "PG-101", "image_b64": _b64(gauge("PG-101", 9.5)[0])})
    assert run["outcome"]["status"] == "alert_pending_approval"
    assert _call(store, "POST", f"/api/runs/{run['run_id']}/decision", {"approve": True})[0] == 400  # no operator
    status, done = _call(store, "POST", f"/api/runs/{run['run_id']}/decision", {"approve": True, "operator": "Sam"})
    assert status == 200 and done["outcome"]["status"] == "alert_raised"
    assert _call(store, "POST", f"/api/runs/{run['run_id']}/decision", {"approve": True, "operator": "Sam"})[0] == 400


def test_bad_requests_are_rejected(store):
    assert _call(store, "POST", "/api/inspect", {"asset_id": "PG-101"})[0] == 400
    assert _call(store, "POST", "/api/inspect", {"asset_id": "NOPE", "image_b64": "AAAA"})[0] == 404
    assert _call(store, "POST", "/api/inspect", {"asset_id": "PG-101", "image_b64": "not base64!"})[0] == 400
    assert _call(store, "POST", "/api/inspect", {"asset_id": "PG-101", "image_b64": "AAAA"})[0] == 400  # not an image
    assert _call(store, "GET", "/api/sample", query="condition=bogus")[0] == 400
    assert _call(store, "GET", "/api/runs/missing")[0] == 404
    assert _call(store, "GET", "/nope")[0] == 404


def test_token_protects_write_endpoints(store, gauge, monkeypatch):
    monkeypatch.setattr(config, "DEMO_TOKEN", "s3cret")
    body = {"asset_id": "PG-101", "image_b64": _b64(gauge("PG-101", 4.0)[0])}
    assert _call(store, "POST", "/api/inspect", body)[0] == 401
    assert _call(store, "POST", "/api/inspect", body, headers={"authorization": "Bearer wrong"})[0] == 401
    assert _call(store, "POST", "/api/inspect", body, headers={"authorization": "Bearer s3cret"})[0] == 200
    assert _call(store, "GET", "/api/health")[0] == 200


def test_lambda_function_url_event(monkeypatch, store):
    import gaugeguard.lambda_handler as lh
    monkeypatch.setattr(lh, "_store", store)
    event = {"requestContext": {"http": {"method": "GET"}}, "rawPath": "/api/health", "rawQueryString": "",
             "headers": {"Host": "example"}}
    resp = lambda_handler(event, None)
    assert resp["statusCode"] == 200 and json.loads(resp["body"])["ok"]


def test_mcp_server_exposes_the_tools(store, gauge, tmp_path):
    path = str(tmp_path / "g.jpg")
    cv2.imwrite(path, gauge("PG-101", 4.0)[0])
    srv = Server(store)

    def rpc(method, params=None, mid=1):
        return srv.handle({"jsonrpc": "2.0", "id": mid, "method": method, "params": params or {}})

    def tool(name, args):
        return json.loads(rpc("tools/call", {"name": name, "arguments": args})["result"]["content"][0]["text"])

    assert rpc("initialize")["result"]["serverInfo"]["name"] == "gaugeguard"
    names = [t["name"] for t in rpc("tools/list")["result"]["tools"]]
    assert "load_image" in names and "read_gauge" in names
    assert "error" in tool("read_gauge", {"candidate": 0})  # no session yet
    assert tool("load_image", {"path": path, "asset_id": "PG-101"})["loaded"]
    assert tool("locate_dial", {"method": "contour"})["found"]
    assert abs(tool("read_gauge", {"candidate": 0})["value"] - 4.0) < 0.05
    assert srv.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
    assert rpc("nope")["error"]["code"] == -32601
