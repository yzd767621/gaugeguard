"""AWS Lambda entry point for a Function URL (payload format 2.0)."""
from __future__ import annotations

import base64

from .api import handle
from .store import default_store

_store = None  # created once per execution environment and reused across invocations


def handler(event, context):
    global _store
    if _store is None:
        _store = default_store()
    http = event.get("requestContext", {}).get("http", {})
    body = event.get("body") or ""
    raw = base64.b64decode(body) if event.get("isBase64Encoded") else body.encode()
    headers = {k.lower(): v for k, v in (event.get("headers") or {}).items()}
    status, out_headers, out_body = handle(http.get("method", "GET"), event.get("rawPath", "/"),
                                           event.get("rawQueryString", ""), headers, raw, _store)
    return {"statusCode": status, "headers": out_headers, "body": out_body.decode("utf-8"),
            "isBase64Encoded": False}
