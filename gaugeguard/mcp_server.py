"""Minimal Model Context Protocol server (stdio, JSON-RPC 2.0) exposing the OpenCV 5 tools.

Lets any MCP-capable agent drive the same perception tools and guardrails:

    python -m gaugeguard.mcp_server

Tools: every tool in ``gaugeguard.agent.tools`` plus ``load_image`` (start a session from a file path).
Only the parts of MCP needed for tools are implemented: initialize, tools/list, tools/call, ping.
"""
from __future__ import annotations

import json
import sys
import uuid

import cv2

from . import __version__
from .agent.loop import fit_image
from .agent.tools import TOOLS, Session, call_tool
from .store import default_store

PROTOCOL = "2025-06-18"
_LOAD = {"name": "load_image", "description": "Start an inspection session from an image file path and an asset id. "
         "Call this before any other tool.",
         "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}, "asset_id": {"type": "string"}},
                         "required": ["path", "asset_id"], "additionalProperties": False}}


class Server:
    def __init__(self, store=None):
        self.store = store or default_store()
        self.session: Session | None = None

    def tools(self):
        return [_LOAD] + [{"name": n, "description": d, "inputSchema": sch} for n, (_, d, sch) in TOOLS.items()]

    def call(self, name, args):
        if name == "load_image":
            asset = self.store.get_asset(args.get("asset_id", ""))
            img = cv2.imread(args.get("path", ""), cv2.IMREAD_COLOR)
            if asset is None or img is None:
                return {"error": "unknown asset_id or unreadable image path"}
            self.session = Session(fit_image(img), asset, self.store, uuid.uuid4().hex[:12])
            return {"loaded": True, "run_id": self.session.run_id, "size": list(self.session.images["original"].shape[:2])}
        if self.session is None:
            return {"error": "call load_image first"}
        return call_tool(self.session, name, args)

    def handle(self, msg: dict):
        method, mid = msg.get("method"), msg.get("id")
        if mid is None:  # notification
            return None
        if method == "initialize":
            result = {"protocolVersion": PROTOCOL, "capabilities": {"tools": {}},
                      "serverInfo": {"name": "gaugeguard", "version": __version__}}
        elif method == "ping":
            result = {}
        elif method == "tools/list":
            result = {"tools": self.tools()}
        elif method == "tools/call":
            params = msg.get("params") or {}
            out = self.call(params.get("name"), params.get("arguments") or {})
            out = {k: v for k, v in out.items() if k != "raw"}
            result = {"content": [{"type": "text", "text": json.dumps(out, default=str)}], "isError": "error" in out}
        else:
            return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"method not found: {method}"}}
        return {"jsonrpc": "2.0", "id": mid, "result": result}


def main():
    server = Server()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            reply = server.handle(json.loads(line))
        except json.JSONDecodeError:
            reply = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "parse error"}}
        if reply is not None:
            sys.stdout.write(json.dumps(reply) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
