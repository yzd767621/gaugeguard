"""Local development server:  python -m gaugeguard.serve [--port 8000]"""
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

from . import config
from .api import handle
from .store import default_store


def make_handler(store):
    class Handler(BaseHTTPRequestHandler):
        def _serve(self):
            parts = urlsplit(self.path)
            length = int(self.headers.get("Content-Length") or 0)
            if length > 2 * config.MAX_IMAGE_BYTES:
                self.send_error(413)
                return
            body = self.rfile.read(length) if length else b""
            headers = {k.lower(): v for k, v in self.headers.items()}
            status, out_headers, out_body = handle(self.command, parts.path, parts.query, headers, body, store)
            self.send_response(status)
            for k, v in out_headers.items():
                self.send_header(k, v)
            self.send_header("Content-Length", str(len(out_body)))
            self.end_headers()
            self.wfile.write(out_body)

        do_GET = do_POST = _serve

        def log_message(self, fmt, *args):
            print("%s - %s" % (self.address_string(), fmt % args))

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(default_store()))
    print(f"GaugeGuard on http://{args.host}:{args.port}  (planner: {config.PLANNER})")
    server.serve_forever()


if __name__ == "__main__":
    main()
