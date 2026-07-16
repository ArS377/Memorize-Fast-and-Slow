#!/usr/bin/env python3
"""Small HTTP boundary for running scallopy from its Python 3.10 environment."""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import scallop_validator
from scallop_validator import DEFAULT_RULE_PARAMETERS, RuleParameters, validate_update_detailed


def _rule_parameters(payload):
    raw = payload.get("rule_parameters") or {}
    if not isinstance(raw, dict):
        raise ValueError("rule_parameters must be an object")
    return RuleParameters(
        version=str(raw.get("version") or DEFAULT_RULE_PARAMETERS.version),
        functional_predicates=tuple(raw.get("functional_predicates") or DEFAULT_RULE_PARAMETERS.functional_predicates),
        generic_objects=tuple(raw.get("generic_objects") or DEFAULT_RULE_PARAMETERS.generic_objects),
        confidence_weights=dict(raw.get("confidence_weights") or DEFAULT_RULE_PARAMETERS.confidence_weights),
        provenance_weight=int(raw.get("provenance_weight", DEFAULT_RULE_PARAMETERS.provenance_weight)),
        confidence_multiplier=int(raw.get("confidence_multiplier", DEFAULT_RULE_PARAMETERS.confidence_multiplier)),
    )


class Handler(BaseHTTPRequestHandler):
    def _send(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path != "/health":
            self._send(404, {"error": "not_found"})
            return
        self._send(200, {
            "status": "ok",
            "engine": "scallopy",
            "scallop_available": scallop_validator.scallopy is not None,
            "rule_version": DEFAULT_RULE_PARAMETERS.version,
        })

    def do_POST(self):
        if self.path != "/validate":
            self._send(404, {"error": "not_found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            decision = validate_update_detailed(
                payload.get("existing_facts") or [],
                payload["new_fact"],
                rule_params=_rule_parameters(payload),
            )
            self._send(200, decision.to_dict())
        except Exception as exc:
            self._send(400, {"error": type(exc).__name__, "message": str(exc)})

    def log_message(self, format, *args):
        return


def main():
    if scallop_validator.scallopy is None:
        raise SystemExit("scallopy is unavailable; refusing to start a mislabeled Scallop service")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
