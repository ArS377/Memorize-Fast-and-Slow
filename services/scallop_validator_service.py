#!/usr/bin/env python3
"""Small HTTP boundary for running scallopy from its Python 3.10 environment."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from neurosym.adapters import scallop as scallop_validator
from neurosym.adapters.scallop import validate_update_detailed
from neurosym.domain.validation_rules import DEFAULT_RULE_PARAMETERS, RuleParameters
from experiments.synthetic_temporal_preferences import resolve_preference_with_scallop
from experiments.private_lineage_reasoning import resolve_private_lineage_with_scallop
from experiments.preference_stream_injection import derive_preference_injections_with_scallop
from experiments.contradiction_ledger import derive_contradiction_ledger_with_scallop


MAX_REQUEST_BYTES = 10_000_000
MAX_PREFERENCE_EVENTS = 5_000


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
            "scallopy_version": importlib.metadata.version("scallopy"),
            "rule_version": DEFAULT_RULE_PARAMETERS.version,
        })

    def do_POST(self):
        if self.path not in {
            "/validate",
            "/resolve_preference",
            "/resolve_private_lineage",
            "/derive_preference_injections",
            "/derive_contradiction_ledger",
        }:
            self._send(404, {"error": "not_found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 1 or length > MAX_REQUEST_BYTES:
                raise ValueError(f"request body must be between 1 and {MAX_REQUEST_BYTES} bytes")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("request body must be an object")
            if self.path in {
                "/resolve_preference",
                "/resolve_private_lineage",
                "/derive_preference_injections",
                "/derive_contradiction_ledger",
            }:
                events = payload.get("events")
                if not isinstance(events, list) or len(events) > MAX_PREFERENCE_EVENTS:
                    raise ValueError(f"events must be a list with at most {MAX_PREFERENCE_EVENTS} entries")
                if not all(isinstance(event, dict) for event in events):
                    raise ValueError("events must contain only objects")
                if self.path == "/resolve_preference":
                    value = resolve_preference_with_scallop(
                        events,
                        str(payload["subject"]),
                        str(payload["date"]),
                        str(payload.get("scope", "default")),
                    )
                    self._send(200, {"preference": value})
                elif self.path == "/resolve_private_lineage":
                    query = payload.get("query")
                    if not isinstance(query, dict):
                        raise ValueError("query must be an object")
                    self._send(
                        200,
                        resolve_private_lineage_with_scallop(
                            events, query, mode=str(payload.get("mode") or "")
                        ),
                    )
                elif self.path == "/derive_preference_injections":
                    self._send(200, derive_preference_injections_with_scallop(events))
                else:
                    self._send(200, derive_contradiction_ledger_with_scallop(events))
            else:
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
