from __future__ import annotations

import json
import os
from pathlib import Path

from experiments.persona_conversation_generator import LLMResponse


def paper_input_dir(name: str) -> Path:
    root = Path(os.environ.get("NEUROSYM_TEST_EVIDENCE_ROOT", Path(__file__).resolve().parents[1]))
    return root / "results" / name


class FakeKimiClient:
    """Return assignment-local mappings and valid dialogue without network access."""

    def complete(
        self,
        *,
        messages: list[dict[str, str]],
        model: str,
        timeout: float,
        seed: int,
        max_tokens: int,
    ) -> LLMResponse:
        del timeout, max_tokens
        payload = json.loads(messages[-1]["content"])
        if "mapping" in payload:
            realization_id = str(payload.get("realization_id", ""))
            prefix = "alpha" if "surface-a" in realization_id else "bravo"
            mapping = []
            for index, row in enumerate(payload["mapping"]):
                history_token = str(row["history_id"]).rsplit("-", 1)[-1]
                stem = f"{prefix}{history_token}{index}"
                target = {
                    "tea": f"{stem} tea",
                    "seating": f"{stem} seating",
                    "food": f"{stem} curry",
                    "delivery": f"{stem} delivery",
                    "receipt": f"{stem} receipts",
                    "private_lineage": f"{stem} token",
                }[row["category"]]
                mapping.append({**row, "target_phrase": target})
            content = json.dumps({"mapping": mapping})
        else:
            events = []
            for event in payload["events"]:
                forbidden = [
                    str(value).casefold()
                    for value in event["forbidden_surface_phrases"]
                ]
                markers = [
                    marker
                    for marker in event["semantic_markers"]
                    if not any(str(marker).casefold() in phrase for phrase in forbidden)
                ]
                parts = ["Please consider"]
                parts.extend(str(value) for value in event["required_surface_values"])
                if markers:
                    parts.append(str(markers[0]))
                if event["authority_markers"]:
                    parts.append(str(event["authority_markers"][0]))
                events.append(
                    {
                        "event_id": event["event_id"],
                        "turns": [
                            {"role": "user", "content": " ".join(parts) + "."},
                            {"role": "assistant", "content": "Understood clearly."},
                        ],
                    }
                )
            content = json.dumps({"events": events})
        return LLMResponse(
            content=content,
            finish_reason="stop",
            model=model,
            usage={"prompt_tokens": 1, "completion_tokens": 1},
        )
