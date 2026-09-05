from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from experiments.persona_conversation_generator import LLMResponse


def paper_input_dir(name: str) -> Path:
    return Path(__file__).resolve().parents[1] / "results" / name


def git_fixture_env():
    env = os.environ.copy()
    for key in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_COMMON_DIR"):
        env.pop(key, None)
    env.update({
        "GIT_AUTHOR_NAME": "Fixture", "GIT_COMMITTER_NAME": "Fixture",
        "GIT_AUTHOR_EMAIL": "fixture@example.invalid", "GIT_COMMITTER_EMAIL": "fixture@example.invalid",
        "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull,
    })
    return env


@pytest.fixture(scope="module")
def bundled_pair(tmp_path_factory):
    import hashlib
    import shutil

    root = tmp_path_factory.mktemp("bundled-kimi-pair")
    names = (
        "persona_conflict_conversations_v1",
        "persona_conflict_conversations_surface_a",
        "persona_conflict_conversations_surface_b",
    )
    for name in names:
        shutil.copytree(paper_input_dir(name), root / name)
    gate = root / "persona_surface_pair_gate.json"
    gate.write_bytes(paper_input_dir("persona_surface_pair_gate.json").read_bytes())
    from experiments.paper_persona import PAIR_GATE_SHA256
    assert hashlib.sha256(gate.read_bytes()).hexdigest() == PAIR_GATE_SHA256
    return *(root / name for name in names), gate


def pytest_collection_modifyitems(items):
    for item in items:
        if item.get_closest_marker("posix_directory_fsync") and os.name != "posix":
            item.add_marker(pytest.mark.skip(reason="generation durability requires POSIX directory-descriptor fsync"))


@pytest.fixture
def archive_source_environment(tmp_path, monkeypatch):
    from neurosym.application.source_provenance import write_source_manifest

    root = Path(__file__).resolve().parents[1]
    manifest = tmp_path / "test-source-manifest.json"
    digest = write_source_manifest(root, manifest)
    monkeypatch.setenv("NEUROSYM_SOURCE_MANIFEST", str(manifest))
    monkeypatch.setenv("NEUROSYM_SOURCE_MANIFEST_SHA256", digest)
    return manifest, digest


@pytest.fixture
def original_root():
    value = os.environ.get("NEUROSYM_ORIGINAL_ROOT")
    if not value:
        pytest.skip("set NEUROSYM_ORIGINAL_ROOT to a separately provided original checkout")
    root = Path(value).resolve(strict=True)
    assert root != Path(__file__).resolve().parents[1]
    return root


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
