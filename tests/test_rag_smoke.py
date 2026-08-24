#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import types
import unittest.mock as mock
from pathlib import Path

if "openai" not in sys.modules:
    fake_openai = types.ModuleType("openai")
    fake_openai.OpenAI = mock.MagicMock
    sys.modules["openai"] = fake_openai

sys.path.insert(0, str(Path(__file__).parent.parent))
import evaluator  # noqa: E402


def _mock_client(answer: str):
    client = mock.MagicMock()
    choice = mock.MagicMock()
    choice.message.content = answer
    resp = mock.MagicMock()
    resp.choices = [choice]
    client.chat.completions.create.return_value = resp
    return client


def main() -> int:
    temp_data = Path("rag_smoke_data.jsonl")
    temp_out = Path("rag_smoke_outputs")
    temp_data.write_text(
        json.dumps(
            {
                "_id": "e1",
                "question": "Where does Kalamang language exist?",
                "answer": "East Indonesia",
                "context": "Kalamang is spoken in East Indonesia by around 130 people.",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    cfg = evaluator.RagConfig(
        dataset_path=temp_data,
        dataset_name=None,
        split="train",
        max_examples=1,
        output_dir=temp_out,
        output_name="smoke",
        model="mock-model",
        vllm_base_url="http://localhost:8000/v1",
        api_key="EMPTY",
        chunk_chars=500,
        top_k=1,
        temperature=0.0,
        max_tokens=64,
    )

    with mock.patch("eval.vanilla_rag.OpenAI", return_value=_mock_client("East Indonesia")):
        pred_path = evaluator.run_rag(cfg)

    assert pred_path.exists(), "prediction file missing"
    rows = [json.loads(x) for x in pred_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    assert len(rows) == 1, "expected one output row"
    assert rows[0]["prediction"] == "East Indonesia", "wrong prediction"
    assert rows[0]["retrieved_chunks"], "retrieval output missing"
    print("RAG smoke test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

