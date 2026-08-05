import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.rlm_answerer import (
    FULL_CONTEXT_COMPACTION_THRESHOLD_PCT,
    FULL_CONTEXT_SUBQUERY_CHARS,
    make_rlm,
    rlm_answer,
)


class FailingRLM:
    def completion(self, **kwargs):
        raise RuntimeError("Bad Request: connection closed")


class SuccessfulRLM:
    def completion(self, **kwargs):
        return type("Completion", (), {"response": "Final answer: C"})()


class RecordingRLM:
    def __init__(self):
        self.kwargs = None

    def completion(self, **kwargs):
        self.kwargs = kwargs
        return type("Completion", (), {"response": "Final answer: A"})()


def test_rlm_answer_does_not_parse_error_messages() -> None:
    predicted, raw, error = rlm_answer(FailingRLM(), "context", "question")

    assert predicted == ""
    assert raw == "Bad Request: connection closed"
    assert error == raw


def test_rlm_answer_extracts_successful_completion_response() -> None:
    predicted, raw, error = rlm_answer(SuccessfulRLM(), "context", "question")

    assert predicted == "C"
    assert raw == "Final answer: C"
    assert error is None


def test_rlm_answer_passes_full_context_as_external_prompt() -> None:
    context = "start-marker" + ("x" * 100_000) + "end-marker"
    rlm = RecordingRLM()

    predicted, _raw, error = rlm_answer(rlm, context, "question")

    assert predicted == "A"
    assert error is None
    assert rlm.kwargs == {"prompt": context, "root_prompt": "question"}


def test_full_context_rlm_enables_early_compaction_and_safe_chunk_advice(
    tmp_path: Path,
) -> None:
    rlm = make_rlm(
        backend="openai",
        model="Qwen/Qwen3-4B",
        base_url="http://localhost:8000/v1",
        api_key="EMPTY",
        max_depth=2,
        max_iterations=10,
        max_tokens=64_000,
        log_dir=tmp_path,
        full_context=True,
    )

    assert rlm.compaction is True
    assert rlm.compaction_threshold_pct == FULL_CONTEXT_COMPACTION_THRESHOLD_PCT
    assert f"{FULL_CONTEXT_SUBQUERY_CHARS:,} characters" in rlm.system_prompt
    assert "Never send the complete `context` value" in rlm.system_prompt

    context = "BEGIN_SENTINEL" + ("x" * 100_000) + "END_SENTINEL"
    root_messages = rlm._setup_prompt(context)
    root_text = "\n".join(message["content"] for message in root_messages)
    assert "BEGIN_SENTINEL" not in root_text
    assert "END_SENTINEL" not in root_text
    assert str(len(context)) in root_text
    assert len(root_text) < 20_000


def main() -> bool:
    tests = [
        test_rlm_answer_does_not_parse_error_messages,
        test_rlm_answer_extracts_successful_completion_response,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"\nResults: {len(tests)}/{len(tests)} tests passed")
    return True


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
