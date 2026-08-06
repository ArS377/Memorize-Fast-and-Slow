import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.rlm_answerer import (
    FULL_CONTEXT_BATCH_CHARS,
    FULL_CONTEXT_COMPACTION_THRESHOLD_PCT,
    FULL_CONTEXT_RESPONSE_CHARS,
    FULL_CONTEXT_SUBQUERY_CHARS,
    FULL_CONTEXT_TOTAL_TOKEN_BUDGET,
    make_rlm,
    rlm_answer,
)
from neurosym.adapters.rlm import _install_full_context_guards


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


class FakeEnvironment:
    def __init__(self):
        self.calls = []
        self.globals = {}

    def _single(self, kind, prompt, model=None):
        self.calls.append((kind, prompt, model))
        return "x" * (FULL_CONTEXT_RESPONSE_CHARS + 100)

    def _batch(self, kind, prompts, model=None):
        self.calls.append((kind, list(prompts), model))
        return ["x" * (FULL_CONTEXT_RESPONSE_CHARS + 100) for _ in prompts]

    def _llm_query(self, prompt, model=None):
        return self._single("llm", prompt, model)

    def _rlm_query(self, prompt, model=None):
        return self._single("rlm", prompt, model)

    def _llm_query_batched(self, prompts, model=None):
        return self._batch("llm_batch", prompts, model)

    def _rlm_query_batched(self, prompts, model=None):
        return self._batch("rlm_batch", prompts, model)


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
        max_tokens=FULL_CONTEXT_TOTAL_TOKEN_BUDGET,
        log_dir=tmp_path,
        full_context=True,
    )

    assert rlm.compaction is True
    assert rlm.compaction_threshold_pct == FULL_CONTEXT_COMPACTION_THRESHOLD_PCT
    assert rlm.max_tokens == FULL_CONTEXT_TOTAL_TOKEN_BUDGET
    assert f"{FULL_CONTEXT_SUBQUERY_CHARS:,} characters" in rlm.system_prompt
    assert f"{FULL_CONTEXT_BATCH_CHARS:,} prompt characters" in rlm.system_prompt
    assert "Never send the complete `context` value" in rlm.system_prompt

    context = "BEGIN_SENTINEL" + ("x" * 100_000) + "END_SENTINEL"
    root_messages = rlm._setup_prompt(context)
    root_text = "\n".join(message["content"] for message in root_messages)
    assert "BEGIN_SENTINEL" not in root_text
    assert "END_SENTINEL" not in root_text
    assert str(len(context)) in root_text
    assert len(root_text) < 20_000


def test_full_context_guard_rejects_large_calls_and_bounds_responses() -> None:
    environment = FakeEnvironment()
    _install_full_context_guards(environment)

    response = environment.globals["llm_query"]("a" * FULL_CONTEXT_SUBQUERY_CHARS)
    assert len(response) < FULL_CONTEXT_RESPONSE_CHARS + 100
    assert "truncated" in response
    assert len(environment.calls) == 1

    rejected = environment.globals["rlm_query"](
        "a" * (FULL_CONTEXT_SUBQUERY_CHARS + 1)
    )
    assert rejected.startswith("Error: sub-query rejected")
    assert len(environment.calls) == 1


def test_full_context_guard_preserves_batch_shape_and_caps_total_chars() -> None:
    environment = FakeEnvironment()
    _install_full_context_guards(environment)
    prompts = [
        "a" * FULL_CONTEXT_SUBQUERY_CHARS,
        "b" * FULL_CONTEXT_SUBQUERY_CHARS,
        "c",
        "d" * (FULL_CONTEXT_SUBQUERY_CHARS + 1),
    ]

    responses = environment.globals["llm_query_batched"](prompts)

    assert len(responses) == len(prompts)
    assert "truncated" in responses[0]
    assert "truncated" in responses[1]
    assert responses[2].startswith("Error: batched sub-query rejected")
    assert responses[3].startswith("Error: sub-query rejected")
    assert environment.calls == [("llm_batch", prompts[:2], None)]


def test_full_context_guard_is_installed_in_real_local_environment(tmp_path: Path) -> None:
    rlm = make_rlm(
        backend="openai",
        model="Qwen/Qwen3-4B",
        base_url="http://localhost:8000/v1",
        api_key="EMPTY",
        max_depth=2,
        max_iterations=10,
        max_tokens=FULL_CONTEXT_TOTAL_TOKEN_BUDGET,
        log_dir=tmp_path,
        full_context=True,
    )

    with rlm._spawn_completion_context("external context") as (_handler, environment):
        rejected = environment.globals["llm_query"](
            "x" * (FULL_CONTEXT_SUBQUERY_CHARS + 1)
        )

    assert rejected.startswith("Error: sub-query rejected")


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
