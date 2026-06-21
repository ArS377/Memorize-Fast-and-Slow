import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.rlm_answerer import rlm_answer


class FailingRLM:
    def completion(self, **kwargs):
        raise RuntimeError("Bad Request: connection closed")


class SuccessfulRLM:
    def completion(self, **kwargs):
        return type("Completion", (), {"response": "Final answer: C"})()


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
