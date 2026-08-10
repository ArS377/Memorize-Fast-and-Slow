"""RLM (recursive language model) answerer for cells 4-6.

Mirrors the per-example RLM instantiation pattern used by
``rlm_baseline.py`` and ``rlm_graph_baseline.py``: each example gets a
fresh RLM (stateless) over the supplied prompt and root_prompt.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Tuple

from neurosym.application.experiment_io import extract_letter


# Qwen is served with a 40,960-token context window in the experiment setup.
# The upstream RLM prompt advertises much larger sub-call windows, so override
# that advice when Cell 4 exposes the complete document through the REPL. This
# is a per-sub-call bound, not a document cap: every source character remains
# available in ``context`` and the RLM can scan ordered chunks or batch them.
FULL_CONTEXT_SUBQUERY_CHARS = 32_000
FULL_CONTEXT_BATCH_CHARS = 64_000
FULL_CONTEXT_RESPONSE_CHARS = 16_000
FULL_CONTEXT_TOTAL_TOKEN_BUDGET = 128_000
FULL_CONTEXT_COMPACTION_THRESHOLD_PCT = 0.20
FULL_CONTEXT_SYSTEM_SUFFIX = f"""

IMPORTANT RUNTIME CONSTRAINT FOR THIS EXPERIMENT:
The underlying Qwen server has a 40,960-token model context window. The complete
source document is available only through the REPL variable `context`; it is not
part of this root prompt. Never send the complete `context` value to
`llm_query`, `llm_query_batched`, `rlm_query`, or `rlm_query_batched`. Keep the
document portion of every individual sub-query at or below
{FULL_CONTEXT_SUBQUERY_CHARS:,} characters. Scan longer documents as ordered
chunks with modest overlap, retain evidence with its chunk position, and
combine only the compact evidence. Batched calls may contain at most
{FULL_CONTEXT_BATCH_CHARS:,} prompt characters in total. These limits are
enforced by the runtime; rejected calls return an explanatory error so you can
retry with smaller slices. Sub-query responses longer than
{FULL_CONTEXT_RESPONSE_CHARS:,} characters are truncated before entering the
REPL. This instruction supersedes any larger sub-LLM context estimate elsewhere
in this prompt.

For this multiple-choice experiment, submit exactly one of A, B, C, or D in
`answer["content"]`; never submit an option number or the option text.
"""


def _rejected_subquery(length: int) -> str:
    return (
        "Error: sub-query rejected by the experiment runtime: "
        f"{length:,} characters exceeds the {FULL_CONTEXT_SUBQUERY_CHARS:,}-character "
        "limit. Slice the document into smaller ordered chunks and retry."
    )


def _rejected_batch(length: int) -> str:
    return (
        "Error: batched sub-query rejected by the experiment runtime: adding this "
        f"prompt would raise the batch to {length:,} characters, above the "
        f"{FULL_CONTEXT_BATCH_CHARS:,}-character aggregate limit. Use a smaller batch."
    )


def _bounded_response(value: Any) -> Any:
    if not isinstance(value, str) or len(value) <= FULL_CONTEXT_RESPONSE_CHARS:
        return value
    omitted = len(value) - FULL_CONTEXT_RESPONSE_CHARS
    return (
        value[:FULL_CONTEXT_RESPONSE_CHARS]
        + f"\n[experiment runtime truncated {omitted:,} response characters]"
    )


def _install_full_context_guards(environment: Any) -> None:
    """Hard-bound Cell 4 sub-calls without modifying the external ``rlms`` package."""
    if getattr(environment, "_neurosym_full_context_guards", False):
        return

    original_single: dict[str, Callable[..., Any]] = {
        name: getattr(environment, name)
        for name in ("_llm_query", "_rlm_query")
    }
    original_batch: dict[str, Callable[..., Any]] = {
        name: getattr(environment, name)
        for name in ("_llm_query_batched", "_rlm_query_batched")
    }

    def guard_single(original: Callable[..., Any]) -> Callable[..., Any]:
        def bounded(prompt: str, model: str | None = None) -> Any:
            if not isinstance(prompt, str):
                return "Error: sub-query prompt must be a string."
            if len(prompt) > FULL_CONTEXT_SUBQUERY_CHARS:
                return _rejected_subquery(len(prompt))
            return _bounded_response(original(prompt, model))

        return bounded

    def guard_batch(original: Callable[..., Any]) -> Callable[..., Any]:
        def bounded(prompts: Sequence[str], model: str | None = None) -> list[Any]:
            if not isinstance(prompts, (list, tuple)):
                return ["Error: batched sub-query prompts must be a list or tuple."]

            results: list[Any] = [None] * len(prompts)
            accepted: list[str] = []
            accepted_indices: list[int] = []
            batch_chars = 0
            for index, prompt in enumerate(prompts):
                if not isinstance(prompt, str):
                    results[index] = "Error: sub-query prompt must be a string."
                    continue
                if len(prompt) > FULL_CONTEXT_SUBQUERY_CHARS:
                    results[index] = _rejected_subquery(len(prompt))
                    continue
                proposed_chars = batch_chars + len(prompt)
                if proposed_chars > FULL_CONTEXT_BATCH_CHARS:
                    results[index] = _rejected_batch(proposed_chars)
                    continue
                accepted.append(prompt)
                accepted_indices.append(index)
                batch_chars = proposed_chars

            if accepted:
                accepted_results = original(accepted, model)
                for index, value in zip(accepted_indices, accepted_results, strict=True):
                    results[index] = _bounded_response(value)
            return results

        return bounded

    environment._llm_query = guard_single(original_single["_llm_query"])
    environment._rlm_query = guard_single(original_single["_rlm_query"])
    environment._llm_query_batched = guard_batch(original_batch["_llm_query_batched"])
    environment._rlm_query_batched = guard_batch(original_batch["_rlm_query_batched"])
    environment._neurosym_full_context_guards = True

    # LocalREPL captures bound callables in its globals during setup. Replace
    # those references too; its scaffold restoration will preserve these
    # instance-level wrappers after subsequent code executions.
    if hasattr(environment, "globals"):
        environment.globals.update(
            llm_query=environment._llm_query,
            llm_query_batched=environment._llm_query_batched,
            rlm_query=environment._rlm_query,
            rlm_query_batched=environment._rlm_query_batched,
        )


def _bounded_rlm_class(base_rlm: type) -> type:
    class BoundedFullContextRLM(base_rlm):
        @contextmanager
        def _spawn_completion_context(self, prompt):
            with super()._spawn_completion_context(prompt) as resources:
                _lm_handler, environment = resources
                _install_full_context_guards(environment)
                yield resources

    BoundedFullContextRLM.__name__ = "BoundedFullContextRLM"
    return BoundedFullContextRLM


def make_rlm(
    *,
    backend: str,
    model: str,
    base_url: str,
    api_key: str,
    max_depth: int,
    max_iterations: int,
    max_tokens: int,
    log_dir: Path,
    verbose: bool = False,
    full_context: bool = False,
):
    """Build a fresh RLM. Imported lazily so callers without ``rlm`` installed
    can still import this module (e.g. for type checking).
    """
    from rlm.core.rlm import RLM
    from rlm.logger.rlm_logger import RLMLogger
    from rlm.utils.prompts import RLM_SYSTEM_PROMPT

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = RLMLogger(log_dir=str(log_dir))
    rlm_kwargs = {}
    if full_context:
        rlm_kwargs.update(
            custom_system_prompt=RLM_SYSTEM_PROMPT + FULL_CONTEXT_SYSTEM_SUFFIX,
            # The RLM package assumes a 128k Qwen window when compacting. A
            # 20% threshold compacts around 25.6k estimated tokens, safely
            # before this experiment server's 40,960-token limit.
            compaction=True,
            compaction_threshold_pct=FULL_CONTEXT_COMPACTION_THRESHOLD_PCT,
        )
    rlm_class = _bounded_rlm_class(RLM) if full_context else RLM
    return rlm_class(
        backend=backend,
        backend_kwargs={"model_name": model, "base_url": base_url, "api_key": api_key},
        environment="local",
        max_depth=max_depth,
        max_iterations=max_iterations,
        max_tokens=max_tokens,
        logger=logger,
        verbose=verbose,
        **rlm_kwargs,
    )


def rlm_answer(
    rlm,
    context: str,
    question: str,
) -> Tuple[str, str, Optional[str]]:
    """Run RLM completion. Returns ``(letter, raw_answer, error_or_none)``."""
    try:
        result = rlm.completion(prompt=context, root_prompt=question)
        # rlms 0.1.x returns an RLMChatCompletion whose .response holds the
        # final answer text. Falling back to str(result) would emit the repr
        # (class name + dict), which never contains A/B/C/D and so was
        # producing pred='' on otherwise-successful completions.
        if result is None:
            raw = ""
        else:
            raw = getattr(result, "response", None) or str(result)
        return extract_letter(raw), raw, None
    except Exception as e:
        msg = str(e)
        # Do NOT run extract_letter on an error string: many error messages
        # contain stray A/B/C/D characters and would silently fabricate a
        # prediction (this is what caused cell 4 to collapse to pred='D').
        return "", msg, msg
