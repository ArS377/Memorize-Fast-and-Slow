"""RLM (recursive language model) answerer for cells 4-6.

Mirrors the per-example RLM instantiation pattern used by
``rlm_baseline.py`` and ``rlm_graph_baseline.py``: each example gets a
fresh RLM (stateless) over the supplied prompt and root_prompt.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

from neurosym.application.experiment_io import extract_letter


# Qwen is served with a 40,960-token context window in the experiment setup.
# The upstream RLM prompt advertises much larger sub-call windows, so override
# that advice when Cell 4 exposes the complete document through the REPL. This
# is a per-sub-call bound, not a document cap: every source character remains
# available in ``context`` and the RLM can scan ordered chunks or batch them.
FULL_CONTEXT_SUBQUERY_CHARS = 32_000
FULL_CONTEXT_COMPACTION_THRESHOLD_PCT = 0.20
FULL_CONTEXT_SYSTEM_SUFFIX = f"""

IMPORTANT RUNTIME CONSTRAINT FOR THIS EXPERIMENT:
The underlying Qwen server has a 40,960-token model context window. The complete
source document is available only through the REPL variable `context`; it is not
part of this root prompt. Never send the complete `context` value to
`llm_query`, `llm_query_batched`, `rlm_query`, or `rlm_query_batched`. Keep the
document portion of every individual sub-query at or below
{FULL_CONTEXT_SUBQUERY_CHARS:,} characters. Scan longer documents as ordered
chunks with modest overlap (batched where useful), retain evidence with its
chunk position, and combine only the compact evidence. This instruction
supersedes any larger sub-LLM context estimate elsewhere in this prompt.
"""


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
    return RLM(
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
