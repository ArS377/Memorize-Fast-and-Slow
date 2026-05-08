"""RLM (recursive language model) answerer for cells 4-6.

Mirrors the per-example RLM instantiation pattern used by
``rlm_baseline.py`` and ``rlm_graph_baseline.py``: each example gets a
fresh RLM (stateless) over the supplied prompt and root_prompt.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

from experiments.common import extract_letter


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
):
    """Build a fresh RLM. Imported lazily so callers without ``rlm`` installed
    can still import this module (e.g. for type checking).
    """
    from rlm.core.rlm import RLM
    from rlm.logger.rlm_logger import RLMLogger

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = RLMLogger(log_dir=str(log_dir))
    return RLM(
        backend=backend,
        backend_kwargs={"model_name": model, "base_url": base_url, "api_key": api_key},
        environment="local",
        max_depth=max_depth,
        max_iterations=max_iterations,
        max_tokens=max_tokens,
        logger=logger,
        verbose=verbose,
    )


def rlm_answer(
    rlm,
    context: str,
    question: str,
) -> Tuple[str, str, Optional[str]]:
    """Run RLM completion. Returns ``(letter, raw_answer, error_or_none)``."""
    try:
        result = rlm.completion(prompt=context, root_prompt=question)
        raw = str(result) if result else ""
        return extract_letter(raw), raw, None
    except Exception as e:
        msg = str(e)
        return extract_letter(msg), msg, msg
