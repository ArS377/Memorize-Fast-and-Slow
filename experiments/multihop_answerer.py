"""Free-form answerer for HotpotQA / 2WikiMultihopQA.

The pipeline's ``flat_answer`` is hard-wired to multiple choice ("Answer with
only the letter A, B, C, or D"). These datasets need a short span answer, so
this module provides a free-form single-call answerer that reuses the same
OpenAI/vLLM client pattern. It is deliberately controller-independent — it does
NOT go through the RLM native-tool loop — so answers can be produced and scored
(``experiments.answer_eval``) while the RLM ``FINAL()`` protocol fix is pending
(see the ``rlm-final-protocol-mismatch`` note).

``context`` can be raw passages or a rendering of retrieved KG facts; callers
that want to measure retrieval quality should pass the retrieved facts.
"""
from __future__ import annotations

import re
from typing import Sequence, Tuple


_SYSTEM = (
    "You answer multi-hop questions with a short, exact span — a name, entity, "
    "date, or yes/no. Reply with ONLY the answer text, no explanation."
)
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_PREFIX = re.compile(r"^(the answer is|final answer|answer)\s*:?\s*", re.IGNORECASE)


def extract_span(raw: str) -> str:
    """Pull a clean short answer out of the model output."""
    text = _THINK.sub("", raw or "").strip()
    # Prefer the last non-empty line (the model's final answer typically ends here).
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    candidate = lines[-1] if lines else text
    candidate = _PREFIX.sub("", candidate).strip().strip('"').strip()
    return candidate


def free_form_answer(
    client,
    model: str,
    context: str,
    question: str,
    max_tokens: int = 512,
    temperature: float = 0.0,
    enable_thinking: bool = True,
) -> Tuple[str, str]:
    """Single-shot free-form answer over ``context``. Returns ``(answer, raw)``."""
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"},
    ]
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body={"chat_template_kwargs": {"enable_thinking": enable_thinking}},
    )
    raw = resp.choices[0].message.content or ""
    return extract_span(raw), raw


def render_facts_as_context(facts: Sequence[dict], max_facts: int = 30) -> str:
    """Render retrieved KG facts as a compact context block for answering."""
    lines = []
    for fact in list(facts)[:max_facts]:
        subj = str(fact.get("subject", "")).strip()
        pred = str(fact.get("predicate", "")).strip()
        obj = str(fact.get("object", "")).strip()
        fact_id = str(fact.get("fact_id", "")).strip()
        tag = f" [{fact_id}]" if fact_id else ""
        lines.append(f"- {subj} {pred} {obj}{tag}")
    return "\n".join(lines)


def render_passages_as_context(context_blocks: Sequence, max_chars: int = 12000) -> str:
    """Render structured ``[[title, [sentences]], ...]`` context as plain text."""
    parts = []
    for block in context_blocks:
        if isinstance(block, (list, tuple)) and len(block) == 2:
            title, sentences = block
            body = " ".join(str(s) for s in sentences)
            parts.append(f"{title}: {body}")
    text = "\n".join(parts)
    return text[:max_chars]
