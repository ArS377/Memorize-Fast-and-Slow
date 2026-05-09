"""Flat (single-call) LLM answerer for cells 1-3.

Reuses the ``openai.OpenAI`` client pattern from ``LongBenchKGPipeline.chat_json``.
"""

from __future__ import annotations

from typing import Tuple

from experiments.common import extract_letter


def flat_answer(
    client,
    model: str,
    context: str,
    question: str,
    max_tokens: int = 2048,
    temperature: float = 0.0,
    enable_thinking: bool = True,
) -> Tuple[str, str]:
    """Single-shot multiple-choice answer over ``context``.

    Returns ``(letter, raw_response)``.

    Defaults to Qwen3 thinking-mode ON with a 2048-token budget so the model
    can produce a <think>...</think> block followed by the final letter.
    extract_letter() handles stripping the think block and grabbing the
    trailing letter.
    """
    messages = [
        {"role": "system", "content": "Answer with only the letter A, B, C, or D."},
        {"role": "user", "content": f"Context:\n{context}\n\n{question}"},
    ]
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body={"chat_template_kwargs": {"enable_thinking": enable_thinking}},
    )
    raw = resp.choices[0].message.content or ""
    return extract_letter(raw), raw
