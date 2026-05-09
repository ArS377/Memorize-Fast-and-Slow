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
    max_tokens: int = 64,
    temperature: float = 0.0,
) -> Tuple[str, str]:
    """Single-shot multiple-choice answer over ``context``.

    Returns ``(letter, raw_response)``.
    """
    messages = [
        {"role": "system", "content": "Answer with only the letter A, B, C, or D."},
        {"role": "user", "content": f"Context:\n{context}\n\n{question}"},
    ]
    # Qwen3 enables <think> reasoning by default, which eats the entire
    # max_tokens budget before emitting the answer letter. For a single-token
    # multiple-choice answer we don't want hidden reasoning at all -- disable
    # via chat_template_kwargs (vLLM passes this through to the chat template).
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    raw = resp.choices[0].message.content or ""
    return extract_letter(raw), raw
