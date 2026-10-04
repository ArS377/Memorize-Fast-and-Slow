"""Tests for the free-form multi-hop answerer helpers."""
from __future__ import annotations

from types import SimpleNamespace

from experiments.multihop_answerer import (
    extract_span,
    free_form_answer,
    render_facts_as_context,
    render_passages_as_context,
)


def test_extract_span_strips_think_and_prefix():
    raw = "<think>reasoning here</think>\nThe answer is: Anurag Kashyap"
    assert extract_span(raw) == "Anurag Kashyap"
    assert extract_span("yes") == "yes"
    assert extract_span('"Paris"') == "Paris"


def test_render_facts_as_context_includes_ids():
    facts = [
        {"subject": "Kingdom of Dreams", "predicate": "director", "object": "V. Motwane", "fact_id": "f1"},
    ]
    rendered = render_facts_as_context(facts)
    assert "Kingdom of Dreams director V. Motwane [f1]" in rendered


def test_render_passages_flattens_structured_context():
    ctx = [["Doc A", ["S1.", "S2."]], ["Doc B", ["S3."]]]
    rendered = render_passages_as_context(ctx)
    assert "Doc A: S1. S2." in rendered
    assert "Doc B: S3." in rendered


class _FakeClient:
    def __init__(self, content):
        self._content = content
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        msg = SimpleNamespace(content=self._content)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


def test_free_form_answer_returns_clean_span():
    client = _FakeClient("<think>...</think>\nAnurag Kashyap")
    answer, raw = free_form_answer(client, "Qwen", "some context", "Who?")
    assert answer == "Anurag Kashyap"
    assert "Anurag Kashyap" in raw
