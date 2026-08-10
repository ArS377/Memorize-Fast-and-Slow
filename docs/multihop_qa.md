# Multi-hop QA track — HotpotQA & 2WikiMultihopQA (PPR verification)

## Why

The dense-seeded PPR sweep on LongBench-v2 was inconclusive by construction:
LongBench-v2 is single-passage multiple-choice with **no evidence labels**, so
Recall@k and multi-hop coverage were uncomputable and the per-example graph was
too small (~10 facts) for PPR to walk (see
[`results/hippo_overnight/dense_ppr_sweep/ANALYSIS.md`](../results/hippo_overnight/dense_ppr_sweep/ANALYSIS.md)).

**2WikiMultihopQA** fixes both problems and is the primary bed for verifying
PPR: multi-document context (so bridge facts are dense-*invisible* yet
structurally near), gold sentence-level `supporting_facts`, **and** gold
`(subject, relation, object)` evidence triples that line up with the KG fact
schema. **HotpotQA** is the second bed (sentence-level supporting facts only).

This track is **controller-independent**: retrieval eval and its PPR metrics sit
entirely upstream of the RLM. It does not depend on the RLM `FINAL()`-protocol
fix that is currently pending (see the internal `rlm-final-protocol-mismatch`
note).

## Files

| File | Purpose |
|---|---|
| [`experiments/multihop_datasets.py`](../experiments/multihop_datasets.py) | Normalize either dataset into the shared example schema (robust to HF parallel-list and release list-of-pairs encodings). |
| [`download_2wiki.py`](../download_2wiki.py) | `--json <release>.json` or `--dataset <hf id>` → `data_2wiki.jsonl`. |
| [`download_hotpotqa.py`](../download_hotpotqa.py) | `hotpotqa/hotpot_qa` (`distractor`) → `data_hotpotqa.jsonl`. |
| [`experiments/gold_relevance.py`](../experiments/gold_relevance.py) | Map gold labels → fact IDs on a `retrieval_eval.jsonl`; report Recall@k **and multi-hop coverage** per mode. |
| [`experiments/answer_eval.py`](../experiments/answer_eval.py) | Free-form EM / token-F1 scoring (SQuAD-style). |
| [`experiments/multihop_answerer.py`](../experiments/multihop_answerer.py) | Controller-independent single-call free-form answerer. |

## Normalized schema

```jsonc
{
  "_id": "...", "dataset": "2wikimultihopqa",
  "question": "...", "answer": "Anurag Kashyap",     // free-form gold
  "context": [["Title", ["sent 0", "sent 1"]], ...], // STRUCTURED — titles survive into fact provenance
  "gold_supporting_facts": [["Title", 0], ...],       // (title, local sentence index)
  "gold_evidence_triples": [["subj","rel","obj"], ...] // 2Wiki only; [] for HotpotQA
}
```

Keeping `context` structured is load-bearing: the KG pipeline's
`flatten_context_to_sentence_records` preserves each document `title` and
`local_sent_id` into every fact's provenance, which is how gold
`(title, sent_id)` labels get mapped back to extracted `fact_id`s.

## Recipe (2Wiki first)

Run download + KG build in the env that has `datasets`/vLLM
(`.venv-vllm-metal`); the label/scoring steps run anywhere (`.venv`).

```bash
# 1. Data. Point --json at an official release file (dev.json), or use --dataset for an HF mirror.
python download_2wiki.py --json /path/to/2wiki/dev.json --limit 200 --out data_2wiki.jsonl

# 2. Build a per-example KG (example scope keeps each question's docs isolated).
python -m experiments.build_kg_facts_file \
    --session twowiki_dev --input data_2wiki.jsonl \
    --model Qwen/Qwen3-4B --vllm-base-url http://localhost:8000/v1 \
    --facts-out-dir results/kg_builds

# 3. Retrieval eval across modes (produces retrieval_eval.jsonl per mode).
#    Use the existing dense_ppr ablation / run_all with --retrieval-mode {sparse,dense,hybrid,dense_ppr}.

# 4. Attach gold labels + score Recall@k and multi-hop coverage (THE PPR verification).
python -m experiments.gold_relevance \
    --retrieval-eval results/<run>/retrieval_eval.jsonl \
    --facts results/kg_builds/twowiki_dev_facts.jsonl \
    --dataset data_2wiki.jsonl \
    --out results/<run>/retrieval_eval.labeled.jsonl \
    --report results/<run>/gold_relevance_report.json

# 5. (Optional) produce + score free-form answers (does NOT use the RLM).
python -m experiments.answer_eval --predictions <preds>.jsonl --dataset data_2wiki.jsonl
```

## What "verify PPR" means here

The decisive comparison is in the `gold_relevance` report:

- **Recall@5 / @10** per mode — does `dense_ppr` retrieve more gold facts than `dense`/`hybrid`?
- **`multi_hop_covered_at_k`** — fraction of questions whose top-k surfaces a
  relevant fact from **every** gold supporting document. This is the metric PPR
  is meant to move: dense scoring ranks facts by query similarity and misses the
  bridge document; PPR walks the entity↔fact graph to reach it. If PPR helps
  anywhere, it is here.

HotpotQA is the same recipe with `download_hotpotqa.py`; its report uses only
the `gold_supporting_facts` source (no evidence triples).
