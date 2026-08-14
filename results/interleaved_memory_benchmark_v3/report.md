# Delayed Interleaved Memory Benchmark

- Benchmark version: `interleaved_delayed_retrieval.v3`
- Eligible queries: 2,048
- Evaluated queries: 2,047
- Excluded queries: 1 (`history-599-preference-incongruity-delayed`)
- Median oldest-required-evidence age: 137,011 tokens
- Checkpoints with an oldest required item beyond 64K: 78.94%
- Checkpoints with an oldest required item beyond 128K: 52.22%

## Matched Methods

Both methods use the same causal checkpoint, original dataset query text without augmentation, exact tokenizer, and prompt-inclusive token cap.
The structured method selects one capsule globally with BM25 and receives no history filter, gold answer, or evidence contract during ranking.
Validity admission uses Scallop-derived relation kind before query-blind, history-blind, stable-hash capacity compaction within admitted capsules.
Answers are scored with a deterministic oracle resolver, so results measure context availability rather than LLM reasoning.

- sliding_context:131072: grounded 47.78%; oracle answer 48.80%; complete provenance 47.78%; stale intrusion 0.00%
- sliding_context:65536: grounded 20.86%; oracle answer 22.57%; complete provenance 20.86%; stale intrusion 0.00%
- structured_capacity_top1:131072: grounded 72.35%; oracle answer 72.84%; complete provenance 72.35%; stale intrusion 19.35%
- structured_capacity_top1:65536: grounded 58.18%; oracle answer 58.77%; complete provenance 58.18%; stale intrusion 29.07%

## Paired Grounded Comparison

- 65,536 tokens: structured minus sliding is 37.30%, with deterministic history-clustered 95% interval [35.94%, 38.77%].
- 131,072 tokens: structured minus sliding is 24.56%, with deterministic history-clustered 95% interval [23.10%, 26.07%].

Each interval uses 2,000 bootstrap samples over 1,024 history clusters with seed 47.

## Family-Stratified Grounded Availability

### preference-change-delayed

- sliding_context:131072: grounded 46.29%; stale intrusion 0.00%.
- sliding_context:65536: grounded 19.34%; stale intrusion 0.00%.
- structured_capacity_top1:131072: grounded 87.21%; stale intrusion 0.00%.
- structured_capacity_top1:65536: grounded 80.27%; stale intrusion 0.00%.

### preference-incongruity-delayed

- sliding_context:131072: grounded 49.27%; stale intrusion 0.00%.
- sliding_context:65536: grounded 22.39%; stale intrusion 0.00%.
- structured_capacity_top1:131072: grounded 57.48%; stale intrusion 38.71%.
- structured_capacity_top1:65536: grounded 36.07%; stale intrusion 58.16%.

## Selector and Error Diagnostics

- Target-history selection rate: 92.13%.
- Expected-family selection rate: 49.98%.
- Target-history and expected-family selection rate: 46.07%.
- Stale among structured 64K errors: 69.51%.
- Stale among structured 128K errors: 69.96%.

The supported claim is bounded deterministic context availability under this synthetic protocol.
This is not LLM answer accuracy, extraction quality, independent reasoning, deployment safety, or a general memory-system superiority claim.
