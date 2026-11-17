# LongMemEval-S: NeuroSym Versus Graphiti on Shared Extracted Facts (500 Histories)

This evaluation compares NeuroSym hybrid KG memory with Graphiti on all 500 LongMemEval-S histories. Both systems ingest the same extracted facts, and each answer is scored by a judge. It extends the [250-history result](../longmemeval_s_shared_facts_250_20261114/README.md), whose per-history labels are included unchanged.

## Exact accuracy

| Arm | Correct | Accuracy | 95% Wilson CI |
|---|---:|---:|---:|
| NeuroSym hybrid KG, shared facts | 251/500 | 50.2% | 45.8–54.6% |
| Graphiti, shared facts | 241/500 | 48.2% | 43.9–52.6% |

| Paired outcome | Histories |
|---|---:|
| Both correct | 207 |
| NeuroSym only correct | 44 |
| Graphiti only correct | 34 |
| Both wrong | 215 |

The paired NeuroSym-minus-Graphiti difference is +2.0 percentage points, 95% paired bootstrap CI [−1.4, +5.6], exact McNemar p=0.31. No difference was detected between the two arms. The interval is consistent with differences from about −1 to about +6 points; it is not an equivalence test. Excluding abstention questions gives +2.6 points, 95% CI [−1.1, +6.2] (n=470).

## Scope

The 500 histories are all LongMemEval-S questions, one question per history, so this result has the benchmark's own question-type mix. The earlier 250 were evaluated first; the later 250 were evaluated afterwards with the same models, prompts and settings.
Both arms ingest the same extracted facts. The Graphiti arm does not perform its own extraction, so this is a different contrast from the Graphiti arm in the [matched Kimi A/B persona benchmark](../persona_joint_kimi_ab_a100_build1/README.md), and the difference does not isolate Scallop validation.
Answers were scored with the official LongMemEval judge prompts using a local Qwen3.5-4B judge rather than the official GPT-4o judge, so absolute accuracies are not directly comparable to published LongMemEval numbers.

## Results by question category (exploratory)

Abstention questions form their own stratum. The subgroups are small and many were examined; no per-category difference reaches McNemar significance.

| Stratum | n | NeuroSym | Graphiti | Difference, 95% paired bootstrap CI | NeuroSym-only / Graphiti-only | Exact McNemar p |
|---|---:|---:|---:|---:|---:|---:|
| single-session-user | 64 | 71.9% | 71.9% | +0.0 [−6.2, +6.2] | 2 / 2 | 1 |
| single-session-assistant | 56 | 57.1% | 58.9% | −1.8 [−10.7, +7.1] | 3 / 4 | 1 |
| single-session-preference | 30 | 23.3% | 16.7% | +6.7 [−10.0, +23.3] | 4 / 2 | 0.688 |
| multi-session | 121 | 37.2% | 35.5% | +1.7 [−6.6, +9.9] | 13 / 11 | 0.839 |
| knowledge-update | 72 | 66.7% | 58.3% | +8.3 [−1.4, +18.1] | 9 / 3 | 0.146 |
| temporal-reasoning | 127 | 37.8% | 35.4% | +2.4 [−4.7, +9.4] | 12 / 9 | 0.664 |
| abstention | 30 | 83.3% | 90.0% | −6.7 [−20.0, +6.7] | 1 / 3 | 0.625 |

[`comparisons.csv`](comparisons.csv) also breaks results down by dataset question type and by evaluation batch.

The scorer's other metric on these histories: task-averaged accuracy 50.4% (NeuroSym) and 48.1% (Graphiti). Population-weighted accuracy equals the unweighted accuracy on the full 500.

## Earlier and later histories

The later 250 histories score lower than the earlier 250 in both arms (NeuroSym 46.8% versus 53.6%, Graphiti 44.0% versus 52.4%), because they contain more of the harder question types: multi-session and temporal-reasoning questions are 32.8% and 32.8% of the later histories versus 15.6% and 18.0% of the earlier 250. Re-weighting each half to the full benchmark's stratum mix removes the gap: NeuroSym 49.5% versus 49.2% (−0.3 points, 95% CI [−9.0, +8.3]), Graphiti 48.7% versus 47.5% (−1.2 points, 95% CI [−9.9, +7.3]). The paired difference is +1.2 points on the earlier 250 and +2.8 points on the later 250.

## Method notes

Both arms ingest one set of facts per history, extracted by the same model and checked by a verifier before ingestion. Where a chunk failed the extractor's or the verifier's checks, it was re-extracted in smaller pieces by the same extractor and verifier (23 histories); a fact the verifier did not confirm was excluded, and a fact judged uncertain or rejected with the whole chunk in view was excluded even if a smaller piece accepted it (19 facts in 10 histories, including 4 excluded by name after manual review). Where the extractor's and the verifier's labels disagreed on a fact the verifier supported, the fact was kept and both labels were recorded (29 histories). Nine histories of the earlier 250 use extractions completed by an earlier repair procedure. NeuroSym stores one Neo4j relationship type per predicate, so after the first database instance reached Neo4j's relationship-type limit, the last 62 histories were evaluated against a second, empty instance with the same configuration; NeuroSym reads are scoped to the history being evaluated.

## Sensitivity

| Histories | n | NeuroSym | Graphiti | Difference, 95% paired bootstrap CI | Exact McNemar p |
|---|---:|---:|---:|---:|---:|
| All histories | 500 | 50.2% | 48.2% | +2.0 [−1.4, +5.6] | 0.308 |
| Without histories kept with a recorded label dispute | 471 | 50.3% | 48.2% | +2.1 [−1.5, +5.7] | 0.295 |
| Without histories recovered in smaller pieces | 477 | 50.3% | 48.0% | +2.3 [−1.3, +5.9] | 0.248 |
| Without histories with full-context exclusions | 490 | 50.6% | 48.8% | +1.8 [−1.6, +5.3] | 0.362 |
| Without earlier repaired extractions | 491 | 50.3% | 48.3% | +2.0 [−1.4, +5.5] | 0.308 |
| Without histories on the second NeuroSym instance | 438 | 50.2% | 48.2% | +2.1 [−1.6, +5.7] | 0.328 |
| Without any of the five groups above | 412 | 50.7% | 48.5% | +2.2 [−1.7, +5.8] | 0.321 |
| Earlier 250 (the published result) | 250 | 53.6% | 52.4% | +1.2 [−3.2, +5.6] | 0.728 |
| Later 250 | 250 | 46.8% | 44.0% | +2.8 [−2.4, +8.0] | 0.371 |

Across these subsets the paired difference ranges from +1.2 to +2.8 points, and no interval excludes zero.

## Metric definitions

- **Unit:** one question per history; each arm produces one answer, judged once.
- **Correct:** the judge's response contains "yes" (case-insensitive). Prompts are the verbatim LongMemEval `get_anscheck_prompt` templates, including the abstention, temporal off-by-one, knowledge-update, and preference variants. A pipeline failure would count as incorrect; none occurred in these 500 histories.
- **Uncertainty:** two-sided 95% Wilson intervals per arm; 95% percentile bootstrap over histories for the paired difference (10,000 resamples, seed 20261114, as for the 250-history result); exact two-sided McNemar test. The earlier-versus-later comparison resamples histories within each half and stratum.

## Models and data

- Dataset: `xiaowu0162/longmemeval-cleaned`, `longmemeval_s_cleaned`, revision `98d7416c24c778c2fee6e6f3006e7a073259d48f`.
- Shared fact extraction and answer generation: `Qwen/Qwen3.5-4B`, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Embeddings: `BAAI/bge-small-en-v1.5`, revision `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`. Graphiti: `graphiti-core` 0.29.3.
- Judge: `Qwen/Qwen3.5-4B`, temperature 0, at most 10 output tokens, thinking disabled; the judge revision was not recorded. The official LongMemEval judge is `gpt-4o-2024-08-06`.

## Files and regeneration

- [`per_history.csv`](per_history.csv): one row per evaluated history with the question ID, stratum, question type, abstention flag, and each arm's binary judge label; the rows of the earlier 250 are identical to the 250-history result. It contains no dataset text, gold answers, or model outputs.
- [`per_history_groups.csv`](per_history_groups.csv): per history, the evaluation batch (`published_250` or a later batch) and 0/1 flags for the groups in the sensitivity table, plus the NeuroSym database instance (1 or 2). It is kept separate so that `per_history.csv` has the same columns as the 250-history table.
- [`summary.csv`](summary.csv), [`paired_comparison.json`](paired_comparison.json), and [`comparisons.csv`](comparisons.csv) are generated by `analyze_results.py`.
- [`run_metadata.json`](run_metadata.json) records the dataset, model revisions, judge settings, and statistics method.
- Regenerate with `python3 results/longmemeval_s_shared_facts_500_20261117/analyze_results.py --markdown` (standard library only).
