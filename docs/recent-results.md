# Recent Results

This index consolidates the experiment evidence produced or integrated from August 2 through August 23, 2026.
The artifacts in each linked result directory remain authoritative.
Status labels distinguish current evidence from preliminary, superseded, smoke-only, and incomplete runs.

## Current Evidence

| Run | Scale | Primary result | Interpretation |
|---|---:|---|---|
| [Continual memory v3](../results/continual_memory_benchmark_v3_stream/report.md) | 150 episodes, 1,950 checkpoints, 408,761 maximum stored tokens | Grounded accuracy: full structured 100%, 128K sliding 95.38%, 64K 90.77%, BM25 61.18%, dense 46.87%, recency 26.15%. Scallop injection reached 99.67-100%. | Completed synthetic context-availability and causal-ablation evidence. It uses a deterministic oracle resolver, not LLM generation or automatic extraction. |
| [Interleaved memory v3](../results/interleaved_memory_benchmark_v3/report.md) | 1,024 histories, 2,047 evaluated delayed queries | Structured versus sliding grounded availability was 58.18% versus 20.86% at 64K and 72.35% versus 47.78% at 128K. | Completed deterministic availability evidence. Structured stale intrusion remained 29.07% at 64K and 19.35% at 128K. |
| [Pair-conditioned persona analysis](../results/persona_surface_generalization_v1/analysis.md) | 240 conditions per arm, clustered over 12 base histories | Pooled exact match: structured 4K 60.8% versus sliding 30.4%, and structured 16K 62.1% versus sliding 52.9%. | Completed descriptive analysis. A and B share latent histories and are not independent replications. |
| [Neo4j NeuroSym persona benchmark](../results/persona_end_to_end_qwen35_4b_neurosym_v1/README.md) | 120 conditions, 840 Qwen generations | Hybrid KG exact match was 75.83% at 4K and 80.83% at 16K, versus structured memory at 63.33% and 65.0%, and sliding context at 32.5% and 56.67%. | Completed authenticated end-to-end run over 12 synthetic history clusters. Server-instance identity was not authenticated. |
| [Graphiti external baseline](../results/persona_joint_surface_a_build2/README.md) | 120 conditions, 960 Qwen generations, 312 Graphiti episodes | Hybrid KG versus Graphiti exact match was 78.33% versus 48.33% at 4K and 75.83% versus 66.67% at 16K. The 4K paired difference was +30 points, 95% CI [+20, +37.5]. | Completed but preliminary external comparison. It is one stochastic, benchmark-matched Graphiti build over the preserved historical Surface-A corpus; the 16K gap is unresolved. |
| [Differentiable admission stress](../results/differentiable_admission_v2_stress/report.md) | 6,300 train, 1,350 dev, 1,350 test candidates; five repeats per noise level | At noise 0.30, eligible mean accuracy was 93.82% logistic, 95.96% MLP, 95.36% differentiable Scallop, and 92.60% fixed symbolic, with zero hard-gate violations. | Completed synthetic robustness test. Repeated templates and synthetic confidence noise limit external validity. |
| [Multi-task context v2](../results/multitask_context_benchmark_v2_hard/report.md) | 120 cases | BM25 evidence recall was 25%, recency 8.33%, random 5.83%, and oracle controls 100%. Head-window recall rose from 33.3% at 4K to 100% at 128K. | Completed context-selection benchmark with deterministic resolution. No language model was evaluated. |
| [Personalized memory v2](../results/personalized_memory_benchmark_v2/report.md) | 100 histories, 300 candidates per condition | Scallop decision accuracy was 100% versus 33.3% for accept-all. Full structured history reached 100%; BM25 reached 44.4-55.6%. | Completed deterministic synthetic sparse-retrieval result. No language-model generation was evaluated. |
| [Cells 1-4 LongBench pilot](../results/cells1_4_exhaustive_20260805/README.md) | 50 matched LongBench-v2 examples | Accuracy was 22% raw, 18% KG without Scallop, 30% KG with Scallop, and 34% raw RLM. The Cell 3 minus Cell 2 difference was +12 points, McNemar p=0.0703. | Current canonical Cells 1-4 pilot, shareable with caveats. Only Cells 2 and 3 closely isolate Scallop. |

## Supporting And Historical Evidence

| Run | Scale | Primary result | Status and limitation |
|---|---:|---|---|
| [Interleaved memory v2](../results/interleaved_memory_benchmark_v2/report.md) | 1,024 tasks, 2.739M-token stream | Complete-provenance loss was 100% at 4K, 95.43% at 128K, and 61.89% at 1M. | Completed predecessor. Scallop parity used generator-labeled fixtures and does not demonstrate contradiction discovery or extraction. |
| [Qwen3.5-0.8B interleaved pilot](../results/interleaved_memory_qwen35_0_8b/metrics.json) | 128 checkpoints, 32 histories | Sliding-context exact match was 3.13% at 64K and 1.56% at 128K; 95-98% of outputs hit the token cap. | Preliminary generation pilot. Structured-memory generation was not evaluated. |
| [Persona surface A](../results/persona_end_to_end_qwen35_4b_surface_a/metrics.json) | 120 conditions, 600 generations | Structured exact match was 62.5% at 4K and 60.83% at 16K; sliding was 32.5% and 50.0%. | Completed pair-conditioned assignment, not an independent replication. |
| [Persona surface B](../results/persona_end_to_end_qwen35_4b_surface_b/metrics.json) | 120 conditions, 600 generations | Structured exact match was 59.17% at 4K and 63.33% at 16K; sliding was 28.33% and 55.83%. | Completed pair-conditioned assignment, not an independent replication. |
| [Original persona benchmark](../results/persona_end_to_end_qwen35_4b_v1/report.md) | 120 conditions, 600 generations | Structured exact match was 65.0% at 4K and 65.83% at 16K; sliding was 32.5% and 56.67%. | Historical authenticated benchmark. Repeated answer surfaces permit frequency shortcuts. |
| [Continual memory v2](../results/continual_memory_benchmark_v2_hard/report.md) | 120 episodes, 1,320 checkpoints | Full structured grounded accuracy was 100%, 4K sliding 95.45%, dense 57.42%, BM25 54.39%, and recency 38.64%. Recursive Scallop was 100% versus one-hop 66.67%. | Valid predecessor with synthetic deterministic resolution and no online learning. |
| [Continual memory v1](../results/continual_memory_benchmark_v1/report.md) | 120 episodes, 1,080 checkpoints | Full structured was 100%, 4K sliding 97.22%, dense 94.81%, BM25 72.04%, and recency 47.22%. | Superseded by stricter grounded scoring and harder streams. |
| [Differentiable admission v1](../results/differentiable_admission_v1/report.md) | 6,300 train, 1,350 dev, 1,350 test candidates | Every method reached 100% clean accuracy with zero hard-gate violations. | Execution ceiling result; repeated deterministic templates prevent an advantage claim. |
| [Multi-task context v1](../results/multitask_context_benchmark_v1/report.md) | 120 cases | BM25 reached 25%, recency 8.33%, random 4.17%, and oracle controls 100%. | Superseded context-layer run with no language-model evaluation. |
| [LongBench flat baselines](../results/rows1_4_500_summary.md) | 503 examples per arm | Raw 30.2%, BM25 38.4%, dense 32.8%, and hybrid 33.2%. | Preliminary descriptive baseline without a run manifest, compliance report, uncertainty, or paired analysis. |
| [Cell 6 selection ablation](../results/longbench_cell6_selection_ablation_20260807/README.md) | 13 runs of 50 examples | Best combined-selection hybrid reached 40%, versus 28% for git-main hybrid and dense-PPR. | Historical local-diff experiment; intervals overlap and extraction was not exhaustive. |
| [Git-main verification](../results/gitmain_verification_2026-08-05/README.md) | Three 50-example evaluations | Cell 5 hybrid 26%, Cell 6 hybrid 28%, and Cell 6 dense-PPR 28%. | Completed reproduction with commit and fallback confounds. |
| Personalized memory v1 variants | 100 histories, 200 candidates per condition | Scallop decision accuracy was 100% versus 50% accept-all; paraphrase dense retrieval was about 83.8%. | Historical templated variants with incomplete execution provenance. |

## Smoke, Incomplete, Or Non-Evidentiary Runs

| Run | Observed output | Why it is not headline evidence |
|---|---|---|
| `e2e-smoke-*` | Raw, dense, and hybrid paths completed on one example; sparse missed its one example. | Execution smoke only. One example and no relevance labels. |
| `pilot2_matched` | Raw 0%, KG without Scallop 50%, KG with Scallop 50% over two examples. | Both KG snapshots contain zero facts, so this does not evaluate KG retrieval. |
| `reproduction_20260802` | Raw 36%, KG without Scallop 32%, KG with Scallop 32% over 50 examples. | Compliance passed, but both KG arms retrieved zero triples. It is execution evidence only. |
| `pilot10_matched` | No completed metrics. | Manifest leaves every cell pending. |
| `personalized_memory_benchmark_v2_dense` | No metrics or report. | Manifest status is `running`. |
| Graphiti one-account smoke, August 21 | 219 extraction calls and 311 facts, but 7 of 10 retrievals were empty. | Invalid scientific run because timestamp and entity extraction failures dominated the result. |
| Persona recall smoke | 40 rows, 25% accuracy. | Only two independent history-query units per persona pair, so it is grossly underpowered. |

## Reading Rules

- Do not pool runs that use different corpora, retrieval contracts, extraction commits, or answer-model settings.
- Treat deterministic resolver scores as evidence availability, not language-model reasoning accuracy.
- Treat smoke runs as execution checks, not comparative evidence.
- Prefer completed manifests with authenticated artifacts and explicit limitations over prose-only summaries.
