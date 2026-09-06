# NeuroSym

Inference-only persona-memory benchmark: sliding context, structured memory, hybrid KG memory, and Graphiti at 4K and 16K token budgets. Paper title and anonymous manuscript link await author confirmation.

## Requirements

**Quick start: verify Table 1 without a GPU.** Run from this repository's root with Python and the lightweight dependencies available. You also need the external saved-answer bundle below; bundled inputs alone cannot produce Table 1.

```powershell
$Python = "python"
$Evidence = "C:\path\to\evidence"
& $Python -B scripts/verify_paper_results.py --table1 --root "$Evidence"
```

Replace `$Python` with your prepared interpreter if needed. `$Evidence` must contain the complete `results/persona_joint_surface_a_build2/` directory (15 files, 13,937,870 bytes); an existing `v002-git` root works directly.

Expected: exit code 0, `Verified pinned reference-persona artifacts and saved-record scores.`, then `120 conditions; 12 histories; 8 arms; 960 generations.` and the four-row table in [Results](#results). Missing or altered evidence fails without printing a table.

For a **new, isolated lightweight environment**, the dependency-install command is:

```powershell
python -m pip install -r requirements.txt
```

This is a specification, not a tested clean-install recipe. Verification used existing Python 3.13.7, NumPy 2.4.4, and the other versions pinned in `requirements.txt`; that file instead pins NumPy 2.2.6. Verification itself installs nothing.

### Dependencies for model execution

Keep these environments separate where indicated. Their evidence and unresolved dependencies are recorded in [dependency_environments.json](configs/dependency_environments.json).

| Specification | Role and recorded versions |
|---|---|
| [requirements.txt](requirements.txt) | Lightweight verification, previews, and tests; no model packages |
| [requirements-generation.txt](requirements-generation.txt) | Primary persona generation: Torch 2.13.0 (recorded build `2.13.0+cu130`), Transformers 5.15.1, Sentence Transformers 6.0.0 |
| [requirements-graphiti-baseline.txt](requirements-graphiti-baseline.txt) | Graphiti client additions: graphiti-core **0.29.3 exactly**, Neo4j driver 5.28.3, OpenAI 1.109.1; requires a compatible generation/embedding environment |
| [requirements-extraction.txt](requirements-extraction.txt) | Independent extraction server: vLLM 0.27.1, Linux/GPU |
| [requirements-scallop.txt](requirements-scallop.txt) | Independent validator: scallopy 0.2.4 **CPython 3.10 Linux x86_64** manylinux_2_27 wheel; not Windows Python 3.13 |
| [requirements-dense.txt](requirements-dense.txt) | Supporting continual/dense run: Python 3.12.9, Sentence Transformers 3.4.1, Transformers 4.57.6, NumPy 2.2.6, tokenizers 0.22.2; Torch build unknown |
| [requirements-qwen35-eval.txt](requirements-qwen35-eval.txt) | Separate historical direct evaluation: Torch 2.6.0 (recorded `+cu124`), Transformers 5.15.0; not the primary persona environment |

Do not merge the primary, continual/dense, direct-evaluation, Scallop, and extraction stacks into one environment or substitute versions from another run. See [Limitations and release status](#limitations-and-release-status) for validation status.

### Data access

The 41 bundled input files occupy 14,168,913 bytes under `results/`: the parent corpus, historical deterministic Surface A, pair-conditioned Kimi A/B, and their pair gate. They include dialogue, queries, facts, candidate updates, source documents, generation requests/responses, and manifests.

**Saved answers are external.** Obtain the primary reference directory from the artifact custodian, preserving layout and bytes; nothing downloads automatically. [artifact_resources.json](configs/artifact_resources.json) lists required members. Reviewer access is a release gate below.

## Training

**Not applicable: there is no training or fine-tuning.** All models are used off-the-shelf for inference. No optimizer, training procedure, `train.py`, or newly trained checkpoint is needed or supplied. Memory ingestion and synthetic-data generation are not model training.

## Evaluation

### Verify saved results: CPU-only, exact saved-record check

Use the quick-start command. It authenticates the primary reference files, rescores all 960 answers against recorded gold, checks all 16 EM/F1 values and eight overall paired contrasts, and rechecks file integrity before printing Table 1.

It does not regenerate answers, reconstruct gold from dialogue, retokenize prompts, rebuild stores, or rerun stratified bootstraps. Models, Neo4j, Graphiti, Scallop, and Kimi API access are unnecessary. Failures return nonzero with no partial table; `--table1` writes only to stdout.

[verify_paper_results.py](scripts/verify_paper_results.py) uses the original persona evaluator and [answer_eval.py](experiments/answer_eval.py). Scoring uses the first nonempty answer line, removes its answer prefix, and normalizes case, articles, punctuation, and whitespace.

Token F1 uses multiset token overlap: precision divides overlap by prediction length, recall by gold length. Scores are aggregated over matched conditions, not model-tokenizer subwords.

### Rerun the experiment: GPU and isolated services

First inspect the authenticated plan; this command loads no models, contacts no services, and creates no output directories:

```powershell
& $Python -B -m experiments.paper_persona --mode historical_a --evidence-root . --output-root outputs/preview --run-id check
```

| Mode | Input and scope | Planned generations |
|---|---|---:|
| `historical_a` | Deterministic historical Surface A; Table 1 input | 960 |
| `kimi_a` | New pair-conditioned Kimi A; not Table 1 | 960 |
| `kimi_ab` | New pair-conditioned Kimi A+B; not Table 1 | 1,920 |

Both Kimi modes authenticate the parent, both bundled siblings, and pair gate, even when only A is evaluated. Counts are plans, not evidence of completed runs. Kimi API access is not required to evaluate the bundled corpora.

This is a pair-conditioned fixed-assignment comparison: the bootstrap unit is the base history. A/B are not statistically independent mapping replicates. B receives only a flat normalized list of A target phrases for collision avoidance, not A's mapping or dialogue.

Prepare the model snapshots below, generation/Graphiti client environment, Scallop HTTP validator, Qwen3 extraction endpoint (recorded context 32,768), and dedicated hybrid/Graphiti Neo4j instances. Never reuse an active run's services, stores, caches, or output paths.

Supply the [execution settings](#advanced-execution-settings) from [the scientific configuration](configs/persona_end_to_end_joint_surface_a.json). The CLI supplies dataset/output/index/cache paths and build IDs. Once independently provisioned, invoke:

```powershell
& $Python -B -m experiments.paper_persona --mode historical_a --evidence-root . --output-root outputs/runsets --run-id rerun01 --execute
```

**This execution command is not end-to-end validated here.** It writes outputs and mutates dedicated stores; it is not quick-start verification. Use a fresh run ID. `--prepare` creates configs only and consumes that ID; do not prepare and then execute the same ID.

For a completed new historical-A run, use its exact generated config to analyze it into a fresh directory:

```powershell
& $Python -B -m experiments.paper_persona_analysis --mode historical_a --run-a outputs/runsets/historical_a/rerun01/surface_a --config-a outputs/runsets/historical_a/rerun01/configs/surface_a.json --evidence-root . --output-dir outputs/analysis-rerun01
```

This analysis command requires a completed new run and was not tested on one here. It does not replace the Table 1 verifier. A+B analysis also requires `--run-b` and `--config-b`; it clusters by the same 12 base histories, not 24 independent histories.

## Pre-trained Models

Only external, off-the-shelf models are used; no weights are redistributed. Obtain the exact snapshots from their upstream repositories for model execution. No model download is needed for saved-answer verification.

| Model | Role | Snapshot revision | Upstream license / terms |
|---|---|---|---|
| [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) | Primary answers **and primary-run tokenizer** | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` | [Apache-2.0](https://huggingface.co/Qwen/Qwen3.5-4B/raw/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a/LICENSE) |
| [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) | Graphiti extraction; separate schedule/supporting tokenizers | `1cfa9a7208912126459214e8b04321603b3df60c` | [Apache-2.0](https://huggingface.co/Qwen/Qwen3-4B/raw/1cfa9a7208912126459214e8b04321603b3df60c/LICENSE) |
| [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) | Hybrid and Graphiti embeddings | `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a` | [MIT (pinned model card)](https://huggingface.co/BAAI/bge-small-en-v1.5/raw/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a/README.md) |
| Kimi K3 (`kimi-k3`) | Synthetic-data generation only | API identifier; immutable API revision not recorded | [Kimi OpenPlatform Terms](https://platform.kimi.ai/docs/agreement/modeluse), not a weights license |

The primary manifest records Qwen3.5/BGE revisions. Qwen3's revision is pinned for separate tokenizers, but extraction records name the model without its served revision. The extraction revision above is author-supplied and needs serving-record confirmation.

The linked Kimi API terms are dated July 30, 2026. Authors must confirm the agreement applicable when the data was generated and permission to release the outputs; a model-weights license is not a substitute.

## Results

### Table 1: Primary end-to-end benchmark (historical Surface A)

120 conditions, 12 histories, four methods at two budgets, 960 answers. Values are percentages; bold marks each column's maximum. The CPU verifier printed these values from authenticated saved answers, matching the supplied paper table.

| Method | 4K EM | 4K F1 | 16K EM | 16K F1 |
|---|---:|---:|---:|---:|
| Sliding context | 35.00 | 37.42 | 60.00 | 64.58 |
| Structured memory | 69.17 | 70.75 | 70.83 | 72.50 |
| Graphiti | 48.33 | 54.33 | 66.67 | 69.17 |
| Hybrid KG memory | **78.33** | **79.58** | **75.83** | **75.83** |

Graphiti denotes **Graphiti ingestion with benchmark-matched retrieval**, not stock Graphiti. Primary decoding is greedy, batch size 1, 32 new tokens, thinking disabled, bfloat16, SDPA. Scientific settings remain in the shipped configuration and evaluator.

Paired differences use 2,000 history-cluster bootstrap resamples (seed 73), with percentile 95% intervals. Hybrid minus Graphiti EM is +30.00 points [20.00, 37.50] at 4K and +9.17 [-1.67, 20.00] at 16K. Hybrid minus structured at 16K is +5.00 [-0.83, 10.00]; neither 16K interval excludes zero.

### Recorded compute

The reference run used a five-RTX-3090 host, extraction on GPU 0 and generation on GPU 2; embeddings ran on CPU. Three Neo4j 5.26.0 Community instances occupied separate ports: one pre-existing instance was untouched, and hybrid/Graphiti used separate instances.

Summing `generation_seconds` over the 960 saved answers gives **89.12 minutes**: 18.81 at 4K and 70.31 at 16K. This excludes setup, Graphiti ingestion (312 episodes), KG construction, and analysis; it is not total experiment wall time or total GPU-hours.

### Limitations and release status

- Table 1 covers **historical A only**. The second-surface run is pending per the authors and was not inspected. No completed Kimi A/B or two-surface robustness result is claimed here.
- The 12-history, synthetic, two-query-family comparison measures downstream utility, not state fidelity. Account scoping is supplied to both memory arms. One stochastic Graphiti ingestion build is represented; intervals omit between-build/model-run variance, and rerun scores may differ.
- **None of the dependency environments has been clean-install validated.** Wheel sources, transitive resolution, CUDA/driver compatibility, and model/service integration remain unvalidated. Local offline success is not an installed lock or end-to-end replication.
- CPU model/count, host RAM, measured peak memory/storage, excluded-stage timings, and complete research-project compute are not established here. Recorded generation time is only a partial compute disclosure.
- **Anonymous reviewer access is not configured, and code/data licenses await author approval.** Exact metadata contains identifying provenance; do not sanitize bundled/frozen bytes or change their pins. An anonymous supplement requires a separately approved solution to this conflict.
- Original execution-source recovery is incomplete. Verified saved artifacts and matching scores do not prove recovery of the complete historical runtime. The paper title/link and Appendix A.6 asset-license statements still need author confirmation.

## Contributing

Project and synthetic-data license grants await author approval; upstream licenses do not cover this repository. Propose changes as reviewable diffs, preserve scientific controls and bundled bytes, and run the offline checks. No identifying issue-tracker or paper link is supplied for review.

Upstream software licenses: Scallop 0.2.4 [MIT](https://raw.githubusercontent.com/scallop-lang/scallop/0.2.4/LICENSE); Graphiti 0.29.3 [Apache-2.0](https://raw.githubusercontent.com/getzep/graphiti/v0.29.3/LICENSE); Neo4j 5.26.0 Community [GPLv3](https://raw.githubusercontent.com/neo4j/neo4j/release/5.26.0/LICENSE.txt). These links verify upstream declarations, not Appendix A.6 or compliance for a final distribution.

## Advanced: execution settings

Set these in a separately provisioned runtime; do not use credentials or service addresses belonging to another experiment. Paths must resolve to the exact local snapshots above, not an unpinned model name that triggers a download.

| Variables | Required values |
|---|---|
| `PERSONA_QWEN_MODEL_PATH`, `PERSONA_QWEN_MODEL_ID`, `PERSONA_QWEN_DEVICE` | Local Qwen3.5 snapshot directory, `Qwen/Qwen3.5-4B`, assigned generation device |
| `PERSONA_EMBEDDING_MODEL_PATH`, `PERSONA_EMBEDDING_MODEL_ID`, `PERSONA_EMBEDDING_REVISION`, `PERSONA_EMBEDDING_DEVICE` | Local BGE snapshot, `BAAI/bge-small-en-v1.5`, pinned revision above, `cpu` for the recorded run |
| `PERSONA_SCALLOP_ENDPOINT` | Dedicated HTTP validator endpoint |
| `PERSONA_NEO4J_URI`, `PERSONA_NEO4J_USER`, `PERSONA_NEO4J_PASSWORD`, `PERSONA_NEO4J_DATABASE` | Dedicated hybrid instance and its authorized credentials/database |
| `PERSONA_GRAPHITI_NEO4J_URI`, `PERSONA_GRAPHITI_NEO4J_USER`, `PERSONA_GRAPHITI_NEO4J_PASSWORD`, `PERSONA_GRAPHITI_NEO4J_DATABASE` | Separate dedicated Graphiti instance and its authorized credentials/database |
| `PERSONA_GRAPHITI_LLM_BASE_URL`, `PERSONA_GRAPHITI_LLM_API_KEY`, `PERSONA_GRAPHITI_LLM_MODEL`, `PERSONA_GRAPHITI_LLM_SMALL_MODEL` | Dedicated extraction endpoint and its credential; both model settings `Qwen/Qwen3-4B` |

Supporting interfaces: `experiments.continual_memory_benchmark` and `experiments.interleaved_memory_benchmark_v3`. Their full datasets/results remain external. Inspect `--help` before separate provisioning. Optional Bash launchers are not Windows quick-start commands.

## Advanced: artifact verification and maintenance

```powershell
& $Python -B scripts/artifact_resources.py verify bundled-persona --root .
& $Python -B scripts/paper_artifacts.py verify "$Evidence" --require-trusted --expected-manifest-sha256 cb432774f962b5d480c6958472bec5eac3343fea3376cad6de2d19b7e9141443
```

The first checks 41 inputs; the second requires the complete `v002-git` freeze. For supporting saved-record aggregates, run `scripts/verify_paper_results.py --root "$Evidence" --expected-manifest-sha256` with that same digest, without `--table1`.

Keep evidence read-only. Historical Git IDs and `source_export_inventory.json` describe provenance, not current working-tree identity. Preserve earlier source manifests; after edits create a fresh manifest and keep its printed SHA-256 independently:

```powershell
& $Python -B -m neurosym.application.source_provenance --create . --output source_manifest-submission-materials.json
& $Python -B -m neurosym.application.source_provenance --verify . --manifest source_manifest-submission-materials.json --sha256 MANIFEST_DIGEST
```

Creation refuses an existing path; choose a fresh name for later edits. Without `.git`, set `NEUROSYM_SOURCE_MANIFEST` and `NEUROSYM_SOURCE_MANIFEST_SHA256` to the matching manifest and independent digest. Source identity is separate from data hashes and environment validation.

### Offline checks

```powershell
$env:PYTHONDONTWRITEBYTECODE = "1"
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = "1"
& $Python -B -m pytest -q -p no:cacheprovider
```

Tests use temporary files, not model experiments. Report optional-package, POSIX-fsync, and external-evidence skips separately. Legacy comments in `requirements.txt` are not instructions to install `rlms`, bypass Python requirements, or infer current package availability.
