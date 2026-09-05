# NeuroSym submission preparation

Local research-code preparation for the persona memory comparison and supporting continual/interleaved memory benchmarks. This repository has fresh history derived from preparation commit `6a114c8d55ec74d05a2673c8fbd622824acbdf26`; historical revisions are provenance identifiers, not commits available in this repository.

This is not yet an anonymized, licensed, reviewer-accessible release. No new model experiment is claimed. The historical primary table uses deterministic Surface A; the newer pair-conditioned Kimi surfaces are a separate selectable evaluation input.

## Supported scope

| Mode | Input | Planned generations |
|---|---|---:|
| `historical_a` (default) | Historical deterministic Surface A | 960 |
| `kimi_a` | Pair-conditioned Kimi Surface A | 960 |
| `kimi_ab` | Pair-conditioned Kimi A and B | 1,920 |

Every mode uses the same eight arms: sliding context, structured memory, hybrid KG, and Graphiti, each at 4K and 16K. Counts above describe plans, not completed new results. `persona_fixed_assignment_analysis` is retained for compatibility and validation of an older five-arm layout; it is not the main eight-arm comparison.

The two supporting pipelines are `experiments.continual_memory_benchmark` and `experiments.interleaved_memory_benchmark_v3`. Shared helpers retain their existing module names and scientific behavior. Unrelated LongBench, RLM, standalone experiment, dashboard, and old-history audit workflows are not part of this package.

## Inputs and artifact access

Four complete persona corpora and their pair gate are bundled under `results/`: the parent `persona_conflict_conversations_v1`, historical `persona_conflict_conversations_surface_a_graphiti_build2`, modern `persona_conflict_conversations_surface_a`, modern `persona_conflict_conversations_surface_b`, and `persona_surface_pair_gate.json`. The 41 files occupy 14,168,913 raw bytes. Both modern modes require the parent, A, B, and pair gate even when only A is evaluated.

These files are exact original Git-blob bytes. Git attributes disable text conversion and filters for bundled data and fixtures. Do not normalize, sanitize, replace, or re-sign corpus manifests: doing so breaks their pinned identity. Keep new outputs in `outputs/` or a separate fresh directory, never in bundled input directories.

`configs/artifact_resources.json` lists required contents, roles, and trusted SHA-256 digests. Access commands require an explicit repo-shaped root containing `results/`; there is no fallback to an original checkout, automatic download, or artifact import.

```bash
python scripts/artifact_resources.py list
python scripts/artifact_resources.py verify bundled-persona --root .
python scripts/artifact_resources.py access kimi_a --root .
python scripts/artifact_resources.py access synthetic-inputs --root /path/to/v002-git
python scripts/artifact_resources.py access reference-results --root /path/to/v002-git
```

`access` returns paths only after verification succeeds. It is a read-only check, not a lock against later modification. It refuses mismatched/missing/extra members, unsafe paths, links, and unmaterialized LFS pointers. It does not expose or copy `baseline_source` as ordinary run output. Full synthetic inputs and historical reference results remain external and are ignored by Git if placed under `external-artifacts/` or unbundled `results/` directories.

The current local historical source is the separately preserved `v002-git` bundle. Obtain that bundle from its custodian, preserving its repo-shaped layout and exact bytes. Verify it before use:

```bash
python scripts/paper_artifacts.py verify /path/to/v002-git --require-trusted --expected-manifest-sha256 cb432774f962b5d480c6958472bec5eac3343fea3376cad6de2d19b7e9141443
```

This checks all frozen contents, not only selected resource groups. `v002-git` verifies 47 historical file identities and inventories 141 additional entries. It does not contain new modern A/B benchmark results. `v001-as-found` is retained separately with digest `e1dfaaf42e478855dfbc869aa58eb2a9c715d2fb360440ca6d98e1ad9923e087` and 43 known historical hash mismatches from checkout conversion; those mismatches must not be repaired in the preserved snapshot.

**Anonymous reviewer access for external bundles is not configured.** There are no public download URLs in this preparation repository. Hosting or ZIP inclusion, permissions/licensing, anonymization, and final supplementary-archive sizing remain release gates. Local access alone does not satisfy reviewer availability.

## Lightweight usage and dependencies

Use root-based `python -m` execution in an independently prepared Python environment. Dependency files are specifications, not validated locks; no package installation is part of this migration. The lightweight specification is `requirements.txt`. Optional environment specifications are kept separate for Scallop, dense retrieval, generation, historical direct evaluation, and Graphiti/extraction. See the environment details below before combining any files.

### Environment specifications

`configs/dependency_environments.json` records the basis and limitations of each specification.

| Specification | Scope / recorded environment |
|---|---|
| `requirements.txt` | NumPy, rank-bm25, OpenAI, Neo4j, HTTPX, pytest; lightweight imports and offline checks |
| `requirements-scallop.txt` | Scallopy 0.2.4 wheel only; CPython 3.10, Linux x86_64, glibc compatible with manylinux_2_27 |
| `requirements-dense.txt` | Recorded continual Python 3.12.9 stack: sentence-transformers 3.4.1, transformers 4.57.6, tokenizers 0.22.2; its Torch build remains unresolved |
| `requirements-generation.txt` | Primary persona stack: Torch 2.13.0 (recorded build `2.13.0+cu130`), transformers 5.15.1, sentence-transformers 3.4.1 |
| `requirements-qwen35-eval.txt` | Unchanged separate direct-evaluation record: Torch 2.6.0 (recorded `+cu124`), transformers 5.15.0 and associated pins |
| `requirements-graphiti-baseline.txt` | Optional graphiti-core 0.29.3 and pinned Neo4j/OpenAI clients; also needs a compatible embedding/generation environment |
| `requirements-extraction.txt` | Separate vLLM 0.27.1 serving environment recorded in historical source notes |

Do **not** combine the conflicting Torch/Transformers specifications. CUDA wheel sources, drivers, binary compatibility, transitive resolution, and model/service integration remain unvalidated. Local checks use already available packages, including NumPy 2.4.4 rather than the recorded 2.2.6; they do not establish a reproducible installed lock. Existing comments in `requirements.txt` are preserved historical notes, not current installation instructions: `rlms` is excluded, Python-version bypasses are not recommended, and old claims about Neo4j release availability are not current facts.

All example paths beginning `/path/to/` are placeholders. Substitute absolute local paths and quote paths containing spaces. The same Python commands work in PowerShell with Windows paths.

### Authenticated previews: no models, services, or writes

```bash
python -m experiments.paper_persona --mode historical_a --evidence-root . --output-root outputs/runsets --run-id trial01
python -m experiments.paper_persona --mode kimi_a --evidence-root . --output-root outputs/runsets --run-id trial01
python -m experiments.paper_persona --mode kimi_ab --evidence-root . --output-root outputs/runsets --run-id trial01
```

Preview is the default. It authenticates provenance and returns a plan without creating directories, loading models, contacting services, or dispatching evaluators. It rejects wrong/missing parent or pair members and historical/modern substitution.

`--prepare` explicitly creates configurations and `plan.json` only. `--execute` explicitly starts a model/service experiment and requires separately prepared model paths, Scallop, Neo4j, Graphiti, and environment variables declared in the template. Neither action is needed for preview or offline verification. This migration does not execute experiments.

Each invocation uses `<output-root>/<mode>/<run-id>/` with independent surface outputs, caches, indexes, and Graphiti build IDs. Existing runsets are refused. Preparation consumes the run ID; subsequent top-level execution requires a fresh ID. Underlying evaluator resume remains subject to exact source/configuration/cache compatibility and does not permit reuse of an incompatible historical run.

### Analyze saved results

Authenticate the external freeze before historical reaggregation:

```bash
python scripts/verify_paper_results.py --root /path/to/v002-git --expected-manifest-sha256 cb432774f962b5d480c6958472bec5eac3343fea3376cad6de2d19b7e9141443
```

This verifies all 16 primary EM/F1 values, eight overall paired contrasts, and supporting continual/interleaved saved-record aggregates. It does not rerun models, extraction, tokenization, Scallop, or full stratified primary bootstraps. Numerical agreement without the external digest is explicitly unauthenticated. Optional `--output` must point to a fresh report outside frozen evidence.

For new completed runs, pass the exact generated configurations:

```bash
python -m experiments.paper_persona_analysis --mode historical_a --run-a /path/to/runset/surface_a --config-a /path/to/runset/configs/surface_a.json --evidence-root . --output-dir outputs/new-analysis
python -m experiments.paper_persona_analysis --mode kimi_ab --run-a /path/to/runset/surface_a --config-a /path/to/runset/configs/surface_a.json --run-b /path/to/runset/surface_b --config-b /path/to/runset/configs/surface_b.json --evidence-root . --output-dir outputs/new-ab-analysis
```

For the preserved table, use `v002-git/results/persona_joint_surface_a_build2` and the exact frozen `v002-git/baseline_source/configs/persona_end_to_end_joint_surface_a.json`. Historical configuration/source bytes are verification references, not project imports or resolvable local Git revisions.

Analysis checks complete eight-arm coverage, result and input hashes, configuration/model/source compatibility, and matched recorded checkpoints. A+B uses surface-qualified condition IDs but the same **12 base-history clusters**, not 24 independent histories. This is a pair-conditioned fixed-assignment comparison: the bootstrap unit is the base history, and A/B are not statistically independent mapping replicates. During generation, B receives only a flat normalized list of A target phrases for collision avoidance, not A's assignment mapping or dialogue.

### Supporting benchmark interfaces

Inspect these interfaces without executing an experiment:

```bash
python -m experiments.continual_memory_benchmark --help
python -m experiments.interleaved_memory_benchmark_v3 --help
```

Continual execution requires `--dataset` pointing to the verified external `synthetic_temporal_preferences_1000_v4_stream`, a fresh `--output-dir`, and `--config configs/continual_memory_benchmark.json`. Interleaved v3 requires the verified `synthetic_temporal_preferences_1200_v5_interleaved`, a fresh `--output-dir`, and `--scallop-endpoint` (or `SCALLOP_VALIDATOR_URL`). Real execution additionally needs the recorded local tokenizers/embeddings and separately prepared service environment; those resources are not downloaded by these preparation commands.

The two Bash Graphiti launchers are retained for later, explicit execution. Set `PYTHON_BIN` to an independently prepared interpreter's absolute path and supply the declared service/model variables. Launchers use fresh output paths and refuse destructive cleanup. Do not start them merely to check this package; `bash -n` is sufficient for syntax verification. Their inherited historical host comments describe provenance, not current services.

## Source archives and offline tests

`source_manifest.json` is rebuilt for this curated source and dependency scope. Its externally retained digest is recorded with migration verification; it is not the old preparation manifest. After deliberate code/specification edits, create a new manifest at a fresh path and keep its digest independently:

```bash
python -m neurosym.application.source_provenance --create . --output source_manifest-new.json
python -m neurosym.application.source_provenance --verify . --manifest source_manifest-new.json --sha256 MANIFEST_DIGEST
```

When running without `.git`, set `NEUROSYM_SOURCE_MANIFEST` and `NEUROSYM_SOURCE_MANIFEST_SHA256` to that manifest and external digest. In PowerShell use `$env:NEUROSYM_SOURCE_MANIFEST` and `$env:NEUROSYM_SOURCE_MANIFEST_SHA256`; in Bash use `export`. Missing, changed, extra, or incompatible source files fail closed. The manifest authenticates declared source/config/dependency bytes, not installed packages or model weights. Dataset identities are checked separately.

```bash
python -B -m pytest -q
bash -n scripts/run_graphiti_baseline.sh
bash -n scripts/supervise_joint_run.sh
```

The retained offline suite uses bundled inputs and temporary Git repositories, not old source revisions. Real-data authentication and tampering checks remain active in archives. Tests distinguish POSIX directory-fsync requirements, optional package availability, and explicit full-evidence integration. Never treat unavailable platform/integration checks as passes. Historical reconstruction/AST audits remain in the original private provenance workflow, not the portable default suite. No test should discover or fall back to an active run or original checkout.

Historical `paper_artifacts.py` maintenance commands (`inventory`, `freeze`, `source-audit`) require an explicit original `--repo` and `--selection`. Do not use this fresh repository to reconstruct old revisions. Ordinary verification and analysis require no original Git objects.

## Provenance limits and release gates

The primary evaluator/configuration have matching historical source blobs, but the complete original execution base/dirty files remain unresolved. Two continual source hashes are unresolved; interleaved source is candidate-only. Matching saved records and bytes is not proof of complete original execution recovery.

Exact input metadata can contain identifying provenance. No sanitization or re-signing of historical corpus manifests occurred. A separate hash-aware anonymized release, licensing review, validated environment builds, model/service reruns if desired, anonymous external access, and final ZIP assembly remain future decisions. No license grant or final NeurIPS compliance claim is made here.
