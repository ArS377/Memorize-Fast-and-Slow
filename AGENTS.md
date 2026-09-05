# Repository verification and safety

- This is a curated local submission-preparation repository, not the original execution checkout. Historical Git IDs are provenance only; do not reconstruct them here.
- Preserve bundled `results/` bytes and hash pins. Do not normalize input data or re-sign its original manifests. Keep external evidence read-only and use explicit repo-shaped roots.
- Do not start experiments, models, service launchers, package installations, or downloads during routine verification. Never touch another active run or its environment.
- Run from the repository root with an independently available Python interpreter. Use `python -B -m pytest -q -p no:cacheprovider` for the retained offline suite. Set `PYTHONDONTWRITEBYTECODE=1` and `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`; temporary test files belong in fresh locations, not source/evidence inputs.
- Run all three `python -m experiments.paper_persona --mode MODE --evidence-root . --output-root outputs/preview --run-id check` previews without `--prepare` or `--execute`. Expected generation counts: historical_a=960, kimi_a=960, kimi_ab=1920. Kimi A authenticates both A and B inputs.
- `python scripts/artifact_resources.py verify bundled-persona --root .` validates 41 exact input files. Full historical checks use explicit external roots and the pinned v002-git digest documented in README.md.
- Bash verification is syntax-only: `bash -n scripts/run_graphiti_baseline.sh` and `bash -n scripts/supervise_joint_run.sh`. Do not launch those scripts to test syntax.
- Modern corpus-generation transaction tests require POSIX directory fsync. Scallopy and tensor/transformer tests have additional optional/platform requirements. Report skips and unavailable integrations separately; do not weaken authentication or scientific assertions.
- Dependency files are separate specifications, not validated locks. See `configs/dependency_environments.json`; do not combine conflicting historical Torch/Transformers environments or change recorded pins to make resolution succeed.
- Rebuild a fresh source manifest after source/config/dependency changes and keep its SHA-256 independently. Verify an archive without `.git` and a local checkout's exact data hashes. Do not overwrite existing source manifests or frozen reports.
- No remote or push is part of the migration. No new license grant, anonymous hosting, exact-execution recovery, or final submission-readiness claim is implied.
