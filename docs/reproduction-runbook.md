# Historical Result Reproduction Runbook

The original author results cannot be reproduced exactly because the original
LongBench input, graph dump, fact snapshots, and model revision are absent.
This run produces a new, reproducible matched comparison using the currently
available LongBench-v2 dataset and model endpoint.

## Active run

- tmux session: `neurosym-reproduction`
- log: `results/reproduction_20260802/run.log`
- data: `pilot_longbench50.jsonl`, current `zai-org/LongBench-v2` train split
- cells: 1 (raw flat), 2 (unconstrained KG flat), 3 (Scallop KG flat)
- graph: isolated `neurosym-baseline-neo4j` at `127.0.0.1:7688`
- validator: actual Scallop HTTP service at `127.0.0.1:8765`
- build bound: one hybrid-selected source chunk per example

## Monitoring

```bash
tmux attach -t neurosym-reproduction
```

The run is complete only when `results/reproduction_20260802/compliance_report.md`
exists and reports PASS. This is a new experiment, not an exact replication of
the historical 50-example result table.
