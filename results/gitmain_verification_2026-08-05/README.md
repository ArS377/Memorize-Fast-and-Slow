# Git-environment verification (2026-08-05)

Clean-room test of the actual committed pipeline, run from isolated git
worktrees (not the session's uncommitted local changes), to verify
reproducibility after the lead reported the pipeline producing zero facts.

- **KG build**: `experiments.build_kg_facts_file` at `main` (`a0678d8`),
  worktree checkout, rigid chunking, `--max-chunks-per-example 3` (default),
  `--max-tokens 8000`, scallop-validated. 4-way sharded across serrano
  GPU0-3 for wall-clock only. Result: 1440 facts, 50/50 docs (`kg_build/`).
- **Cell 5/6 eval**: `experiments._cli.run_cell` at the
  `agent-set-rlm-depth-one` branch (`08f1bc8`) instead of `main`, since that
  branch has the intended `--max-depth 1` default (confirmed unmerged into
  main at the time of this test -- main still defaults to 2). All other
  flags matched the established multi-hop config (`ppr-seed-count=40`,
  `branch-candidate-multiplier=5`, `max-tool-calls=3`,
  `--no-allow-unsupported-fallback`).

## Results

| Test | Correct/50 |
|---|---|
| Cell 5 (no scallop, hybrid) | 13 |
| Cell 6 (scallop-validated, hybrid) | 14 |
| Cell 6 (scallop-validated, dense_ppr) | 14 |
| **Cell 6 combined** | **28/100** |

All three runs completed with zero failures/retries. This lands close to
the historical `allfixes_genrefix` baseline (29/100), confirming the
current git environment reproduces expected accuracy -- the "zero facts"
report was not caused by the code (see also the separate 3-example direct
repro of commit `a0678d8` alone, which produced 94 facts with no anomalies;
not archived here since it used scratch paths under `/tmp` on serrano).

## `prejul31_ablation/`: walking back `a0678d8`

Isolated test of the "Fix KG extraction bias, controller blank-answer bug,
add free-text answer scoring" commit (`a0678d8`, Jul 31) by checking out its
parent (`bab533a`) in a separate worktree and repeating the build + cell 6
dense_ppr eval. Verified directly that this worktree has the old, biased
extraction prompt restored (`"question_relevance": "why this fact could
help answer the current multiple-choice question"` and `"Is the fact useful
or potentially useful for answering the question?"` -- both removed by
`a0678d8`). `--branch-candidate-multiplier/-cap` and
`--allow-unsupported-fallback`/`--no-allow-unsupported-fallback` don't
exist at this commit (added by `a0678d8`); the former was simply omitted,
the latter can't be overridden here -- cell 6 hardcodes
`allow_unsupported_fallback=True` at this revision, so that's one
uncontrolled variable in the comparison below (the post-fix run used
`--no-allow-unsupported-fallback` explicitly). `--max-depth 1` was passed
explicitly to keep depth controlled and consistent with the other test.

| | Facts built | Cell 6 dense_ppr |
|---|---|---|
| **Pre-`a0678d8`** (question-relevance-filtered extraction) | 634 | 9/50 |
| **Post-`a0678d8`** (general-purpose extraction, this dir's main result) | 1440 | 14/50 |

The bias fix roughly **doubles fact yield** (634 -> 1440) and improves
dense_ppr accuracy (9/50 -> 14/50), consistent with the commit's stated
intent: question-relevance filtering during extraction was discarding facts
that don't look useful in isolation but matter for bridging multi-hop
reasoning. The accuracy delta is directionally clear but not a fully
controlled ablation given the `allow_unsupported_fallback` confound noted
above.
