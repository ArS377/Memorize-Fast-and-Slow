# Persona Surface Coverage Audit

This audit records which visible dialogue surfaces were actually evaluated by each persona-memory method.
It distinguishes the historical deterministic-surface comparison from the matched pair-conditioned Kimi A/B run.

## Corpus Identities

| Corpus | Generation-manifest SHA-256 | Role |
|---|---|---|
| `persona_conflict_conversations_v1` | `49247f1c361319cba951b21362ffa4920b2633eb394c6aeee59765dada252fc3` | Original parent corpus used by the standalone Neo4j hybrid run |
| `persona_conflict_conversations_surface_a` | `2cfdbf18705204a9ca7412e9e962ad3bb03bd08b2426f09270abbc387b047f65` | Pair-conditioned Kimi Surface A |
| `persona_conflict_conversations_surface_b` | `986cbe8ce306e19169626d5b8729b92e120f45831a76b328cfa1e32401dfd121` | Pair-conditioned Kimi Surface B |
| `persona_conflict_conversations_surface_a_graphiti_build2` | `c64c5c4268d93691b9bdf12119fa31a91cc7e9a798182135593c85128fc89c89` | Historical deterministic Surface A used by the joint hybrid-versus-Graphiti run |

The historical deterministic Surface A is not the Kimi Surface A.
Their manifest and dialogue hashes differ, so their results cannot be pooled as repeated evaluations of one surface.

## Completed Method Coverage

| Completed result | Sliding | Structured | Full context | Hybrid KG | Graphiti |
|---|---:|---:|---:|---:|---:|
| Pair-conditioned Kimi Surface A | Yes | Yes | Yes | Yes | Yes |
| Pair-conditioned Kimi Surface B | Yes | Yes | Yes | Yes | Yes |
| Original parent Neo4j run | Yes | Yes | Yes | Yes | No |
| Historical deterministic Surface A joint run | Yes | Yes | No | Yes | Yes |

The matched eight-arm run in `results/persona_joint_kimi_ab_a100_build1` evaluates hybrid KG and Graphiti on both authenticated Kimi surfaces.
The earlier five-arm runs remain the source for the full-context arm, which is intentionally excluded from the eight-arm four-method comparison.

## What The Existing A/B Data Supports

The earlier five-arm Kimi A/B runs contain 120 matched conditions and 600 generations per surface over the same 12 base histories.
They change visible dialogue and answer wording while preserving the latent histories, facts, checkpoint identities, and evaluation structure.

| Contrast | Surface A | Surface B |
|---|---:|---:|
| Structured minus sliding at 4K | +30.00 EM points | +30.83 EM points |
| Structured minus sliding at 16K | +10.83 EM points | +7.50 EM points |

This supports the bounded descriptive statement that the structured-over-sliding contrast remained positive across these two fixed pair-conditioned wordings.
It does not establish independent replication, a causal wording effect, or out-of-sample wording robustness because both surfaces reuse the same latent histories and B was conditioned on A for lexical disjointness.

## Matched Hybrid Versus Graphiti Result

| Contrast | Surface A | Surface B | Pooled paired A/B |
|---|---:|---:|---:|
| Hybrid minus Graphiti at 4K | +28.33 points, 95% CI [+18.33, +38.33] | +25.83 points, 95% CI [+15.00, +35.00] | +27.08 points, 95% CI [+19.17, +35.42] |
| Hybrid minus Graphiti at 16K | +19.17 points, 95% CI [+5.83, +31.67] | +8.33 points, 95% CI [+0.83, +16.67] | +13.75 points, 95% CI [+5.42, +21.25] |

This establishes that the hybrid-over-Graphiti ordering remained positive across these two fixed pair-conditioned wordings for this ingestion realization.
It does not establish independent or out-of-sample wording robustness because A and B share latent histories.
The Graphiti protocol still requires multiple independent ingestion builds to measure extraction variance.

## Authoritative Artifacts

The [confidence-interval CSV](persona-confidence-intervals.csv) contains Surface A,
Surface B, pooled, and available subgroup intervals from the matched eight-arm
run and the earlier fixed five-arm analysis. Estimates and 95% intervals are
paired exact-match or F1 differences in percentage points; `source_file`
identifies each source. The export preserves the saved estimates and bootstrap
settings without recomputing intervals.

- [Pair-conditioned A/B analysis](../results/persona_surface_generalization_v1/analysis.md)
- [Pair gate](../results/persona_surface_pair_gate.json)
- [Standalone Neo4j hybrid run](../results/persona_end_to_end_qwen35_4b_neurosym_v1/manifest.json)
- [Historical joint hybrid-versus-Graphiti run](../results/persona_joint_surface_a_build2/manifest.json)
- [Matched Kimi A/B hybrid-versus-Graphiti run](../results/persona_joint_kimi_ab_a100_build1/README.md)
