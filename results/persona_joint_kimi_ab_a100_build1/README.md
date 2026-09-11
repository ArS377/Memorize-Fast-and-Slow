# Matched Kimi A/B Hybrid-KG Versus Graphiti Benchmark

This run evaluates the same eight memory arms on both authenticated pair-conditioned Kimi surfaces.
Each surface contains 120 conditions over the same 12 latent histories and 960 Qwen3.5-4B generations.
Surface A and B differ in visible dialogue, query, and answer wording; they are paired surface realizations, not independent replications.

## Exact Match

| Surface | Hybrid KG 4K | Graphiti 4K | Hybrid minus Graphiti | Hybrid KG 16K | Graphiti 16K | Hybrid minus Graphiti |
|---|---:|---:|---:|---:|---:|---:|
| A | 69.17% | 40.83% | +28.33 points, 95% CI [+18.33, +38.33] | 71.67% | 52.50% | +19.17 points, 95% CI [+5.83, +31.67] |
| B | 65.83% | 40.00% | +25.83 points, 95% CI [+15.00, +35.00] | 66.67% | 58.33% | +8.33 points, 95% CI [+0.83, +16.67] |
| Pooled paired A/B | 67.50% | 40.42% | +27.08 points, 95% CI [+19.17, +35.42] | 69.17% | 55.42% | +13.75 points, 95% CI [+5.42, +21.25] |

The pooled bootstrap keeps both surface observations for each latent history in one of 12 resampled history clusters.
It therefore describes these two fixed wordings without treating them as 24 independent histories.

## Compute And Runtime

- Host: `scai3`, which has 8 NVIDIA A100-SXM4-40GB GPUs.
- GPUs used by this run: 2.
- GPU 6 served Graphiti extraction with Qwen3-4B through vLLM 0.21.0.
- GPU 7 ran direct Qwen3.5-4B answer generation with BF16 Transformers inference.
- BGE-small-en-v1.5 embeddings and prompt construction ran on CPU.
- Surface A wall time: approximately 55 minutes 14 seconds.
- Surface B wall time: approximately 55 minutes 7 seconds.
- Combined scored run wall time: approximately 1 hour 50 minutes 21 seconds.
- Measured `model.generate()` time was 29 minutes 22 seconds for A and 29 minutes 25 seconds for B.

The scored-run wall time excludes environment provisioning and the separate smoke/full preflights.

## Authentication

- Surface A generation-manifest SHA-256: `2cfdbf18705204a9ca7412e9e962ad3bb03bd08b2426f09270abbc387b047f65`.
- Surface B generation-manifest SHA-256: `986cbe8ce306e19169626d5b8729b92e120f45831a76b328cfa1e32401dfd121`.
- Pair-gate SHA-256: `d03540a6575c9d2967d8bfade7fef4e38a0951d98cb702420c06113dd228fb5b`.
- Analysis SHA-256: `9d7480d69bafc0b5c3e750f101d82ee4789832607aa18c9ca529e41c32731dc4`.
- Both surfaces used Qwen3.5-4B revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Each surface ingested 312 Graphiti episodes with zero episode failures and zero empty retrievals.
- All artifact hashes in both completed manifests were reverified after execution.

## Limitations

- This is one stochastic Graphiti ingestion realization per surface. Additional builds are required to estimate extraction variance.
- A and B share 12 latent histories, and B was conditioned on A for lexical disjointness. The result does not establish out-of-sample wording robustness.
- Each surface has 28 token-distance conditions where the nominal Scallop relation was absent despite valid underlying hybrid facts. The run does not establish complete Scallop-relation coverage at every token-distance checkpoint.
- Surface B dropped one temporally invalid Graphiti server candidate through the configured client-side point-in-time filter. All retained candidates satisfy the temporal predicate.
- Both manifests record the same dirty Git source identity. The evaluator script hash is preserved, but the complete dirty execution tree was not archived independently.

The authenticated pooled analysis is in [`analysis/analysis.json`](analysis/analysis.json).
