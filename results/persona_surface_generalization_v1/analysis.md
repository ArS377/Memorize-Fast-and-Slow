# Pooled Fixed-Assignment Analysis

Primary metric: exact match (EM). Secondary metric: token F1.

A+B uses 12 base-history bootstrap clusters; each sampled history retains both assignment replicas.

## Pooled Arm Scores

| Arm | EM | F1 | Rows | Histories |
|---|---:|---:|---:|---:|
| sliding_context_4096 | 0.304 | 0.336 | 240 | 12 |
| sliding_context_16384 | 0.529 | 0.574 | 240 | 12 |
| structured_memory_4096 | 0.608 | 0.628 | 240 | 12 |
| structured_memory_16384 | 0.621 | 0.646 | 240 | 12 |
| full_qwen_context | 0.575 | 0.615 | 240 | 12 |

## Per-Assignment Arm Scores

| Assignment | Arm | EM | F1 |
|---|---|---:|---:|
| A | sliding_context_4096 | 0.325 | 0.358 |
| A | sliding_context_16384 | 0.500 | 0.556 |
| A | structured_memory_4096 | 0.625 | 0.645 |
| A | structured_memory_16384 | 0.608 | 0.640 |
| A | full_qwen_context | 0.550 | 0.602 |
| B | sliding_context_4096 | 0.283 | 0.314 |
| B | sliding_context_16384 | 0.558 | 0.592 |
| B | structured_memory_4096 | 0.592 | 0.610 |
| B | structured_memory_16384 | 0.633 | 0.652 |
| B | full_qwen_context | 0.600 | 0.628 |
| v1 | sliding_context_4096 | 0.325 | 0.342 |
| v1 | sliding_context_16384 | 0.567 | 0.604 |
| v1 | structured_memory_4096 | 0.650 | 0.671 |
| v1 | structured_memory_16384 | 0.658 | 0.692 |
| v1 | full_qwen_context | 0.500 | 0.508 |

## pooled_A_B: Structured Minus Sliding

| Contrast | Stratum | EM delta [95% CI] | F1 delta [95% CI] |
|---|---|---:|---:|
| structured_memory_4096 - sliding_context_4096 | overall | 0.304 [0.267, 0.333] | 0.291 [0.258, 0.318] |
| structured_memory_4096 - sliding_context_4096 | delayed | 0.729 [0.604, 0.854] | 0.729 [0.604, 0.854] |
| structured_memory_4096 - sliding_context_4096 | source_present | 0.830 [0.755, 0.907] | 0.794 [0.708, 0.871] |
| structured_memory_4096 - sliding_context_4096 | source_absent | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| structured_memory_16384 - sliding_context_16384 | overall | 0.092 [0.042, 0.142] | 0.072 [0.039, 0.100] |
| structured_memory_16384 - sliding_context_16384 | delayed | 0.271 [0.104, 0.417] | 0.250 [0.115, 0.375] |
| structured_memory_16384 - sliding_context_16384 | source_present | 0.250 [0.110, 0.384] | 0.196 [0.104, 0.277] |
| structured_memory_16384 - sliding_context_16384 | source_absent | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |

## pooled_A_B: Full-Context Comparisons

| Contrast | Stratum | EM delta [95% CI] | F1 delta [95% CI] |
|---|---|---:|---:|
| full_qwen_context - sliding_context_16384 | overall | 0.046 [0.000, 0.083] | 0.041 [0.001, 0.074] |
| full_qwen_context - sliding_context_16384 | delayed | 0.229 [0.062, 0.375] | 0.242 [0.102, 0.380] |
| full_qwen_context - sliding_context_16384 | source_present | 0.170 [0.095, 0.250] | 0.157 [0.092, 0.228] |
| full_qwen_context - sliding_context_16384 | source_absent | -0.026 [-0.067, 0.000] | -0.026 [-0.068, 0.000] |
| full_qwen_context - structured_memory_16384 | overall | -0.046 [-0.092, 0.000] | -0.031 [-0.059, -0.003] |
| full_qwen_context - structured_memory_16384 | delayed | -0.042 [-0.125, 0.042] | -0.008 [-0.051, 0.036] |
| full_qwen_context - structured_memory_16384 | source_present | -0.080 [-0.196, 0.025] | -0.039 [-0.105, 0.022] |
| full_qwen_context - structured_memory_16384 | source_absent | -0.026 [-0.068, 0.000] | -0.026 [-0.069, 0.000] |

## A: Structured Minus Sliding

| Contrast | Stratum | EM delta [95% CI] | F1 delta [95% CI] |
|---|---|---:|---:|
| structured_memory_4096 - sliding_context_4096 | overall | 0.300 [0.275, 0.325] | 0.287 [0.265, 0.307] |
| structured_memory_4096 - sliding_context_4096 | delayed | 0.708 [0.583, 0.833] | 0.708 [0.583, 0.833] |
| structured_memory_4096 - sliding_context_4096 | source_present | 0.818 [0.745, 0.902] | 0.782 [0.700, 0.868] |
| structured_memory_4096 - sliding_context_4096 | source_absent | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| structured_memory_16384 - sliding_context_16384 | overall | 0.108 [0.058, 0.158] | 0.084 [0.053, 0.111] |
| structured_memory_16384 - sliding_context_16384 | delayed | 0.292 [0.167, 0.417] | 0.271 [0.146, 0.396] |
| structured_memory_16384 - sliding_context_16384 | source_present | 0.295 [0.167, 0.435] | 0.228 [0.146, 0.307] |
| structured_memory_16384 - sliding_context_16384 | source_absent | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |

## A: Full-Context Comparisons

| Contrast | Stratum | EM delta [95% CI] | F1 delta [95% CI] |
|---|---|---:|---:|
| full_qwen_context - sliding_context_16384 | overall | 0.050 [0.008, 0.092] | 0.046 [0.011, 0.078] |
| full_qwen_context - sliding_context_16384 | delayed | 0.208 [0.000, 0.375] | 0.222 [0.056, 0.375] |
| full_qwen_context - sliding_context_16384 | source_present | 0.159 [0.043, 0.279] | 0.148 [0.052, 0.245] |
| full_qwen_context - sliding_context_16384 | source_absent | -0.013 [-0.039, 0.000] | -0.013 [-0.039, 0.000] |
| full_qwen_context - structured_memory_16384 | overall | -0.058 [-0.125, 0.000] | -0.037 [-0.079, 0.000] |
| full_qwen_context - structured_memory_16384 | delayed | -0.083 [-0.208, 0.000] | -0.049 [-0.125, 0.000] |
| full_qwen_context - structured_memory_16384 | source_present | -0.136 [-0.326, 0.000] | -0.080 [-0.189, 0.000] |
| full_qwen_context - structured_memory_16384 | source_absent | -0.013 [-0.039, 0.000] | -0.013 [-0.040, 0.000] |

## B: Structured Minus Sliding

| Contrast | Stratum | EM delta [95% CI] | F1 delta [95% CI] |
|---|---|---:|---:|
| structured_memory_4096 - sliding_context_4096 | overall | 0.308 [0.267, 0.350] | 0.296 [0.250, 0.333] |
| structured_memory_4096 - sliding_context_4096 | delayed | 0.750 [0.625, 0.875] | 0.750 [0.624, 0.875] |
| structured_memory_4096 - sliding_context_4096 | source_present | 0.841 [0.750, 0.932] | 0.807 [0.707, 0.902] |
| structured_memory_4096 - sliding_context_4096 | source_absent | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| structured_memory_16384 - sliding_context_16384 | overall | 0.075 [0.008, 0.142] | 0.060 [0.020, 0.096] |
| structured_memory_16384 - sliding_context_16384 | delayed | 0.250 [0.042, 0.417] | 0.229 [0.083, 0.375] |
| structured_memory_16384 - sliding_context_16384 | source_present | 0.205 [0.023, 0.383] | 0.164 [0.049, 0.263] |
| structured_memory_16384 - sliding_context_16384 | source_absent | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |

## B: Full-Context Comparisons

| Contrast | Stratum | EM delta [95% CI] | F1 delta [95% CI] |
|---|---|---:|---:|
| full_qwen_context - sliding_context_16384 | overall | 0.042 [-0.042, 0.117] | 0.036 [-0.031, 0.089] |
| full_qwen_context - sliding_context_16384 | delayed | 0.250 [0.083, 0.417] | 0.263 [0.113, 0.408] |
| full_qwen_context - sliding_context_16384 | source_present | 0.182 [0.044, 0.333] | 0.166 [0.064, 0.266] |
| full_qwen_context - sliding_context_16384 | source_absent | -0.039 [-0.118, 0.000] | -0.039 [-0.118, 0.000] |
| full_qwen_context - structured_memory_16384 | overall | -0.033 [-0.108, 0.033] | -0.024 [-0.082, 0.020] |
| full_qwen_context - structured_memory_16384 | delayed | 0.000 [-0.125, 0.125] | 0.033 [-0.050, 0.113] |
| full_qwen_context - structured_memory_16384 | source_present | -0.023 [-0.200, 0.133] | 0.002 [-0.103, 0.087] |
| full_qwen_context - structured_memory_16384 | source_absent | -0.039 [-0.120, 0.000] | -0.039 [-0.118, 0.000] |

## v1: Structured Minus Sliding

| Contrast | Stratum | EM delta [95% CI] | F1 delta [95% CI] |
|---|---|---:|---:|
| structured_memory_4096 - sliding_context_4096 | overall | 0.325 [0.292, 0.358] | 0.329 [0.296, 0.362] |
| structured_memory_4096 - sliding_context_4096 | delayed | 0.833 [0.708, 0.958] | 0.854 [0.729, 0.958] |
| structured_memory_4096 - sliding_context_4096 | source_present | 0.886 [0.795, 0.976] | 0.898 [0.806, 0.977] |
| structured_memory_4096 - sliding_context_4096 | source_absent | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| structured_memory_16384 - sliding_context_16384 | overall | 0.092 [0.033, 0.158] | 0.087 [0.033, 0.150] |
| structured_memory_16384 - sliding_context_16384 | delayed | 0.333 [0.208, 0.458] | 0.354 [0.229, 0.479] |
| structured_memory_16384 - sliding_context_16384 | source_present | 0.250 [0.091, 0.426] | 0.239 [0.095, 0.400] |
| structured_memory_16384 - sliding_context_16384 | source_absent | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |

## v1: Full-Context Comparisons

| Contrast | Stratum | EM delta [95% CI] | F1 delta [95% CI] |
|---|---|---:|---:|
| full_qwen_context - sliding_context_16384 | overall | -0.067 [-0.192, 0.050] | -0.096 [-0.217, 0.017] |
| full_qwen_context - sliding_context_16384 | delayed | 0.125 [-0.042, 0.292] | 0.083 [-0.104, 0.271] |
| full_qwen_context - sliding_context_16384 | source_present | -0.023 [-0.205, 0.163] | -0.068 [-0.256, 0.115] |
| full_qwen_context - sliding_context_16384 | source_absent | -0.092 [-0.203, 0.013] | -0.112 [-0.227, -0.014] |
| full_qwen_context - structured_memory_16384 | overall | -0.158 [-0.250, -0.067] | -0.183 [-0.271, -0.104] |
| full_qwen_context - structured_memory_16384 | delayed | -0.208 [-0.333, -0.083] | -0.271 [-0.396, -0.146] |
| full_qwen_context - structured_memory_16384 | source_present | -0.273 [-0.409, -0.149] | -0.307 [-0.438, -0.173] |
| full_qwen_context - structured_memory_16384 | source_absent | -0.092 [-0.198, 0.013] | -0.112 [-0.220, -0.013] |

## Descriptive Difference-in-Paired-Differences vs v1

CONTAMINATION WARNING: A and B use separately Kimi-generated, pair-conditioned surfaces and dialogue over the same v1 histories, facts, and evaluation structure, not independent latent replications. These difference-in-paired-differences estimates are descriptive only and do not support causal, independence, or out-of-sample generalization claims.

| Budget | Stratum | EM | F1 |
|---:|---|---:|---:|
| 16384 | delayed | -0.062 | -0.104 |
| 16384 | overall | 0.000 | -0.016 |
| 16384 | source_absent | 0.000 | 0.000 |
| 16384 | source_present | 0.000 | -0.043 |
| 4096 | delayed | -0.104 | -0.125 |
| 4096 | overall | -0.021 | -0.038 |
| 4096 | source_absent | 0.000 | 0.000 |
| 4096 | source_present | -0.057 | -0.103 |

## Authentication

- Result JSON SHA-256: `d298adfe2bd4566284d083159f339a5a8c992dac21667834d13b3204ed6b4c84`
- A manifest SHA-256: `bf33313ea3ee1adc9ea9fe5c89a0715afd89a1ec47be95fe9f1300ddbefd0a2b`
- A predictions SHA-256: `38fd82c092ac796913b807fbaf29ad92cab8eb71a9d84f36f62afbb3c5a4322d`
- B manifest SHA-256: `d9485e32163fef9a3ec178a4d581cdd1e7f9847cf64c6a0df6f9f2ca9936874a`
- B predictions SHA-256: `c66ae165f7e444b908672723287b847b940f6e05175feabe149fa2100bde9683`
- v1 manifest SHA-256: `b53a29f137de639fd09656b2efe681011f1083034ca7842cb270b19ed848f9dd`
- v1 predictions SHA-256: `f6c4afe028d54c37e85694649ab7d4d8827db16201edda018855e2e7f0e5957b`

## Bounded Claims

- Intervals quantify sampling variation across the 12 observed base histories only.
- Exact match is primary; token F1 is secondary.
- Source-absent structured-minus-sliding effects are required to be exactly zero.
- Hashes and invariants authenticate internal consistency and corpus-semantic binding; they do not prove execution or provide hostile artifact-origin attestation.
- Exact A/B gold semantics are bound to authenticated v1 corpus checkpoints through causal resolution and each assignment corpus's canonical surface mapping.
- CONTAMINATION WARNING: A and B use separately Kimi-generated, pair-conditioned surfaces and dialogue over the same v1 histories, facts, and evaluation structure, not independent latent replications. These difference-in-paired-differences estimates are descriptive only and do not support causal, independence, or out-of-sample generalization claims.
