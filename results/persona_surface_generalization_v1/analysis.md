# Pooled Fixed-Assignment Analysis

Primary metric: exact match (EM). Secondary metric: token F1.

A+B uses 12 base-history bootstrap clusters; each sampled history retains both assignment replicas.

## Pooled Arm Scores

| Arm | EM | F1 | Rows | Histories |
|---|---:|---:|---:|---:|
| sliding_context_4096 | 0.350 | 0.377 | 240 | 12 |
| sliding_context_16384 | 0.617 | 0.658 | 240 | 12 |
| structured_memory_4096 | 0.679 | 0.695 | 240 | 12 |
| structured_memory_16384 | 0.713 | 0.727 | 240 | 12 |
| full_qwen_context | 0.721 | 0.731 | 240 | 12 |

## Per-Assignment Arm Scores

| Assignment | Arm | EM | F1 |
|---|---|---:|---:|
| A | sliding_context_4096 | 0.342 | 0.366 |
| A | sliding_context_16384 | 0.600 | 0.646 |
| A | structured_memory_4096 | 0.683 | 0.699 |
| A | structured_memory_16384 | 0.708 | 0.721 |
| A | full_qwen_context | 0.733 | 0.742 |
| B | sliding_context_4096 | 0.358 | 0.388 |
| B | sliding_context_16384 | 0.633 | 0.671 |
| B | structured_memory_4096 | 0.675 | 0.692 |
| B | structured_memory_16384 | 0.717 | 0.733 |
| B | full_qwen_context | 0.708 | 0.721 |
| v1 | sliding_context_4096 | 0.325 | 0.342 |
| v1 | sliding_context_16384 | 0.567 | 0.604 |
| v1 | structured_memory_4096 | 0.650 | 0.671 |
| v1 | structured_memory_16384 | 0.658 | 0.692 |
| v1 | full_qwen_context | 0.500 | 0.508 |

## pooled_A_B: Structured Minus Sliding

| Contrast | Stratum | EM delta [95% CI] | F1 delta [95% CI] |
|---|---|---:|---:|
| structured_memory_4096 - sliding_context_4096 | overall | 0.329 [0.296, 0.358] | 0.319 [0.283, 0.352] |
| structured_memory_4096 - sliding_context_4096 | delayed | 0.833 [0.729, 0.938] | 0.812 [0.667, 0.938] |
| structured_memory_4096 - sliding_context_4096 | source_present | 0.898 [0.841, 0.954] | 0.869 [0.799, 0.938] |
| structured_memory_4096 - sliding_context_4096 | source_absent | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| structured_memory_16384 - sliding_context_16384 | overall | 0.096 [0.054, 0.142] | 0.069 [0.042, 0.094] |
| structured_memory_16384 - sliding_context_16384 | delayed | 0.333 [0.208, 0.438] | 0.281 [0.156, 0.406] |
| structured_memory_16384 - sliding_context_16384 | source_present | 0.261 [0.144, 0.400] | 0.188 [0.109, 0.269] |
| structured_memory_16384 - sliding_context_16384 | source_absent | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |

## pooled_A_B: Full-Context Comparisons

| Contrast | Stratum | EM delta [95% CI] | F1 delta [95% CI] |
|---|---|---:|---:|
| full_qwen_context - sliding_context_16384 | overall | 0.104 [0.033, 0.175] | 0.073 [0.019, 0.125] |
| full_qwen_context - sliding_context_16384 | delayed | 0.333 [0.167, 0.479] | 0.271 [0.115, 0.427] |
| full_qwen_context - sliding_context_16384 | source_present | 0.250 [0.085, 0.407] | 0.170 [0.039, 0.287] |
| full_qwen_context - sliding_context_16384 | source_absent | 0.020 [-0.007, 0.054] | 0.016 [-0.010, 0.046] |
| full_qwen_context - structured_memory_16384 | overall | 0.008 [-0.033, 0.042] | 0.004 [-0.033, 0.037] |
| full_qwen_context - structured_memory_16384 | delayed | 0.000 [-0.062, 0.062] | -0.010 [-0.073, 0.052] |
| full_qwen_context - structured_memory_16384 | source_present | -0.011 [-0.091, 0.048] | -0.017 [-0.101, 0.044] |
| full_qwen_context - structured_memory_16384 | source_absent | 0.020 [-0.013, 0.054] | 0.016 [-0.012, 0.045] |

## A: Structured Minus Sliding

| Contrast | Stratum | EM delta [95% CI] | F1 delta [95% CI] |
|---|---|---:|---:|
| structured_memory_4096 - sliding_context_4096 | overall | 0.342 [0.300, 0.383] | 0.333 [0.287, 0.379] |
| structured_memory_4096 - sliding_context_4096 | delayed | 0.875 [0.750, 1.000] | 0.854 [0.708, 1.000] |
| structured_memory_4096 - sliding_context_4096 | source_present | 0.932 [0.864, 1.000] | 0.909 [0.826, 0.979] |
| structured_memory_4096 - sliding_context_4096 | source_absent | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| structured_memory_16384 - sliding_context_16384 | overall | 0.108 [0.058, 0.167] | 0.075 [0.046, 0.104] |
| structured_memory_16384 - sliding_context_16384 | delayed | 0.375 [0.250, 0.500] | 0.292 [0.167, 0.396] |
| structured_memory_16384 - sliding_context_16384 | source_present | 0.295 [0.159, 0.462] | 0.205 [0.120, 0.287] |
| structured_memory_16384 - sliding_context_16384 | source_absent | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |

## A: Full-Context Comparisons

| Contrast | Stratum | EM delta [95% CI] | F1 delta [95% CI] |
|---|---|---:|---:|
| full_qwen_context - sliding_context_16384 | overall | 0.133 [0.067, 0.225] | 0.096 [0.046, 0.150] |
| full_qwen_context - sliding_context_16384 | delayed | 0.417 [0.292, 0.500] | 0.333 [0.229, 0.438] |
| full_qwen_context - sliding_context_16384 | source_present | 0.318 [0.191, 0.467] | 0.227 [0.152, 0.302] |
| full_qwen_context - sliding_context_16384 | source_absent | 0.026 [-0.027, 0.096] | 0.020 [-0.027, 0.079] |
| full_qwen_context - structured_memory_16384 | overall | 0.025 [-0.017, 0.067] | 0.021 [-0.013, 0.058] |
| full_qwen_context - structured_memory_16384 | delayed | 0.042 [0.000, 0.125] | 0.042 [0.000, 0.125] |
| full_qwen_context - structured_memory_16384 | source_present | 0.023 [0.000, 0.070] | 0.023 [0.000, 0.070] |
| full_qwen_context - structured_memory_16384 | source_absent | 0.026 [-0.027, 0.096] | 0.020 [-0.026, 0.078] |

## B: Structured Minus Sliding

| Contrast | Stratum | EM delta [95% CI] | F1 delta [95% CI] |
|---|---|---:|---:|
| structured_memory_4096 - sliding_context_4096 | overall | 0.317 [0.283, 0.350] | 0.304 [0.271, 0.342] |
| structured_memory_4096 - sliding_context_4096 | delayed | 0.792 [0.667, 0.917] | 0.771 [0.604, 0.917] |
| structured_memory_4096 - sliding_context_4096 | source_present | 0.864 [0.776, 0.951] | 0.830 [0.723, 0.933] |
| structured_memory_4096 - sliding_context_4096 | source_absent | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| structured_memory_16384 - sliding_context_16384 | overall | 0.083 [0.050, 0.125] | 0.062 [0.033, 0.092] |
| structured_memory_16384 - sliding_context_16384 | delayed | 0.292 [0.167, 0.417] | 0.271 [0.125, 0.417] |
| structured_memory_16384 - sliding_context_16384 | source_present | 0.227 [0.116, 0.348] | 0.170 [0.085, 0.263] |
| structured_memory_16384 - sliding_context_16384 | source_absent | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |

## B: Full-Context Comparisons

| Contrast | Stratum | EM delta [95% CI] | F1 delta [95% CI] |
|---|---|---:|---:|
| full_qwen_context - sliding_context_16384 | overall | 0.075 [-0.025, 0.158] | 0.050 [-0.050, 0.125] |
| full_qwen_context - sliding_context_16384 | delayed | 0.250 [-0.042, 0.500] | 0.208 [-0.083, 0.458] |
| full_qwen_context - sliding_context_16384 | source_present | 0.182 [-0.083, 0.390] | 0.114 [-0.135, 0.297] |
| full_qwen_context - sliding_context_16384 | source_absent | 0.013 [0.000, 0.039] | 0.013 [0.000, 0.039] |
| full_qwen_context - structured_memory_16384 | overall | -0.008 [-0.092, 0.050] | -0.013 [-0.096, 0.054] |
| full_qwen_context - structured_memory_16384 | delayed | -0.042 [-0.250, 0.125] | -0.062 [-0.271, 0.104] |
| full_qwen_context - structured_memory_16384 | source_present | -0.045 [-0.267, 0.093] | -0.057 [-0.272, 0.089] |
| full_qwen_context - structured_memory_16384 | source_absent | 0.013 [0.000, 0.039] | 0.013 [0.000, 0.040] |

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

CONTAMINATION WARNING: A and B are deterministic surface derivations of the same v1 histories, facts, and evaluation structure, not independent replications. These difference-in-paired-differences estimates are descriptive only and do not support causal, independence, or out-of-sample generalization claims.

| Budget | Stratum | EM | F1 |
|---:|---|---:|---:|
| 16384 | delayed | 0.000 | -0.073 |
| 16384 | overall | 0.004 | -0.019 |
| 16384 | source_absent | 0.000 | 0.000 |
| 16384 | source_present | 0.011 | -0.051 |
| 4096 | delayed | 0.000 | -0.042 |
| 4096 | overall | 0.004 | -0.010 |
| 4096 | source_absent | 0.000 | 0.000 |
| 4096 | source_present | 0.011 | -0.028 |

## Authentication

- Result JSON SHA-256: `d451a0f9329bc76acf6b1c3b423e1321f973e2c6938e7bb37056898fdd9de801`
- A manifest SHA-256: `116bfebb1b968286ef12ae7ab15f821b5efac3e3b85667fa3b59ac5cf1629db2`
- A predictions SHA-256: `e54a650705adb5c479c1232530a50f851e5f3e90840417c51532347779052f1d`
- B manifest SHA-256: `42a73474ae513ca68dabbc6920d8658db30d7f910d7d8d11146faed88ab9956a`
- B predictions SHA-256: `316c860cfcd6591312c0dbcf2baa8e8cbb4852329a154d80d045e32e2e2cc68f`
- v1 manifest SHA-256: `b53a29f137de639fd09656b2efe681011f1083034ca7842cb270b19ed848f9dd`
- v1 predictions SHA-256: `f6c4afe028d54c37e85694649ab7d4d8827db16201edda018855e2e7f0e5957b`

## Bounded Claims

- Intervals quantify sampling variation across the 12 observed base histories only.
- Exact match is primary; token F1 is secondary.
- Source-absent structured-minus-sliding effects are required to be exactly zero.
- Hashes and invariants authenticate internal consistency and corpus-semantic binding; they do not prove execution or provide hostile artifact-origin attestation.
- Exact A/B gold semantics are bound to authenticated v1 corpus checkpoints through causal resolution and each assignment corpus's canonical surface mapping.
- CONTAMINATION WARNING: A and B are deterministic surface derivations of the same v1 histories, facts, and evaluation structure, not independent replications. These difference-in-paired-differences estimates are descriptive only and do not support causal, independence, or out-of-sample generalization claims.
