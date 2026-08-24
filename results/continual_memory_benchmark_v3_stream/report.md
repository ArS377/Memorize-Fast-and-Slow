# Continual Memory Benchmark

This is continual memory under interleaved tasks without online parameter learning, not generic long-context question answering.

All baselines use deterministic structured resolution after memory selection; this is an oracle reasoning control, not a language-model reasoning score.

Grounded answer accuracy requires every declared event and fact evidence group for both known and UNKNOWN answers.

Cross-task interference counts only failures absent from the same method, history, and checkpoint under the matched zero-distractor tier, divided by treated comparisons whose zero-distractor control was correct.

Model parameter tiers are metadata only and are not evaluated.
No online model weight updates occur.

## Baselines

| Method | Grounded accuracy | Raw answer accuracy | Exact evidence hit | Evidence recall | Stale intrusion | Retraction compliance | Causal cross-task interference |
|---|---:|---:|---:|---:|---:|---:|---:|
| full_structured_memory | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 |
| recency | 0.2615 | 0.4538 | 0.2615 | 0.3041 | 0.0000 | 0.2000 | 0.7778 |
| bm25 | 0.6118 | 0.7405 | 0.6118 | 0.7138 | 0.0651 | 0.4067 | 0.0952 |
| sliding_context:window_4k | 0.5077 | 0.6308 | 0.5077 | 0.5391 | 0.0000 | 0.5000 | 0.6154 |
| sliding_context:window_16k | 0.7077 | 0.8000 | 0.7077 | 0.7430 | 0.0000 | 0.7000 | 0.3654 |
| sliding_context:window_64k | 0.9077 | 0.9538 | 0.9077 | 0.9430 | 0.0000 | 0.9000 | 0.1154 |
| sliding_context:window_128k | 0.9538 | 0.9692 | 0.9538 | 0.9673 | 0.0000 | 0.9000 | 0.0577 |
| dense | 0.4687 | 0.6015 | 0.4687 | 0.5648 | 0.0928 | 0.4967 | 0.2522 |
| sliding_context:window_4k+scallop_injection | 0.9967 | 0.9967 | 0.9967 | 0.9967 | 0.0033 | 0.0000 | 0.0042 |
| sliding_context:window_4k+reverse_ranked_injection | 0.7067 | 0.7067 | 0.7067 | 0.7067 | 0.0133 | 0.0000 | 0.3667 |
| sliding_context:window_4k+scallop_change_only | 0.6967 | 0.7000 | 0.6967 | 0.6983 | 0.3000 | 0.0000 | 0.3792 |
| sliding_context:window_4k+scallop_incongruity_only | 0.5967 | 0.5967 | 0.5967 | 0.5967 | 0.0000 | 0.0000 | 0.5042 |
| sliding_context:window_16k+scallop_injection | 0.9967 | 0.9967 | 0.9967 | 0.9967 | 0.0033 | 0.0000 | 0.0042 |
| sliding_context:window_16k+reverse_ranked_injection | 0.7067 | 0.7067 | 0.7067 | 0.7067 | 0.0133 | 0.0000 | 0.3667 |
| sliding_context:window_16k+scallop_change_only | 0.8000 | 0.8000 | 0.8000 | 0.8000 | 0.2000 | 0.0000 | 0.2500 |
| sliding_context:window_16k+scallop_incongruity_only | 0.6967 | 0.7967 | 0.6967 | 0.7217 | 0.0000 | 0.0000 | 0.3792 |
| sliding_context:window_64k+scallop_injection | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| sliding_context:window_64k+reverse_ranked_injection | 0.7467 | 0.8000 | 0.7467 | 0.7600 | 0.0000 | 0.0000 | 0.3167 |
| sliding_context:window_64k+scallop_change_only | 0.9000 | 0.9000 | 0.9000 | 0.9000 | 0.1000 | 0.0000 | 0.1250 |
| sliding_context:window_64k+scallop_incongruity_only | 0.8000 | 0.9000 | 0.8000 | 0.8250 | 0.0000 | 0.0000 | 0.2500 |
| sliding_context:window_128k+scallop_injection | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| sliding_context:window_128k+reverse_ranked_injection | 0.8000 | 0.8000 | 0.8000 | 0.8000 | 0.0000 | 0.0000 | 0.2500 |
| sliding_context:window_128k+scallop_change_only | 0.9000 | 0.9000 | 0.9000 | 0.9000 | 0.1000 | 0.0000 | 0.1250 |
| sliding_context:window_128k+scallop_incongruity_only | 0.9000 | 0.9000 | 0.9000 | 0.9000 | 0.0000 | 0.0000 | 0.1250 |

## Retention By Interference

| Tier | Method | Grounded accuracy | Exact evidence hit |
|---|---|---:|---:|
| no_interference | full_structured_memory | 1.0000 | 1.0000 |
| no_interference | recency | 0.6923 | 0.6923 |
| no_interference | bm25 | 0.5385 | 0.5385 |
| no_interference | sliding_context:window_4k | 1.0000 | 1.0000 |
| no_interference | sliding_context:window_16k | 1.0000 | 1.0000 |
| no_interference | sliding_context:window_64k | 1.0000 | 1.0000 |
| no_interference | sliding_context:window_128k | 1.0000 | 1.0000 |
| no_interference | dense | 0.5872 | 0.5872 |
| no_interference | sliding_context:window_4k+scallop_injection | 1.0000 | 1.0000 |
| no_interference | sliding_context:window_4k+reverse_ranked_injection | 1.0000 | 1.0000 |
| no_interference | sliding_context:window_4k+scallop_change_only | 1.0000 | 1.0000 |
| no_interference | sliding_context:window_4k+scallop_incongruity_only | 1.0000 | 1.0000 |
| no_interference | sliding_context:window_16k+scallop_injection | 1.0000 | 1.0000 |
| no_interference | sliding_context:window_16k+reverse_ranked_injection | 1.0000 | 1.0000 |
| no_interference | sliding_context:window_16k+scallop_change_only | 1.0000 | 1.0000 |
| no_interference | sliding_context:window_16k+scallop_incongruity_only | 1.0000 | 1.0000 |
| no_interference | sliding_context:window_64k+scallop_injection | 1.0000 | 1.0000 |
| no_interference | sliding_context:window_64k+reverse_ranked_injection | 1.0000 | 1.0000 |
| no_interference | sliding_context:window_64k+scallop_change_only | 1.0000 | 1.0000 |
| no_interference | sliding_context:window_64k+scallop_incongruity_only | 1.0000 | 1.0000 |
| no_interference | sliding_context:window_128k+scallop_injection | 1.0000 | 1.0000 |
| no_interference | sliding_context:window_128k+reverse_ranked_injection | 1.0000 | 1.0000 |
| no_interference | sliding_context:window_128k+scallop_change_only | 1.0000 | 1.0000 |
| no_interference | sliding_context:window_128k+scallop_incongruity_only | 1.0000 | 1.0000 |
| light_interference | full_structured_memory | 1.0000 | 1.0000 |
| light_interference | recency | 0.6154 | 0.6154 |
| light_interference | bm25 | 0.5513 | 0.5513 |
| light_interference | sliding_context:window_4k | 0.9231 | 0.9231 |
| light_interference | sliding_context:window_16k | 1.0000 | 1.0000 |
| light_interference | sliding_context:window_64k | 1.0000 | 1.0000 |
| light_interference | sliding_context:window_128k | 1.0000 | 1.0000 |
| light_interference | dense | 0.5436 | 0.5436 |
| light_interference | sliding_context:window_4k+scallop_injection | 1.0000 | 1.0000 |
| light_interference | sliding_context:window_4k+reverse_ranked_injection | 1.0000 | 1.0000 |
| light_interference | sliding_context:window_4k+scallop_change_only | 0.9833 | 0.9833 |
| light_interference | sliding_context:window_4k+scallop_incongruity_only | 0.5000 | 0.5000 |
| light_interference | sliding_context:window_16k+scallop_injection | 1.0000 | 1.0000 |
| light_interference | sliding_context:window_16k+reverse_ranked_injection | 1.0000 | 1.0000 |
| light_interference | sliding_context:window_16k+scallop_change_only | 1.0000 | 1.0000 |
| light_interference | sliding_context:window_16k+scallop_incongruity_only | 1.0000 | 1.0000 |
| light_interference | sliding_context:window_64k+scallop_injection | 1.0000 | 1.0000 |
| light_interference | sliding_context:window_64k+reverse_ranked_injection | 1.0000 | 1.0000 |
| light_interference | sliding_context:window_64k+scallop_change_only | 1.0000 | 1.0000 |
| light_interference | sliding_context:window_64k+scallop_incongruity_only | 1.0000 | 1.0000 |
| light_interference | sliding_context:window_128k+scallop_injection | 1.0000 | 1.0000 |
| light_interference | sliding_context:window_128k+reverse_ranked_injection | 1.0000 | 1.0000 |
| light_interference | sliding_context:window_128k+scallop_change_only | 1.0000 | 1.0000 |
| light_interference | sliding_context:window_128k+scallop_incongruity_only | 1.0000 | 1.0000 |
| heavy_interference | full_structured_memory | 1.0000 | 1.0000 |
| heavy_interference | recency | 0.0000 | 0.0000 |
| heavy_interference | bm25 | 0.6487 | 0.6487 |
| heavy_interference | sliding_context:window_4k | 0.6154 | 0.6154 |
| heavy_interference | sliding_context:window_16k | 0.9231 | 0.9231 |
| heavy_interference | sliding_context:window_64k | 1.0000 | 1.0000 |
| heavy_interference | sliding_context:window_128k | 1.0000 | 1.0000 |
| heavy_interference | dense | 0.4744 | 0.4744 |
| heavy_interference | sliding_context:window_4k+scallop_injection | 1.0000 | 1.0000 |
| heavy_interference | sliding_context:window_4k+reverse_ranked_injection | 1.0000 | 1.0000 |
| heavy_interference | sliding_context:window_4k+scallop_change_only | 0.5000 | 0.5000 |
| heavy_interference | sliding_context:window_4k+scallop_incongruity_only | 0.5000 | 0.5000 |
| heavy_interference | sliding_context:window_16k+scallop_injection | 1.0000 | 1.0000 |
| heavy_interference | sliding_context:window_16k+reverse_ranked_injection | 1.0000 | 1.0000 |
| heavy_interference | sliding_context:window_16k+scallop_change_only | 1.0000 | 1.0000 |
| heavy_interference | sliding_context:window_16k+scallop_incongruity_only | 0.5000 | 0.5000 |
| heavy_interference | sliding_context:window_64k+scallop_injection | 1.0000 | 1.0000 |
| heavy_interference | sliding_context:window_64k+reverse_ranked_injection | 1.0000 | 1.0000 |
| heavy_interference | sliding_context:window_64k+scallop_change_only | 1.0000 | 1.0000 |
| heavy_interference | sliding_context:window_64k+scallop_incongruity_only | 1.0000 | 1.0000 |
| heavy_interference | sliding_context:window_128k+scallop_injection | 1.0000 | 1.0000 |
| heavy_interference | sliding_context:window_128k+reverse_ranked_injection | 1.0000 | 1.0000 |
| heavy_interference | sliding_context:window_128k+scallop_change_only | 1.0000 | 1.0000 |
| heavy_interference | sliding_context:window_128k+scallop_incongruity_only | 1.0000 | 1.0000 |
| extreme_interference | full_structured_memory | 1.0000 | 1.0000 |
| extreme_interference | recency | 0.0000 | 0.0000 |
| extreme_interference | bm25 | 0.6282 | 0.6282 |
| extreme_interference | sliding_context:window_4k | 0.0000 | 0.0000 |
| extreme_interference | sliding_context:window_16k | 0.6154 | 0.6154 |
| extreme_interference | sliding_context:window_64k | 0.9231 | 0.9231 |
| extreme_interference | sliding_context:window_128k | 1.0000 | 1.0000 |
| extreme_interference | dense | 0.4026 | 0.4026 |
| extreme_interference | sliding_context:window_4k+scallop_injection | 0.9833 | 0.9833 |
| extreme_interference | sliding_context:window_4k+reverse_ranked_injection | 0.5333 | 0.5333 |
| extreme_interference | sliding_context:window_4k+scallop_change_only | 0.5000 | 0.5000 |
| extreme_interference | sliding_context:window_4k+scallop_incongruity_only | 0.4833 | 0.4833 |
| extreme_interference | sliding_context:window_16k+scallop_injection | 0.9833 | 0.9833 |
| extreme_interference | sliding_context:window_16k+reverse_ranked_injection | 0.5333 | 0.5333 |
| extreme_interference | sliding_context:window_16k+scallop_change_only | 0.5000 | 0.5000 |
| extreme_interference | sliding_context:window_16k+scallop_incongruity_only | 0.4833 | 0.4833 |
| extreme_interference | sliding_context:window_64k+scallop_injection | 1.0000 | 1.0000 |
| extreme_interference | sliding_context:window_64k+reverse_ranked_injection | 0.7333 | 0.7333 |
| extreme_interference | sliding_context:window_64k+scallop_change_only | 1.0000 | 1.0000 |
| extreme_interference | sliding_context:window_64k+scallop_incongruity_only | 0.5000 | 0.5000 |
| extreme_interference | sliding_context:window_128k+scallop_injection | 1.0000 | 1.0000 |
| extreme_interference | sliding_context:window_128k+reverse_ranked_injection | 1.0000 | 1.0000 |
| extreme_interference | sliding_context:window_128k+scallop_change_only | 1.0000 | 1.0000 |
| extreme_interference | sliding_context:window_128k+scallop_incongruity_only | 1.0000 | 1.0000 |
| deep_context_rot | full_structured_memory | 1.0000 | 1.0000 |
| deep_context_rot | recency | 0.0000 | 0.0000 |
| deep_context_rot | bm25 | 0.6923 | 0.6923 |
| deep_context_rot | sliding_context:window_4k | 0.0000 | 0.0000 |
| deep_context_rot | sliding_context:window_16k | 0.0000 | 0.0000 |
| deep_context_rot | sliding_context:window_64k | 0.6154 | 0.6154 |
| deep_context_rot | sliding_context:window_128k | 0.7692 | 0.7692 |
| deep_context_rot | dense | 0.3359 | 0.3359 |
| deep_context_rot | sliding_context:window_4k+scallop_injection | 1.0000 | 1.0000 |
| deep_context_rot | sliding_context:window_4k+reverse_ranked_injection | 0.0000 | 0.0000 |
| deep_context_rot | sliding_context:window_4k+scallop_change_only | 0.5000 | 0.5000 |
| deep_context_rot | sliding_context:window_4k+scallop_incongruity_only | 0.5000 | 0.5000 |
| deep_context_rot | sliding_context:window_16k+scallop_injection | 1.0000 | 1.0000 |
| deep_context_rot | sliding_context:window_16k+reverse_ranked_injection | 0.0000 | 0.0000 |
| deep_context_rot | sliding_context:window_16k+scallop_change_only | 0.5000 | 0.5000 |
| deep_context_rot | sliding_context:window_16k+scallop_incongruity_only | 0.5000 | 0.5000 |
| deep_context_rot | sliding_context:window_64k+scallop_injection | 1.0000 | 1.0000 |
| deep_context_rot | sliding_context:window_64k+reverse_ranked_injection | 0.0000 | 0.0000 |
| deep_context_rot | sliding_context:window_64k+scallop_change_only | 0.5000 | 0.5000 |
| deep_context_rot | sliding_context:window_64k+scallop_incongruity_only | 0.5000 | 0.5000 |
| deep_context_rot | sliding_context:window_128k+scallop_injection | 1.0000 | 1.0000 |
| deep_context_rot | sliding_context:window_128k+reverse_ranked_injection | 0.0000 | 0.0000 |
| deep_context_rot | sliding_context:window_128k+scallop_change_only | 0.5000 | 0.5000 |
| deep_context_rot | sliding_context:window_128k+scallop_incongruity_only | 0.5000 | 0.5000 |

## Scallop Recursive-Reasoning Ablation

| Method | Overall accuracy | Direct retraction | Unretracted lineage | Two-hop lineage retraction |
|---|---:|---:|---:|---:|
| python_gold_resolver_replay | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| scallop_recursive | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| scallop_one_hop | 0.6667 | 1.0000 | 1.0000 | 0.0000 |

## Scallop Stream-Injection Ablation

Scallop receives only the causal structured event prefix and returns source event IDs. The injected context copies those original statements under the same token budget; it receives neither the query nor gold answer.

| Window | Interference tier | Checkpoint | Raw grounded | Scallop grounded | Reverse-ranked grounded | Change rule only | Incongruity rule only | Paired delta |
|---|---|---|---:|---:|---:|---:|---:|---:|
| window_4k | deep_context_rot | preference-change-delayed | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 |
| window_4k | deep_context_rot | preference-incongruity-delayed | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| window_4k | extreme_interference | preference-change-delayed | 0.0000 | 1.0000 | 0.4667 | 1.0000 | 0.0000 | 1.0000 |
| window_4k | extreme_interference | preference-incongruity-delayed | 0.0000 | 0.9667 | 0.6000 | 0.0000 | 0.9667 | 0.9667 |
| window_4k | heavy_interference | preference-change-delayed | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 |
| window_4k | heavy_interference | preference-incongruity-delayed | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| window_4k | light_interference | preference-change-delayed | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 |
| window_4k | light_interference | preference-incongruity-delayed | 1.0000 | 1.0000 | 1.0000 | 0.9667 | 1.0000 | 0.0000 |
| window_4k | no_interference | preference-change-delayed | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| window_4k | no_interference | preference-incongruity-delayed | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| window_16k | deep_context_rot | preference-change-delayed | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 |
| window_16k | deep_context_rot | preference-incongruity-delayed | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| window_16k | extreme_interference | preference-change-delayed | 0.0000 | 1.0000 | 0.4667 | 1.0000 | 0.0000 | 1.0000 |
| window_16k | extreme_interference | preference-incongruity-delayed | 0.0000 | 0.9667 | 0.6000 | 0.0000 | 0.9667 | 0.9667 |
| window_16k | heavy_interference | preference-change-delayed | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 |
| window_16k | heavy_interference | preference-incongruity-delayed | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| window_16k | light_interference | preference-change-delayed | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| window_16k | light_interference | preference-incongruity-delayed | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| window_16k | no_interference | preference-change-delayed | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| window_16k | no_interference | preference-incongruity-delayed | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| window_64k | deep_context_rot | preference-change-delayed | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 |
| window_64k | deep_context_rot | preference-incongruity-delayed | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| window_64k | extreme_interference | preference-change-delayed | 0.0000 | 1.0000 | 0.4667 | 1.0000 | 0.0000 | 1.0000 |
| window_64k | extreme_interference | preference-incongruity-delayed | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| window_64k | heavy_interference | preference-change-delayed | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| window_64k | heavy_interference | preference-incongruity-delayed | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| window_64k | light_interference | preference-change-delayed | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| window_64k | light_interference | preference-incongruity-delayed | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| window_64k | no_interference | preference-change-delayed | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| window_64k | no_interference | preference-incongruity-delayed | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| window_128k | deep_context_rot | preference-change-delayed | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 |
| window_128k | deep_context_rot | preference-incongruity-delayed | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| window_128k | extreme_interference | preference-change-delayed | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| window_128k | extreme_interference | preference-incongruity-delayed | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| window_128k | heavy_interference | preference-change-delayed | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| window_128k | heavy_interference | preference-incongruity-delayed | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| window_128k | light_interference | preference-change-delayed | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| window_128k | light_interference | preference-incongruity-delayed | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| window_128k | no_interference | preference-change-delayed | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| window_128k | no_interference | preference-incongruity-delayed | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |

## Memory Growth

- Checkpoints: 1950
- Mean stored turns: 401.15
- Maximum stored turns: 2967
- Mean stored tokens: 54211.47
- Maximum stored tokens: 408761
