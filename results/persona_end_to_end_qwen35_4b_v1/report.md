# Persona Conflict End-to-End Benchmark

## Scope

This benchmark runs Qwen3.5-4B on all 120 scheduled conditions from the 12 test histories in `persona_conflict_conversations_v1`.
Each condition is evaluated with five arms: 4K and 16K sliding context, matched 4K and 16K Scallop-derived structured memory, and the complete causal Qwen context.

The structured arm receives only natural source turns cited by query-blind Scallop rules from the causal same-history prefix, plus a maximal recent suffix under the same prompt-token cap.
When Scallop has no causal source, the structured and sliding prompts are byte-identical.

## Headline Results

| Arm | Exact match | Token F1 |
|---|---:|---:|
| Structured memory, 4K | 65.00% | 67.08% |
| Structured memory, 16K | 65.83% | 69.17% |
| Sliding context, 4K | 32.50% | 34.17% |
| Sliding context, 16K | 56.67% | 60.42% |
| Full Qwen context | 50.00% | 50.83% |

History-clustered paired exact-match differences over 12 histories:

| Comparison | Delta | 95% bootstrap interval |
|---|---:|---:|
| Structured 4K minus sliding 4K | +32.50 points | [+29.17, +35.83] |
| Structured 16K minus sliding 16K | +9.17 points | [+3.33, +15.83] |
| Full context minus sliding 16K | -6.67 points | [-19.17, +5.83] |
| Full context minus structured 16K | -15.83 points | [-24.17, -7.50] |

The complete context underperforms structured 16K despite fitting within the model's 262K context capacity.
This is evidence of mixed-account context interference rather than context truncation.

## Causal Conditions

At delayed probes, where every query has the expected causal Scallop relation:

| Comparison | Exact-match delta | 95% bootstrap interval |
|---|---:|---:|
| Structured 4K minus sliding 4K | +83.33 points | [+70.83, +95.83] |
| Structured 16K minus sliding 16K | +33.33 points | [+20.83, +45.83] |

At the immediate pre-update and post-update controls, Scallop has no causal relation source and same-cap structured/sliding prompts are identical.
Their paired differences are exactly zero, as required.

Across all 44 source-present conditions, structured-minus-sliding exact-match differences are +88.64 points at 4K and +25.00 points at 16K.
Across the 76 source-absent conditions, both differences are exactly zero.

## Integrity

- 600 unique generations: 120 conditions times five arms.
- Zero generation truncations at the 32-token answer cap.
- No prompt exceeded its declared token budget.
- Full context retained every causal-prefix turn; the largest request needed 52,061 positions.
- Every Scallop source was same-history and causally prior.
- No latent event, fact, history, subject, query, or synthetic value identifier entered a prompt.
- Qwen revision, model files, tokenizer files, evaluator source, configuration, source corpus, prompt specifications, and Scallop injection map are cryptographically bound in the manifests.

## Limitations

All 12 test histories reuse the same two final surface answers, so unrelated-account conversations frequently repeat target answers.
This is deliberate cross-account interference but also permits lexical-frequency shortcuts.
The intervals therefore describe this fixed repeated-surface synthetic corpus and do not establish generalization to unseen answer vocabularies or natural human conversations.

Only 44 of 120 conditions have a causal Scallop source.
The supported claim is that causal source-turn injection improves Qwen accuracy when the relation is available, not that structured memory changes every phase.

Exact match is the primary metric.
Token F1 partially rewards semantically wrong pairs such as `evening delivery` versus `weekend delivery` and should not be used alone.

These results establish end-to-end relevance for NeuroSym's causal structured-memory mechanism on this benchmark.
They do not establish universal model, dataset, or deployment superiority.
