# Interleaved Continual Memory Benchmark

- Benchmark version: `interleaved_continual_memory.v2`
- Evolving peer tasks: 1,024
- Stream tokens: 2,738,857
- Declared context coverage: 2.612x
- Canonical contradiction pairs: 1,024 (1,024 resolved)

## Final Transcript-Suffix Retention

These loss rates reserve the declared budget for transcript turns only; they do not include a query prompt.

- 4,096 tokens: 100.00% complete-provenance loss
- 16,384 tokens: 99.63% complete-provenance loss
- 65,536 tokens: 97.90% complete-provenance loss
- 131,072 tokens: 95.43% complete-provenance loss
- 262,144 tokens: 90.70% complete-provenance loss
- 1,048,576 tokens: 61.89% complete-provenance loss

## Online Causal Checkpoints

Online windows include the exact query prompt in their token budget.
Answer sufficiency uses the disclosed deterministic oracle resolver and is not LLM generation accuracy.

- full_structured_memory: oracle answer 100.00%; complete provenance 100.00%; grounded answer 100.00%
- sliding_context:1048576: oracle answer 100.00%; complete provenance 100.00%; grounded answer 100.00%
- sliding_context:131072: oracle answer 100.00%; complete provenance 100.00%; grounded answer 100.00%
- sliding_context:16384: oracle answer 92.63%; complete provenance 81.40%; grounded answer 81.40%
- sliding_context:262144: oracle answer 100.00%; complete provenance 100.00%; grounded answer 100.00%
- sliding_context:4096: oracle answer 41.43%; complete provenance 13.11%; grounded answer 13.11%
- sliding_context:65536: oracle answer 100.00%; complete provenance 100.00%; grounded answer 100.00%
