# Synthetic Temporal Benchmark

Status: completed
Rule version: synthetic_temporal_preferences.v1

| Condition | Policy | Decision accuracy | no_history | recency | full_history | bm25 | dense |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| lexical | accept_all | 0.500 | 0.000 | 0.000 | 0.800 | 1.000 | 0.800 |
| lexical | scallop_fail_closed | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 |
| paraphrase | accept_all | 0.500 | 0.000 | 0.000 | 0.800 | 0.800 | 0.836 |
| paraphrase | scallop_fail_closed | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.838 |
