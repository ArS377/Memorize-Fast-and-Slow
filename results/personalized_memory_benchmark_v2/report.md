# Synthetic Temporal Benchmark

Status: completed
Rule version: synthetic_temporal_preferences.v1

| Condition | Policy | Decision accuracy | no_history | recency | full_history | bm25 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| lexical | accept_all | 0.333 | 0.000 | 0.000 | 0.667 | 0.778 |
| lexical | scallop_fail_closed | 1.000 | 0.000 | 0.000 | 1.000 | 0.444 |
| paraphrase | accept_all | 0.333 | 0.000 | 0.000 | 0.667 | 0.556 |
| paraphrase | scallop_fail_closed | 1.000 | 0.000 | 0.000 | 1.000 | 0.556 |

## Query Accuracy by Kind

| Condition | Policy | Baseline | Aggregate | preference | recommendation | private_recall | ambiguity |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| lexical | accept_all | no_history | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| lexical | accept_all | recency | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| lexical | accept_all | full_history | 0.667 | 0.500 | 1.000 | 1.000 | 1.000 |
| lexical | accept_all | bm25 | 0.778 | 0.667 | 1.000 | 1.000 | 1.000 |
| lexical | scallop_fail_closed | no_history | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| lexical | scallop_fail_closed | recency | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| lexical | scallop_fail_closed | full_history | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| lexical | scallop_fail_closed | bm25 | 0.444 | 0.167 | 1.000 | 1.000 | 1.000 |
| paraphrase | accept_all | no_history | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| paraphrase | accept_all | recency | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| paraphrase | accept_all | full_history | 0.667 | 0.500 | 1.000 | 1.000 | 1.000 |
| paraphrase | accept_all | bm25 | 0.556 | 0.333 | 1.000 | 1.000 | 1.000 |
| paraphrase | scallop_fail_closed | no_history | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| paraphrase | scallop_fail_closed | recency | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| paraphrase | scallop_fail_closed | full_history | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| paraphrase | scallop_fail_closed | bm25 | 0.556 | 0.333 | 1.000 | 1.000 | 1.000 |
