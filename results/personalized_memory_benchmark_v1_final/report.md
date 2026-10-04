# Synthetic Temporal Benchmark

Status: completed
Rule version: synthetic_temporal_preferences.v1

| Condition | Policy | Decision accuracy | no_history | recency | full_history | bm25 | dense |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| lexical | accept_all | 0.500 | 0.000 | 0.000 | 0.800 | 1.000 | 0.800 |
| lexical | scallop_fail_closed | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 |
| paraphrase | accept_all | 0.500 | 0.000 | 0.000 | 0.800 | 0.800 | 0.836 |
| paraphrase | scallop_fail_closed | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.838 |

## Query Accuracy by Kind

| Condition | Policy | Baseline | Aggregate | preference | recommendation | private_recall | ambiguity |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| lexical | accept_all | no_history | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| lexical | accept_all | recency | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| lexical | accept_all | full_history | 0.800 | 0.500 | 1.000 | 1.000 | 1.000 |
| lexical | accept_all | bm25 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| lexical | accept_all | dense | 0.800 | 0.500 | 1.000 | 1.000 | 1.000 |
| lexical | scallop_fail_closed | no_history | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| lexical | scallop_fail_closed | recency | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| lexical | scallop_fail_closed | full_history | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| lexical | scallop_fail_closed | bm25 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| lexical | scallop_fail_closed | dense | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| paraphrase | accept_all | no_history | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| paraphrase | accept_all | recency | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| paraphrase | accept_all | full_history | 0.800 | 0.500 | 1.000 | 1.000 | 1.000 |
| paraphrase | accept_all | bm25 | 0.800 | 0.500 | 1.000 | 1.000 | 1.000 |
| paraphrase | accept_all | dense | 0.836 | 0.605 | 0.980 | 0.990 | 1.000 |
| paraphrase | scallop_fail_closed | no_history | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| paraphrase | scallop_fail_closed | recency | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| paraphrase | scallop_fail_closed | full_history | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| paraphrase | scallop_fail_closed | bm25 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| paraphrase | scallop_fail_closed | dense | 0.838 | 0.605 | 0.980 | 1.000 | 1.000 |
