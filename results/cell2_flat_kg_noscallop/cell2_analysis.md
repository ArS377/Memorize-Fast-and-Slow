# Cell 2 Flat KG No-Scallop Results

Source run: `results/cell2_ynez_resume_20260704` on `serrano`.

## Summary

- Cell: `2`, `flat_kg_noscallop`.
- Examples answered: `50`.
- Correct: `26/50` (`52.0%`).
- Errors: `0`.
- Mean latency: `23.50s` (min `4.93s`, max `40.68s`).
- KG triples retrieved: all `50/50` examples had nonzero retrieved triples; mean `9.56`, range `2-21`.
- Context size: mean `3794` chars, range `3026-4016` chars.
- KG facts file: `results/kg_builds/pilot_noscallop_facts.jsonl` (`13705` JSONL facts loaded by the run).

## KG Usage Check

The run invoked Cell 2 with `--facts-file results/cell2_ynez_resume_20260704/kg_builds/pilot_noscallop_facts.jsonl` and the log reports `Loaded 13705 facts`. This is the extracted KG facts JSONL, not the original LongBench `data.jsonl`. The result rows also show nonzero `n_triples` for every example, with `n_context_chars` around 3-4k, so the answering pass did use KG retrieval context.

## Breakdown

### By Difficulty

- `easy`: `11/18` (`61.1%`)
- `hard`: `15/32` (`46.9%`)

### By Domain

- `Code Repository Understanding`: `2/3` (`66.7%`)
- `Long In-context Learning`: `3/8` (`37.5%`)
- `Long Structured Data Understanding`: `4/4` (`100.0%`)
- `Long-dialogue History Understanding`: `1/2` (`50.0%`)
- `Multi-Document QA`: `6/14` (`42.9%`)
- `Single-Document QA`: `10/19` (`52.6%`)

### Answer Distribution

- `A`: predicted `14`, gold `10`
- `B`: predicted `8`, gold `15`
- `C`: predicted `12`, gold `9`
- `D`: predicted `16`, gold `16`

Most common wrong gold/predicted pairs:
- gold `D` -> predicted `C`: `6`
- gold `B` -> predicted `D`: `5`
- gold `B` -> predicted `C`: `3`
- gold `D` -> predicted `A`: `2`
- gold `C` -> predicted `D`: `2`
- gold `C` -> predicted `B`: `2`
- gold `C` -> predicted `A`: `2`
- gold `B` -> predicted `A`: `1`

## Key Signs

- The KG retrieval path is active and materially populated: every row has retrieved triples, unlike the Cell 6 result pattern where many examples had effectively empty context.
- Accuracy is higher than prior raw/Cell 6 snapshots, but still near chance-plus on a four-choice task; the model is not reliably grounded just because KG context is present.
- Wrong answers often occur despite 8-16 retrieved triples and near-max context length, which points to retrieval precision and answer-selection issues, not just missing context.
- Latency varies substantially, suggesting some examples cause long reasoning/generation even after retrieval is bounded.

## Failure Examples

### `671b3cabbb02136c067d5252`

- Predicted `C`, gold `B`, triples `21`, context chars `3896`, latency `15.42s`.
- Domain: `Long-dialogue History Understanding` / `Agent history QA`, difficulty `hard`.
- Question: Which player got the least utility in the game?
  - A: player_1
  - B: player_3
  - C: player_5
  - D: player_7

### `670aac92bb02136c067d218a`

- Predicted `C`, gold `D`, triples `15`, context chars `3951`, latency `15.86s`.
- Domain: `Single-Document QA` / `Detective`, difficulty `hard`.
- Question: Which floor do Becca DiNuzio and Greg Barney live on?
  - A: The first floor
  - B: The second floor
  - C: The third floor
  - D: There are contradictory descriptions

### `66f920d8bb02136c067c4b81`

- Predicted `B`, gold `C`, triples `12`, context chars `3878`, latency `14.50s`.
- Domain: `Single-Document QA` / `Literary`, difficulty `hard`.
- Question: What is mainly symbolized by the frequent cholera outbreaks in the novel?
  - A: Confusion of The Times
  - B: The impermanence of the character's fate
  - C: Love is dangerous and uncontrollable
  - D: Social indifference

### `66ecfe1e821e116aacb1e41c`

- Predicted `D`, gold `C`, triples `11`, context chars `4016`, latency `26.89s`.
- Domain: `Single-Document QA` / `Financial`, difficulty `hard`.
- Question: Based on the challenges and nuances discussed in "Understanding China’s Economic Statistics – Third Edition," which of the following best explains the limitations in accurately assessing China's economic momentum through official data, particularly when comparing industrial production (IP) and GDP growth?
  - A: Although industrial production (IP) data is released more frequently and is considered reliable, it lacks comprehensive coverage of small enterprises, resulting in a limited view of the overall economy’s growth momentum.
  - B: While China’s GDP data is compiled according to international standards, its smoothness in reported growth rates during downturns has raised skepticism, and therefore IP data must always be weighted more heavily in economic assessments.
  - C: China's reliance on production-side data rather than expenditure-side data leads to an overestimation of economic momentum, as production data reflects industrial growth but often overlooks key sectors such as services and consumption.
  - D: GDP revisions, particularly for tertiary industries like services, create significant discrepancies between reported growth rates, making it difficult to cross-check data with high-frequency indicators such as electricity consumption and freight traffic.

### `66ebed525a08c7b9b35e1cb4`

- Predicted `D`, gold `B`, triples `11`, context chars `3954`, latency `9.44s`.
- Domain: `Single-Document QA` / `Academic`, difficulty `hard`.
- Question: When Miller tried to answer the question "should we read Heart of Darkness?", he put forward a new concept for read "but perform a reading in the strong sense, an active responsible response that renders justice to a book by generating more language in its turn". However, he actually laid an implied premise for his argument, which one of the followings is true?
  - A: Each must read for himself or herself and testify anew.
  - B: Readers must reach a high standrad to some degree.
  - C: It is the readers' obligation to get the "truth" from the primary narrator.
  - D: The performative interpretation of language transforms what it interprets.

## Ways To Improve

- Add retrieval diagnostics to each result row: seed entities, selected fact IDs, predicates, and a short excerpt of the formatted KG context. The current rows only expose counts, making error attribution harder.
- Score or rerank retrieved facts by question-option overlap before truncation; many failures have plenty of triples, so precision is likely more important than simply increasing `limit_triples`.
- Add an answer-verification pass that checks the chosen option against retrieved facts and explicitly compares all four choices.
- Keep per-example KG fact files for resumability. The per-example strategy used for examples 39-50 avoided losing all progress when servers rebooted.
- Compare Cell 2 against raw Cell 1 on the same exact ordered 50-example slice; current historical summaries were produced under different result states, so comparisons should be treated as directional.

## Artifacts

- `results/cell2_flat_kg_noscallop/results.jsonl`
- `results/summary.csv`
- `results/figures/accuracy_grid.png`
- `results/figures/accuracy_grid_cell2.png`
- `results/kg_builds/pilot_noscallop_facts.jsonl`
- `results/run_metadata/cell2_ynez_resume_20260704.json`
- `results/cell2_flat_kg_noscallop/cell2_ynez_full50_run.log`
