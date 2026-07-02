# RLM Improvement Analysis — cell 4 (`rlm_raw`)

Derived from `results/rlm_reasoning.md` (50 examples), the harness in
`rlm_baseline.py` / `experiments/rlm_answerer.py` / `experiments/common.py`, and
the aggregate in `results/summary.csv`.

## Headline result

The RLM is **strictly worse than the flat baseline** as currently configured:

| Cell | Accuracy | Mean latency | Errors | Answered |
| --- | --- | --- | --- | --- |
| `flat_raw` (cell 1) | **0.34** | **16s** | **0** | 50/50 |
| `rlm_raw` (cell 4) | **0.32** (~10/31) | **153s** (~10x) | **19/50 (38%)** | 31/50 |

Recursion adds ~10x cost and a 38% crash rate for **no accuracy gain**. Every
improvement below targets that gap, ordered by expected impact.

---

## 1. The final-answer submission API is broken (highest impact, easy fix)

The model doesn't know how to submit and guesses wrong function names:

- `FINAL(D)` → `NameError: name 'FINAL' is not defined`
  (`results/rlm_reasoning.md:46-56`). Example 1 *reasoned correctly* to D and
  only scored because `extract_letter` salvaged the letter from prose.
- `FINAL_VAR(answer)` while `answer` was never assigned →
  `NameError: name 'answer' is not defined`, repeated every iteration
  (`results/rlm_reasoning.md:376-398`).

**Fix:** State the exact finalize contract in the `root_prompt`.
`format_question` (`rlm_baseline.py:48-57`, duplicated in
`experiments/common.py:72-81`) says only *"Answer with only the letter"* — it
never tells the model which REPL call ends the episode or that a variable must
be populated first. Add explicit instructions (correct sentinel,
assign-then-finalize pattern).

## 2. The model never grounds in the actual context

Across almost all examples the model says *"since I can't see the actual
content… I'll rely on the previous answer"* and answers from parametric memory
(`results/rlm_reasoning.md:198-213`, `results/rlm_reasoning.md:4282-4292`). It
**never slices, searches, or prints the raw `context`** — no `print(context[...])`,
keyword search, regex, or chunking; it only fires blind `llm_query()` calls.

**Fix:** Scaffold a retrieval-first loop in the prompt: chunk/keyword-search
`context`, `print` the matching snippets, then answer *with a quoted evidence
span*. This is the whole point of the REPL and it is currently unused.

## 3. `llm_query` sub-calls don't receive the context

When it does delegate, the sub-LM also has no context and hallucinates: in
Example 2 the sub-LM reasons *"the context isn't provided here"* and guesses D
(`results/rlm_reasoning.md:99-161`). Only once did the model manually
concatenate context into the query (`results/rlm_reasoning.md:14641`).

**Fix:** Ensure `llm_query` is passed the relevant context chunk
(retrieval-augment the sub-call), and instruct the model that `llm_query` does
not see `context` unless it is included.

## 4. Sub-LM output is unusable (raw `<think>` leaks into variables)

`llm_query` returns full Qwen3 `<think>…</think>` blocks. The model then does
`answer = analysis.split()[-1].upper()` (`results/rlm_reasoning.md:875`), which
grabs the last token of a reasoning dump rather than the letter.

**Fix:** Strip `<think>` and return a short normalized answer from `llm_query`
(mirror the logic already in `extract_letter`, `experiments/common.py:105-115`)
so downstream parsing is deterministic.

## 5. Runaway duplicate iterations cause the 38% token-limit crashes

19/50 runs died with **"Token limit exceeded"** (e.g.
`results/rlm_reasoning.md:340`, `results/rlm_reasoning.md:4785` at
87,854/64,000). Root cause: the model emits the *same* answer 8–11 times with no
new information (Example 14 repeats "B" for iterations 2–11,
`results/rlm_reasoning.md:4379-4471`), while each turn re-injects the 32k context
+ full `<think>` history until it overflows `max_tokens=64000`
(`rlm_baseline.py:76`).

**Fixes (any one helps; combined they largely eliminate the errors):**

- **Early-stop:** terminate as soon as a valid A/B/C/D final answer is produced
  (Example 1 needed 1 iteration; most need only 1–2).
- **Loop-detection:** break on N identical consecutive responses.
- **Do not resend the full 32k context every iteration**; keep it addressable
  via the REPL variable only.
- **Exclude `<think>` from the running transcript** so history doesn't compound.

## 6. Config / cost tuning

- `max_depth=2` (`rlm_baseline.py:73-74`) is wasted because context never
  reaches depth-2 sub-calls (see #3).
- Consider disabling Qwen3 thinking (or `enable_thinking=False`) for the root
  loop to cut the token bloat that drives both latency and the overflow in #5.

---

## Token-limit failures (19/50)

Examples that hard-failed with "Token limit exceeded": 3, 4, 8, 12, 17, 19, 20,
22, 23, 25, 27, 29 (first 30), plus 7 more in examples 31–50.

## Bottom line

The RLM's *reasoning* is often fine (Example 1 derived the right answer); the
losses come almost entirely from **plumbing**:

- No finalize contract (#1)
- No context grounding (#2 / #3)
- Dirty sub-LM outputs (#4)
- Unbounded loops (#5)

Fixing **#1 + #5** alone should recover most of the 19 crashed examples and cut
latency dramatically. **#2 + #3** are what would let recursion actually *beat*
the flat baseline.
