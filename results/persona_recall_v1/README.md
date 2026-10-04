# Persona Recall Smoke Evaluation

Status: exploratory smoke only

The run contains 40 Qwen evaluation rows and reports 25% overall accuracy.
It has only two independent history-query units per persona pair and evaluates only ethnicity and education, so it is underpowered for persona-effect conclusions.
The structured-memory control is an oracle rather than a Qwen generation arm.

See [`persona_recall_report.json`](persona_recall_report.json), [`evaluation_manifest.json`](evaluation_manifest.json), and [`qwen_predictions.jsonl`](qwen_predictions.jsonl).
