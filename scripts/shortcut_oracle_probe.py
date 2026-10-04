"""Score oracle shortcut rules that need no memory reasoning.

If a rule this cheap matches the memory arms, the benchmark is not measuring
revision. Three rules, all given perfect access to the account's own events:

  last_value   - the most recently asserted surface value before the checkpoint
  last_pref    - as above but restricted to PREFERS predicates
  first_value  - the earliest asserted value, as a control
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
from transformers import AutoTokenizer

from experiments.persona_end_to_end_benchmark import (
    _load_source_and_rebuild,
    load_benchmark_config,
)

config = load_benchmark_config(Path("configs/persona_end_to_end_joint_surface_a.json"))
tokenizer = AutoTokenizer.from_pretrained(config.model_path, local_files_only=True)
scheduled, turns, _ = _load_source_and_rebuild(config, tokenizer)

scores = {"last_value": 0, "last_pref": 0, "first_value": 0}
total = 0
for condition in scheduled["inputs"]:
    history = str(condition["history_id"])
    checkpoint = int(condition["checkpoint_turn_index"])
    gold = str(condition["gold"]).strip().casefold()
    values, prefs = [], []
    for turn in turns[:checkpoint]:
        if str(turn.get("history_id")) != history:
            continue
        event = turn.get("stream_event") or {}
        surface = str(event.get("surface_object") or "").strip()
        if not surface:
            continue
        values.append(surface)
        if str((event.get("fact") or {}).get("predicate", "")) == "PREFERS":
            prefs.append(surface)
    total += 1
    if values and values[-1].casefold() == gold:
        scores["last_value"] += 1
    if prefs and prefs[-1].casefold() == gold:
        scores["last_pref"] += 1
    if values and values[0].casefold() == gold:
        scores["first_value"] += 1

print(f"conditions: {total}")
for name, hits in scores.items():
    print(f"  {name:<12} {hits:>3}/{total} = {100 * hits / max(total, 1):5.2f}% EM")
