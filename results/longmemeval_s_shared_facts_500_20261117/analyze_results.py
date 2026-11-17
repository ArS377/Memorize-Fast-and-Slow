#!/usr/bin/env python3
"""Summarize the LongMemEval-S shared-facts NeuroSym versus Graphiti comparison.

The script uses only the Python standard library. It reads the committed
`per_history.csv` (500 histories, one binary judge label per arm) and, for the
sensitivity analyses, the companion `per_history_groups.csv`, and regenerates
`summary.csv`, `paired_comparison.json`, and `comparisons.csv`. Pass
`--markdown` to also print the README tables.

The seeded paired bootstrap resamples rows in file order, so reordering the
file changes bootstrap interval endpoints in the last digit; counts, Wilson
intervals, and McNemar p-values do not depend on order. The seed and the
number of resamples are the same as for the 250-history result, so the
`published 250` cohort row reproduces that result exactly.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
BOOTSTRAP_SEED = 20261114
BOOTSTRAP_REPS = 10000
EXPECTED_ROWS = 500
STRATA = ("single-session-user", "single-session-assistant", "single-session-preference",
          "multi-session", "knowledge-update", "temporal-reasoning", "abstention")
# LongMemEval-S cleaned stratum counts over all 500 questions (scorer `stratum_population`).
POPULATION_STRATA = {"abstention": 30, "knowledge-update": 72, "multi-session": 121,
                     "single-session-assistant": 56, "single-session-preference": 30,
                     "single-session-user": 64, "temporal-reasoning": 127}
BATCHES = ("published_250", "A1", "A2", "B1", "L1a", "L1b", "L2", "V1", "O11", "O12")
# Disclosed groups in per_history_groups.csv (1 = member).
GROUPS = (
    ("label_dispute_kept", "extractor/verifier label dispute recorded, fact kept"),
    ("smaller_pieces", "recovered by re-extraction in smaller pieces"),
    ("full_context_corrected", "facts excluded by the full-context verdict rule"),
    ("repaired_extraction", "earlier repaired extraction"),
)


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value: doubled binomial(b+c, 0.5) tail, capped at 1."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def paired_bootstrap(diffs: list[int]) -> tuple[float, float]:
    """Percentile 95% interval for the mean paired difference, resampling histories."""
    rng = random.Random(BOOTSTRAP_SEED)
    n = len(diffs)
    stats = []
    for _ in range(BOOTSTRAP_REPS):
        s = 0
        for _ in range(n):
            s += diffs[rng.randrange(n)]
        stats.append(s / n)
    stats.sort()
    return stats[int(math.floor(0.025 * BOOTSTRAP_REPS))], stats[int(math.ceil(0.975 * BOOTSTRAP_REPS)) - 1]


def compare(rows: list[dict[str, str]], family: str, label: str) -> dict[str, Any]:
    ns = [int(r["neurosym_correct"]) for r in rows]
    gr = [int(r["graphiti_correct"]) for r in rows]
    n = len(rows)
    b = sum(1 for x, y in zip(ns, gr) if x and not y)
    c = sum(1 for x, y in zip(ns, gr) if y and not x)
    both = sum(1 for x, y in zip(ns, gr) if x and y)
    lo, hi = paired_bootstrap([x - y for x, y in zip(ns, gr)])
    return {
        "family": family, "group": label, "n": n,
        "neurosym_correct": sum(ns), "neurosym_accuracy": sum(ns) / n,
        "neurosym_wilson95": wilson(sum(ns), n),
        "graphiti_correct": sum(gr), "graphiti_accuracy": sum(gr) / n,
        "graphiti_wilson95": wilson(sum(gr), n),
        "difference": (sum(ns) - sum(gr)) / n, "bootstrap95": (lo, hi),
        "neurosym_only": b, "graphiti_only": c, "both_correct": both, "both_wrong": n - b - c - both,
        "mcnemar_exact_p": mcnemar_exact(b, c),
    }


def grouped(rows, family, key, order):
    buckets = defaultdict(list)
    for r in rows:
        buckets[key(r)].append(r)
    return [compare(buckets[k], family, k) for k in order if k in buckets]


def metric_variants(rows, col):
    labels = [int(r[col]) for r in rows]
    by_type, by_stratum = defaultdict(list), defaultdict(list)
    for r in rows:
        by_type[r["question_type"]].append(int(r[col]))
        by_stratum[r["stratum"]].append(int(r[col]))
    total = sum(POPULATION_STRATA.values())
    return {
        "overall_accuracy": sum(labels) / len(labels),
        "task_averaged_accuracy": statistics.mean(sum(v) / len(v) for v in by_type.values()),
        "population_weighted_accuracy": sum(sum(by_stratum[s]) / len(by_stratum[s]) * w / total
                                            for s, w in POPULATION_STRATA.items()),
    }


def standardized(rows, col):
    """Accuracy re-weighted to the full-500 stratum mix (the scorer's population-weighted accuracy)."""
    return metric_variants(rows, col)["population_weighted_accuracy"]


def cohort_mix(first: list[dict[str, str]], second: list[dict[str, str]]) -> dict[str, Any]:
    """Accuracy of the later 250 minus the published 250, raw and after standardizing both cohorts to the
    full-500 stratum mix; 95% percentile bootstrap resampling histories within each cohort and stratum."""
    rng = random.Random(BOOTSTRAP_SEED)
    out: dict[str, Any] = {"published_n": len(first), "later_n": len(second),
                           "published_stratum_counts": {s: sum(r["stratum"] == s for r in first) for s in STRATA},
                           "later_stratum_counts": {s: sum(r["stratum"] == s for r in second) for s in STRATA}}
    strat_first = {s: [r for r in first if r["stratum"] == s] for s in STRATA}
    strat_second = {s: [r for r in second if r["stratum"] == s] for s in STRATA}
    for arm, col in (("neurosym", "neurosym_correct"), ("graphiti", "graphiti_correct")):
        raw = statistics.mean(int(r[col]) for r in second) - statistics.mean(int(r[col]) for r in first)
        std = standardized(second, col) - standardized(first, col)
        raw_bs, std_bs = [], []
        for _ in range(BOOTSTRAP_REPS):
            a = [rng.choice(v) for s, v in strat_first.items() for _ in v]
            b = [rng.choice(v) for s, v in strat_second.items() for _ in v]
            raw_bs.append(statistics.fmean(int(r[col]) for r in b) - statistics.fmean(int(r[col]) for r in a))
            std_bs.append(standardized(b, col) - standardized(a, col))
        raw_bs.sort(); std_bs.sort()
        lo, hi = int(math.floor(0.025 * BOOTSTRAP_REPS)), int(math.ceil(0.975 * BOOTSTRAP_REPS)) - 1
        out[arm] = {"published_accuracy": statistics.mean(int(r[col]) for r in first),
                    "later_accuracy": statistics.mean(int(r[col]) for r in second),
                    "published_standardized": standardized(first, col), "later_standardized": standardized(second, col),
                    "raw_difference": raw, "raw_bootstrap95": [raw_bs[lo], raw_bs[hi]],
                    "standardized_difference": std, "standardized_bootstrap95": [std_bs[lo], std_bs[hi]]}
    out["method"] = ("later minus published; standardized = population-weighted accuracy (full-500 stratum mix); "
                     f"bootstrap resamples histories within cohort and stratum, {BOOTSTRAP_REPS} resamples, seed {BOOTSTRAP_SEED}")
    return out


def analyze(path: Path, groups_path: Path | None, expected_rows: int) -> dict[str, Any]:
    rows = list(csv.DictReader(open(path, newline="")))
    assert len(rows) == expected_rows, len(rows)
    assert len({r["question_id"] for r in rows}) == len(rows), "duplicate question_id"
    for r in rows:
        assert r["neurosym_correct"] in ("0", "1") and r["graphiti_correct"] in ("0", "1"), r["question_id"]
        assert r["stratum"] in STRATA, r["question_id"]

    headline = compare(rows, "headline", f"{len(rows)} evaluated histories")
    comparisons = [headline]
    comparisons += grouped(rows, "stratum", lambda r: r["stratum"], STRATA)
    comparisons += grouped(rows, "question_type", lambda r: r["question_type"], STRATA[:-1])
    comparisons.append(compare([r for r in rows if r["abstention"] != "True"], "sensitivity", "non_abstention_only"))
    mix = None
    if groups_path is not None and groups_path.exists():
        groups = {g["question_id"]: g for g in csv.DictReader(open(groups_path, newline=""))}
        assert set(groups) == {r["question_id"] for r in rows}, "groups file does not match per_history.csv"
        for col, _ in GROUPS:
            members = [r for r in rows if groups[r["question_id"]][col] == "1"]
            if members:
                comparisons.append(compare([r for r in rows if groups[r["question_id"]][col] != "1"], "sensitivity",
                                           f"excluding_{col}"))
                comparisons.append(compare(members, "sensitivity", f"only_{col}"))
        second_db = [r for r in rows if groups[r["question_id"]]["neurosym_db_instance"] == "2"]
        if second_db:
            comparisons.append(compare([r for r in rows if groups[r["question_id"]]["neurosym_db_instance"] != "2"],
                                       "sensitivity", "excluding_neurosym_db_instance_2"))
            comparisons.append(compare(second_db, "sensitivity", "only_neurosym_db_instance_2"))
        flagged = lambda g: any(g[c] == "1" for c, _ in GROUPS) or g["neurosym_db_instance"] == "2"
        comparisons.append(compare([r for r in rows if not flagged(groups[r["question_id"]])], "sensitivity",
                                   "excluding_all_flagged"))
        published = [r for r in rows if groups[r["question_id"]]["evaluation_batch"] == "published_250"]
        later = [r for r in rows if groups[r["question_id"]]["evaluation_batch"] != "published_250"]
        comparisons.append(compare(published, "cohort", "published_250"))
        comparisons.append(compare(later, "cohort", f"later_{len(later)}"))
        comparisons += grouped(rows, "batch", lambda r: groups[r["question_id"]]["evaluation_batch"], BATCHES)
        if published and later:
            mix = cohort_mix(published, later)
    return {
        "headline": headline, "comparisons": comparisons, "cohort_type_mix": mix,
        "metric_variants": {"neurosym": metric_variants(rows, "neurosym_correct"),
                            "graphiti": metric_variants(rows, "graphiti_correct")},
    }


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def markdown_table(results, first_col):
    lines = [f"| {first_col} | n | NeuroSym | Graphiti | Difference, 95% paired bootstrap CI | NeuroSym-only / Graphiti-only | Exact McNemar p |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        lo, hi = r["bootstrap95"]
        lines.append(
            f"| {r['group']} | {r['n']} | {pct(r['neurosym_accuracy'])} | {pct(r['graphiti_accuracy'])} | "
            f"{100 * r['difference']:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}] | "
            f"{r['neurosym_only']} / {r['graphiti_only']} | {r['mcnemar_exact_p']:.3g} |")
    return "\n".join(lines)


def write_outputs(res: dict[str, Any], out_dir: Path) -> None:
    h = res["headline"]
    with open(out_dir / "summary.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["arm", "n", "correct", "accuracy", "ci95_low", "ci95_high", "task_averaged_accuracy",
                    "population_weighted_accuracy"])
        for arm in ("neurosym", "graphiti"):
            v = res["metric_variants"][arm]
            w.writerow([arm, h["n"], h[f"{arm}_correct"], f"{h[f'{arm}_accuracy']:.4f}", f"{h[f'{arm}_wilson95'][0]:.4f}",
                        f"{h[f'{arm}_wilson95'][1]:.4f}", f"{v['task_averaged_accuracy']:.4f}",
                        f"{v['population_weighted_accuracy']:.4f}"])
    paired = {
        "n": h["n"], "both_correct": h["both_correct"], "neurosym_only_correct": h["neurosym_only"],
        "graphiti_only_correct": h["graphiti_only"], "both_wrong": h["both_wrong"],
        "difference_neurosym_minus_graphiti": h["difference"],
        "bootstrap95": list(h["bootstrap95"]), "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_reps": BOOTSTRAP_REPS,
        "bootstrap_method": "percentile; resamples histories with replacement",
        "mcnemar_exact_two_sided_p": h["mcnemar_exact_p"],
    }
    if res["cohort_type_mix"] is not None:
        paired["cohort_type_mix"] = res["cohort_type_mix"]
    (out_dir / "paired_comparison.json").write_text(json.dumps(paired, indent=2) + "\n")
    with open(out_dir / "comparisons.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["family", "group", "n", "neurosym_correct", "neurosym_accuracy", "neurosym_ci95_low",
                    "neurosym_ci95_high", "graphiti_correct", "graphiti_accuracy", "graphiti_ci95_low",
                    "graphiti_ci95_high", "difference_pp", "bootstrap95_low_pp", "bootstrap95_high_pp",
                    "neurosym_only", "graphiti_only", "both_correct", "both_wrong", "mcnemar_exact_p"])
        for r in res["comparisons"]:
            w.writerow([r["family"], r["group"], r["n"], r["neurosym_correct"], f"{r['neurosym_accuracy']:.4f}",
                        f"{r['neurosym_wilson95'][0]:.4f}", f"{r['neurosym_wilson95'][1]:.4f}", r["graphiti_correct"],
                        f"{r['graphiti_accuracy']:.4f}", f"{r['graphiti_wilson95'][0]:.4f}",
                        f"{r['graphiti_wilson95'][1]:.4f}", f"{100 * r['difference']:.2f}",
                        f"{100 * r['bootstrap95'][0]:.2f}", f"{100 * r['bootstrap95'][1]:.2f}", r["neurosym_only"],
                        r["graphiti_only"], r["both_correct"], r["both_wrong"], f"{r['mcnemar_exact_p']:.4f}"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, default=HERE / "per_history.csv")
    parser.add_argument("--groups", type=Path, default=HERE / "per_history_groups.csv")
    parser.add_argument("--output-dir", type=Path, default=HERE)
    parser.add_argument("--expected-rows", type=int, default=EXPECTED_ROWS)
    parser.add_argument("--markdown", action="store_true", help="print the README tables")
    args = parser.parse_args()
    res = analyze(args.input, args.groups, args.expected_rows)
    write_outputs(res, args.output_dir)
    h = res["headline"]
    print(f"NeuroSym {h['neurosym_correct']}/{h['n']}, Graphiti {h['graphiti_correct']}/{h['n']}, "
          f"difference {100 * h['difference']:+.1f} pp, bootstrap95 [{100 * h['bootstrap95'][0]:+.1f}, "
          f"{100 * h['bootstrap95'][1]:+.1f}], McNemar p={h['mcnemar_exact_p']:.3g}")
    if args.markdown:
        fams = defaultdict(list)
        for r in res["comparisons"]:
            fams[r["family"]].append(r)
        for fam, rs in fams.items():
            print(f"\n### {fam}\n\n{markdown_table(rs, fam)}")
        mix = res["cohort_type_mix"]
        if mix:
            print("\n### cohort type mix (later minus published 250)\n")
            print("| Arm | Published 250 | Later | Raw difference [95% CI] | Standardized to 500 mix: published / later | Standardized difference [95% CI] |")
            print("|---|---:|---:|---:|---:|---:|")
            for arm in ("neurosym", "graphiti"):
                m = mix[arm]
                print(f"| {arm} | {pct(m['published_accuracy'])} | {pct(m['later_accuracy'])} | "
                      f"{100 * m['raw_difference']:+.1f} [{100 * m['raw_bootstrap95'][0]:+.1f}, {100 * m['raw_bootstrap95'][1]:+.1f}] | "
                      f"{pct(m['published_standardized'])} / {pct(m['later_standardized'])} | "
                      f"{100 * m['standardized_difference']:+.1f} [{100 * m['standardized_bootstrap95'][0]:+.1f}, "
                      f"{100 * m['standardized_bootstrap95'][1]:+.1f}] |")
            print("\nStratum counts, published / later: " + ", ".join(
                f"{s} {mix['published_stratum_counts'][s]}/{mix['later_stratum_counts'][s]}" for s in STRATA))


if __name__ == "__main__":
    main()
