#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional

from eval.longbench_loader import load_examples
from eval.official_scorer_adapter import write_official_scorer_input
from eval.official_scorer_runner import run_official_scorer, write_scorer_summary
from eval.vanilla_rag import run_vanilla_rag, write_predictions


@dataclass
class RagConfig:
    dataset_path: Optional[Path]
    dataset_name: Optional[str]
    split: str
    max_examples: Optional[int]
    output_dir: Path
    output_name: str
    model: str
    vllm_base_url: str
    api_key: str
    chunk_chars: int
    top_k: int
    temperature: float
    max_tokens: int
    use_official_scorer: bool = False
    official_scorer_script: Optional[Path] = None
    official_scorer_working_dir: Optional[Path] = None

def _parse_official_stdout(stdout: str) -> Dict[str, object]:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    for line in reversed(lines):
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return {}


def run_rag(config: RagConfig) -> Dict[str, Path]:
    examples = load_examples(
        dataset_path=config.dataset_path,
        dataset_name=config.dataset_name,
        split=config.split,
        max_examples=config.max_examples,
    )
    rows = run_vanilla_rag(
        examples,
        model=config.model,
        vllm_base_url=config.vllm_base_url,
        api_key=config.api_key,
        chunk_chars=config.chunk_chars,
        top_k=config.top_k,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    pred_path = config.output_dir / f"{config.output_name}.predictions.jsonl"
    meta_path = config.output_dir / f"{config.output_name}.run_config.json"
    write_predictions(pred_path, rows)
    artifacts: Dict[str, Path] = {"predictions": pred_path, "run_config": meta_path}

    if config.use_official_scorer:
        if config.official_scorer_script is None:
            raise ValueError("--use-official-scorer requires --official-scorer-script.")
        official_input_path = config.output_dir / f"{config.output_name}.official_input.jsonl"
        official_result_path = config.output_dir / f"{config.output_name}.official_result.json"
        scorer_log_path = config.output_dir / f"{config.output_name}.official_scorer_log.json"
        write_official_scorer_input(official_input_path, rows)
        scorer_payload = run_official_scorer(
            config.official_scorer_script,
            official_input_path,
            output_json_path=official_result_path,
            cwd=config.official_scorer_working_dir,
        )
        parsed_stdout = _parse_official_stdout(str(scorer_payload.get("stdout", "")))
        if parsed_stdout:
            scorer_payload["parsed_stdout_json"] = parsed_stdout
        write_scorer_summary(scorer_log_path, scorer_payload)
        artifacts["official_input"] = official_input_path
        artifacts["official_result"] = official_result_path
        artifacts["official_log"] = scorer_log_path

    meta_path.write_text(json.dumps(asdict(config), indent=2, default=str), encoding="utf-8")
    return artifacts


def parse_args() -> RagConfig:
    parser = argparse.ArgumentParser(description="Run vanilla BM25 RAG baseline.")
    parser.add_argument("--dataset-path", type=Path, default=None, help="Path to local JSON/JSONL.")
    parser.add_argument("--dataset-name", type=str, default=None, help="HF dataset name, e.g. zai-org/LongBench-v2.")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("eval_outputs"))
    parser.add_argument("--output-name", type=str, default="rag_baseline")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--vllm-base-url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--api-key", type=str, default=os.getenv("VLLM_API_KEY", "EMPTY"))
    parser.add_argument("--chunk-chars", type=int, default=2500)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--use-official-scorer", action="store_true")
    parser.add_argument("--official-scorer-script", type=Path, default=None)
    parser.add_argument("--official-scorer-working-dir", type=Path, default=None)
    args = parser.parse_args()

    if not args.dataset_path and not args.dataset_name:
        parser.error("Provide --dataset-path or --dataset-name.")
    if not args.model:
        parser.error("--model is required.")

    return RagConfig(
        dataset_path=args.dataset_path,
        dataset_name=args.dataset_name,
        split=args.split,
        max_examples=args.max_examples,
        output_dir=args.output_dir,
        output_name=args.output_name,
        model=args.model,
        vllm_base_url=args.vllm_base_url,
        api_key=args.api_key,
        chunk_chars=args.chunk_chars,
        top_k=args.top_k,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        use_official_scorer=args.use_official_scorer,
        official_scorer_script=args.official_scorer_script,
        official_scorer_working_dir=args.official_scorer_working_dir,
    )


def main() -> None:
    config = parse_args()
    paths = run_rag(config)
    print(f"Wrote RAG predictions to {paths['predictions']}")
    if "official_result" in paths:
        print(f"Wrote official scorer output to {paths['official_result']}")


if __name__ == "__main__":
    main()
