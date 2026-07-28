from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from time import monotonic
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

from neurosym.domain import ExperimentResult


@dataclass(frozen=True)
class CellSpec:
    cell_id: int
    label: str
    answer_strategy: str
    context_strategy: str
    validator: str = "n/a"
    session_id: Optional[str] = None

    def __post_init__(self) -> None:
        if self.cell_id not in range(1, 7):
            raise ValueError("cell_id must be between 1 and 6")
        if self.answer_strategy not in {"flat", "rlm"}:
            raise ValueError("answer_strategy must be flat or rlm")
        if self.context_strategy not in {"raw", "kg"}:
            raise ValueError("context_strategy must be raw or kg")
        if self.context_strategy == "kg" and not self.session_id:
            raise ValueError("KG cells require a session_id")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CellSpec":
        return cls(
            cell_id=int(value["cell_id"]),
            label=str(value["label"]),
            answer_strategy=str(value.get("answer_strategy", value.get("kind", value.get("recursion", "")))),
            context_strategy=str(value.get("context_strategy", value.get("retrieval", ""))),
            validator=str(value.get("validator", "n/a")),
            session_id=value.get("session_id"),
        )


@dataclass(frozen=True)
class RunConfig:
    input_path: Path
    output_path: Path
    results_dir: Path
    run_id: Optional[str]
    limit: Optional[int]
    seed: int
    model: str
    base_url: str
    api_key: str
    memory_scope: str = "example"
    retrieval_mode: Optional[str] = None
    options: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_namespace(
        cls,
        namespace: Any,
        *,
        output_path: Path,
    ) -> "RunConfig":
        known = {
            "input",
            "output",
            "results_dir",
            "run_id",
            "limit",
            "seed",
            "model",
            "vllm_base_url",
            "api_key",
            "memory_scope",
            "retrieval_mode",
        }
        values = vars(namespace)
        return cls(
            input_path=Path(values["input"]),
            output_path=Path(output_path),
            results_dir=Path(values["results_dir"]),
            run_id=values.get("run_id"),
            limit=values.get("limit"),
            seed=int(values.get("seed", 0)),
            model=str(values["model"]),
            base_url=str(values["vllm_base_url"]),
            api_key=str(values["api_key"]),
            memory_scope=str(values.get("memory_scope", "example")),
            retrieval_mode=values.get("retrieval_mode"),
            options={key: item for key, item in values.items() if key not in known},
        )


@dataclass
class CellDependencies:
    answer: Callable[[Mapping[str, Any], str, str], Mapping[str, Any]]
    context: Callable[[Mapping[str, Any]], Mapping[str, Any]]
    clock: Callable[[], float] = monotonic
    trace_sink: Optional[Callable[[Mapping[str, Any]], None]] = None
    result_sink: Optional[Callable[[Mapping[str, Any]], None]] = None


class CellExecutor:
    def __init__(
        self,
        spec: CellSpec,
        config: RunConfig,
        dependencies: CellDependencies,
    ) -> None:
        self.spec = spec
        self.config = config
        self.dependencies = dependencies

    def execute(self, examples: Iterable[Mapping[str, Any]]) -> List[ExperimentResult]:
        results: List[ExperimentResult] = []
        for index, example in enumerate(examples, start=1):
            started = self.dependencies.clock()
            example_id = str(example.get("_id", f"ex_{index}"))
            context_value = self.dependencies.context(example)
            context = str(context_value.get("context", ""))
            question = str(context_value.get("question", example.get("question", "")))
            error: Optional[str] = None
            answer_value: Mapping[str, Any]
            try:
                answer_value = self.dependencies.answer(example, context, question)
            except Exception as exc:
                answer_value = {"predicted": ""}
                error = str(exc)
            predicted = str(answer_value.get("predicted", ""))
            gold = str(example.get("answer", "")).strip().upper()
            elapsed = self.dependencies.clock() - started
            result = ExperimentResult(
                cell_id=self.spec.cell_id,
                label=self.spec.label,
                example_id=example_id,
                predicted=predicted,
                gold=gold,
                correct=bool(predicted) and predicted == gold and error is None,
                n_context_chars=int(context_value.get("n_context_chars", len(context))),
                n_triples=int(context_value.get("n_triples", 0)),
                elapsed_seconds=round(elapsed, 6),
                error=error or answer_value.get("error"),
                run_id=self.config.run_id,
                session_id=self.spec.session_id,
                memory_scope=self.config.memory_scope,
                orchestration_mode=answer_value.get("orchestration_mode"),
                validator_backend=self.spec.validator,
                configured_retrieval_mode=self.config.retrieval_mode,
                effective_retrieval_mode=answer_value.get("effective_retrieval_mode"),
                retrieval_degraded=answer_value.get("retrieval_degraded"),
                extension_fields={
                    key: item
                    for key, item in answer_value.items()
                    if key
                    not in {
                        "predicted",
                        "error",
                        "orchestration_mode",
                        "effective_retrieval_mode",
                        "retrieval_degraded",
                    }
                },
            )
            results.append(result)
            mapping = result.to_mapping()
            if self.dependencies.trace_sink is not None:
                self.dependencies.trace_sink(mapping)
            if self.dependencies.result_sink is not None:
                self.dependencies.result_sink(mapping)
        return results


def serialize_experiment_result(
    result: ExperimentResult | Mapping[str, Any],
    schema: Sequence[str],
) -> Dict[str, Any]:
    value = result.to_mapping() if isinstance(result, ExperimentResult) else dict(result)
    return {key: value.get(key) for key in schema}
