from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Literal, Optional


RetrievalMode = Literal["sparse", "dense", "hybrid", "dense_ppr"]
DenseFailurePolicy = Literal["error", "sparse"]
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
FACT_TEMPLATE_VERSION = "fact_text.v1"
QUERY_TEMPLATE_VERSION = "bge_query.v1"
PPR_GRAPH_TEMPLATE_VERSION = "entity_fact.v1"


class DenseRetrievalError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class PPRRetrievalError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class EmbeddingConfig:
    model: str = DEFAULT_EMBEDDING_MODEL
    requested_revision: Optional[str] = None
    device: str = "cpu"
    batch_size: int = 32
    normalize: bool = True
    fact_template_version: str = FACT_TEMPLATE_VERSION
    query_template_version: str = QUERY_TEMPLATE_VERSION

    def __post_init__(self) -> None:
        if not str(self.model).strip():
            raise ValueError("embedding model must not be empty")
        if not str(self.device).strip():
            raise ValueError("embedding device must not be empty")
        if self.batch_size < 1:
            raise ValueError("embedding batch size must be at least 1")
        if not self.normalize:
            raise ValueError("dense retrieval requires normalized embeddings")
        if self.fact_template_version != FACT_TEMPLATE_VERSION:
            raise ValueError(f"unsupported fact template: {self.fact_template_version}")
        if self.query_template_version != QUERY_TEMPLATE_VERSION:
            raise ValueError(f"unsupported query template: {self.query_template_version}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PPRConfig:
    seed_count: int = 20
    similarity_threshold: float = 0.0
    temperature: float = 0.1
    damping: float = 0.5
    tolerance: float = 1e-8
    max_iterations: int = 100
    graph_template_version: str = PPR_GRAPH_TEMPLATE_VERSION

    def __post_init__(self) -> None:
        if self.seed_count < 1:
            raise ValueError("PPR seed count must be at least 1")
        if not -1.0 <= self.similarity_threshold <= 1.0:
            raise ValueError("PPR similarity threshold must be between -1 and 1")
        if self.temperature <= 0:
            raise ValueError("PPR temperature must be positive")
        if not 0.0 < self.damping < 1.0:
            raise ValueError("PPR damping must be between 0 and 1")
        if self.tolerance <= 0:
            raise ValueError("PPR tolerance must be positive")
        if self.max_iterations < 1:
            raise ValueError("PPR max iterations must be at least 1")
        if self.graph_template_version != PPR_GRAPH_TEMPLATE_VERSION:
            raise ValueError(f"unsupported PPR graph template: {self.graph_template_version}")

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["restart_probability"] = 1.0 - self.damping
        return value


@dataclass(frozen=True)
class RetrievalConfig:
    mode: RetrievalMode = "hybrid"
    rrf_k: int = 60
    branch_candidate_multiplier: int = 3
    branch_candidate_cap: int = 50
    index_root: Path = Path("results/dense_indexes")
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    ppr: PPRConfig = field(default_factory=PPRConfig)
    failure_policy: DenseFailurePolicy = "error"

    def __post_init__(self) -> None:
        if self.mode not in {"sparse", "dense", "hybrid", "dense_ppr"}:
            raise ValueError("retrieval mode must be sparse, dense, hybrid, or dense_ppr")
        if self.failure_policy not in {"error", "sparse"}:
            raise ValueError("dense failure policy must be error or sparse")
        if self.rrf_k < 1:
            raise ValueError("RRF k must be at least 1")
        if self.branch_candidate_multiplier < 1:
            raise ValueError("branch candidate multiplier must be at least 1")
        if self.branch_candidate_cap < 1:
            raise ValueError("branch candidate cap must be at least 1")
        object.__setattr__(self, "index_root", Path(self.index_root))

    def branch_depth(self, top_k: int) -> int:
        return min(
            max(1, int(top_k)) * self.branch_candidate_multiplier,
            self.branch_candidate_cap,
        )

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["index_root"] = str(self.index_root)
        if self.mode != "dense_ppr":
            value.pop("ppr", None)
        else:
            value["ppr"] = self.ppr.to_dict()
        return value
