from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence

import numpy as np

from neurosym.domain.retrieval_config import DenseRetrievalError, EmbeddingConfig


INDEX_SCHEMA_VERSION = "dense_fact_index.v1"
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class EmbeddingProvider(Protocol):
    def encode_documents(self, texts: Sequence[str]) -> np.ndarray: ...
    def encode_query(self, text: str) -> np.ndarray: ...
    def metadata(self) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class EmbeddingTextWindow:
    text: str
    token_start: int
    token_end: int
    token_count: int


def fact_text_v1(fact: Mapping[str, Any]) -> str:
    return (
        f"subject: {fact.get('subject', '')}\n"
        f"predicate: {fact.get('predicate', '')}\n"
        f"object: {fact.get('object', '')}\n"
        f"support: {fact.get('support_text', '')}"
    )


def query_text_v1(query: str) -> str:
    return f"{BGE_QUERY_INSTRUCTION}{query}"


def _normalized_matrix(value: Any, *, rows: Optional[int] = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[1] < 1:
        raise DenseRetrievalError("dense_backend_failure", "embedding backend returned an invalid shape")
    if rows is not None and array.shape[0] != rows:
        raise DenseRetrievalError("dense_backend_failure", "embedding backend returned an invalid row count")
    if not np.isfinite(array).all():
        raise DenseRetrievalError("dense_backend_failure", "embedding backend returned NaN or infinity")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise DenseRetrievalError("dense_backend_failure", "embedding backend returned a zero vector")
    return np.ascontiguousarray(array / norms, dtype=np.float32)


def _provider_metadata(provider: EmbeddingProvider, config: EmbeddingConfig) -> Dict[str, Any]:
    try:
        metadata = dict(provider.metadata())
        dimension = int(metadata["vector_dimension"])
        resolved_revision = str(metadata["resolved_revision"]).strip()
        package_version = str(metadata["sentence_transformers_version"]).strip()
    except (KeyError, TypeError, ValueError) as exc:
        raise DenseRetrievalError(
            "dense_backend_failure",
            "embedding provider metadata is invalid",
        ) from exc
    if str(metadata.get("model_name")) != config.model:
        raise DenseRetrievalError("dense_backend_failure", "embedding provider model differs from configuration")
    if metadata.get("requested_revision") != config.requested_revision:
        raise DenseRetrievalError("dense_backend_failure", "embedding provider revision differs from configuration")
    if not resolved_revision or resolved_revision.lower() in {"none", "unresolved"}:
        raise DenseRetrievalError("dense_backend_failure", "embedding provider resolved revision is invalid")
    if not package_version or dimension < 1:
        raise DenseRetrievalError("dense_backend_failure", "embedding provider metadata is invalid")
    metadata["resolved_revision"] = resolved_revision
    metadata["sentence_transformers_version"] = package_version
    metadata["vector_dimension"] = dimension
    return metadata


class SentenceTransformerEmbedder:
    def __init__(self, config: EmbeddingConfig, *, local_files_only: bool = False) -> None:
        self.config = config
        self.local_files_only = local_files_only
        self._model: Any = None
        self._metadata: Optional[Dict[str, Any]] = None

    def _load(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise DenseRetrievalError(
                    "dense_backend_failure",
                    "sentence-transformers is required for dense retrieval",
                ) from exc
            try:
                self._model = SentenceTransformer(
                    self.config.model,
                    revision=self.config.requested_revision,
                    device=self.config.device,
                    local_files_only=self.local_files_only,
                )
            except Exception as exc:
                raise DenseRetrievalError("dense_backend_failure", "embedding model could not be loaded") from exc
        return self._model

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        model = self._load()
        try:
            value = model.encode(
                list(texts),
                batch_size=self.config.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=False,
                show_progress_bar=False,
            )
        except Exception as exc:
            raise DenseRetrievalError("dense_backend_failure", "embedding backend failed") from exc
        return _normalized_matrix(value, rows=len(texts))

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts)

    def encode_query(self, text: str) -> np.ndarray:
        return self._encode([text])[0]

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts)

    def document_windows(
        self,
        text: str,
        *,
        max_tokens: int,
        overlap_tokens: int,
    ) -> List[EmbeddingTextWindow]:
        if max_tokens < 1:
            raise ValueError("embedding window size must be at least 1")
        if overlap_tokens < 0 or overlap_tokens >= max_tokens:
            raise ValueError(
                "embedding window overlap must be non-negative and smaller "
                "than the window size"
            )

        model = self._load()
        tokenizer = model.tokenizer
        special_tokens = int(tokenizer.num_special_tokens_to_add(pair=False))
        model_limit = int(model.max_seq_length)
        if max_tokens + special_tokens > model_limit:
            raise ValueError(
                "embedding window plus special tokens exceeds the model's "
                f"{model_limit}-token sequence limit"
            )
        try:
            encoded = tokenizer(
                str(text),
                add_special_tokens=False,
                return_offsets_mapping=True,
                truncation=False,
                verbose=False,
            )
            token_ids = list(encoded["input_ids"])
            offsets = list(encoded["offset_mapping"])
        except (KeyError, TypeError, ValueError, NotImplementedError) as exc:
            raise DenseRetrievalError(
                "dense_backend_failure",
                "embedding tokenizer could not produce token offsets",
            ) from exc
        if len(token_ids) != len(offsets):
            raise DenseRetrievalError(
                "dense_backend_failure",
                "embedding tokenizer returned inconsistent token offsets",
            )
        if not token_ids:
            return [
                EmbeddingTextWindow(
                    text=str(text),
                    token_start=0,
                    token_end=0,
                    token_count=0,
                )
            ]

        windows: List[EmbeddingTextWindow] = []
        stride = max_tokens - overlap_tokens
        token_start = 0
        while token_start < len(token_ids):
            token_end = min(token_start + max_tokens, len(token_ids))
            char_start = int(offsets[token_start][0])
            char_end = int(offsets[token_end - 1][1])
            windows.append(
                EmbeddingTextWindow(
                    text=str(text)[char_start:char_end],
                    token_start=token_start,
                    token_end=token_end,
                    token_count=token_end - token_start,
                )
            )
            if token_end == len(token_ids):
                break
            token_start += stride
        return windows

    def metadata(self) -> Mapping[str, Any]:
        if self._metadata is None:
            model = self._load()
            resolved = None
            try:
                module = model._first_module()
                auto_model = module.auto_model
                config = getattr(auto_model, "config", None)
                tokenizer = getattr(module, "tokenizer", None)
                resolved = (
                    getattr(config, "_commit_hash", None)
                    or getattr(auto_model, "_commit_hash", None)
                    or getattr(tokenizer, "init_kwargs", {}).get("_commit_hash")
                )
                if not resolved:
                    paths = [
                        str(getattr(config, "_name_or_path", "")),
                        str(getattr(auto_model, "name_or_path", "")),
                    ]
                    for value in paths:
                        match = re.search(r"[\\/]snapshots[\\/]([^\\/]+)", value)
                        if match:
                            resolved = match.group(1)
                            break
            except Exception:
                resolved = None
            if not resolved:
                raise DenseRetrievalError(
                    "dense_backend_failure",
                    "embedding model resolved commit could not be determined",
                )
            dimension = int(model.get_sentence_embedding_dimension())
            self._metadata = {
                "model_name": self.config.model,
                "requested_revision": self.config.requested_revision,
                "resolved_revision": str(resolved),
                "sentence_transformers_version": importlib.metadata.version("sentence-transformers"),
                "vector_dimension": dimension,
                "max_sequence_length": int(model.max_seq_length),
            }
        return dict(self._metadata)


@dataclass(frozen=True)
class DenseIndexManifest:
    schema_version: str
    model_name: str
    requested_revision: Optional[str]
    resolved_revision: str
    sentence_transformers_version: str
    embedding_device: str
    embedding_batch_size: int
    fact_template_version: str
    query_template_version: str
    vector_dimension: int
    dtype: str
    normalized: bool
    similarity: str
    source_session_id: str
    source_snapshot_sha256: str
    ordered_fact_digest: str
    fact_count: int
    built_at_utc: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DenseIndexManifest":
        try:
            return cls(**{name: value[name] for name in cls.__dataclass_fields__})
        except (KeyError, TypeError, ValueError) as exc:
            raise DenseRetrievalError("dense_index_corrupt", "dense index manifest is invalid") from exc

    @property
    def identity(self) -> Dict[str, Any]:
        value = self.to_dict()
        value.pop("built_at_utc", None)
        return value


def _remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _recover_atomic_swap(target: Path) -> None:
    previous = target.with_name(f".{target.name}.previous")
    if target.exists():
        _remove_path(previous)
    elif previous.exists():
        os.replace(previous, target)


def ordered_fact_digest(facts: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for fact in facts:
        digest.update(str(fact.get("fact_id", "")).encode("utf-8"))
        digest.update(b"\0")
        digest.update(fact_text_v1(fact).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def snapshot_sha256(facts: Sequence[Mapping[str, Any]]) -> str:
    payload = "".join(
        json.dumps(dict(fact), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for fact in facts
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def session_index_path(root: Path, session_id: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(session_id)).strip("._") or "session"
    suffix = hashlib.sha256(str(session_id).encode("utf-8")).hexdigest()[:10]
    return Path(root) / f"{safe}-{suffix}"


class DenseFactIndex:
    def __init__(self, vectors: np.ndarray, facts: List[Dict[str, Any]], manifest: DenseIndexManifest, provider: EmbeddingProvider) -> None:
        self.vectors = vectors
        self.facts = facts
        self.manifest = manifest
        self.provider = provider

    @classmethod
    def build(cls, path: Path, facts: Sequence[Mapping[str, Any]], provider: EmbeddingProvider, *, config: EmbeddingConfig, source_session_id: str, source_snapshot_sha256: str) -> "DenseFactIndex":
        rows = [dict(fact) for fact in facts]
        fact_ids = [str(row.get("fact_id", "")) for row in rows]
        if any(not value for value in fact_ids) or len(set(fact_ids)) != len(fact_ids):
            raise DenseRetrievalError("dense_index_mismatch", "dense index facts require unique non-empty fact IDs")
        metadata = _provider_metadata(provider, config)
        texts = [fact_text_v1(row) for row in rows]
        dimension = int(metadata["vector_dimension"])
        vectors = provider.encode_documents(texts) if rows else np.empty((0, dimension), dtype=np.float32)
        if rows:
            vectors = _normalized_matrix(vectors, rows=len(rows))
        if vectors.shape != (len(rows), dimension):
            raise DenseRetrievalError("dense_backend_failure", "embedding dimension does not match provider metadata")
        manifest = DenseIndexManifest(
            schema_version=INDEX_SCHEMA_VERSION,
            model_name=config.model,
            requested_revision=config.requested_revision,
            resolved_revision=str(metadata["resolved_revision"]),
            sentence_transformers_version=str(metadata["sentence_transformers_version"]),
            embedding_device=config.device,
            embedding_batch_size=config.batch_size,
            fact_template_version=config.fact_template_version,
            query_template_version=config.query_template_version,
            vector_dimension=dimension,
            dtype="float32",
            normalized=True,
            similarity="cosine_dot_product",
            source_session_id=str(source_session_id),
            source_snapshot_sha256=str(source_snapshot_sha256),
            ordered_fact_digest=ordered_fact_digest(rows),
            fact_count=len(rows),
            built_at_utc=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        )
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        _recover_atomic_swap(target)
        previous = target.with_name(f".{target.name}.previous")
        temporary = target.with_name(f".{target.name}.tmp-{uuid.uuid4().hex}")
        temporary.mkdir()
        try:
            np.save(temporary / "vectors.npy", vectors, allow_pickle=False)
            with (temporary / "facts.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
            (temporary / "manifest.json").write_text(json.dumps(manifest.to_dict(), ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
            if target.exists():
                os.replace(target, previous)
            try:
                os.replace(temporary, target)
            except Exception:
                if previous.exists() and not target.exists():
                    os.replace(previous, target)
                raise
            _remove_path(previous)
        finally:
            _remove_path(temporary)
        return cls(vectors, rows, manifest, provider)

    @classmethod
    def load(
        cls,
        path: Path,
        provider: EmbeddingProvider,
        *,
        config: EmbeddingConfig,
        source_session_id: str,
        source_snapshot_sha256: str,
    ) -> "DenseFactIndex":
        root = Path(path)
        _recover_atomic_swap(root)
        if not root.is_dir():
            raise DenseRetrievalError("dense_index_unavailable", "dense index directory is unavailable")
        try:
            manifest_value = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
            manifest = DenseIndexManifest.from_dict(manifest_value)
        except DenseRetrievalError:
            raise
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            raise DenseRetrievalError("dense_index_corrupt", "dense index manifest cannot be read") from exc
        metadata = _provider_metadata(provider, config)
        expected = {
            "schema_version": INDEX_SCHEMA_VERSION,
            "model_name": config.model,
            "requested_revision": config.requested_revision,
            "resolved_revision": str(metadata["resolved_revision"]),
            "sentence_transformers_version": str(metadata["sentence_transformers_version"]),
            "embedding_device": config.device,
            "embedding_batch_size": config.batch_size,
            "fact_template_version": config.fact_template_version,
            "query_template_version": config.query_template_version,
            "vector_dimension": int(metadata["vector_dimension"]),
            "dtype": "float32",
            "normalized": True,
            "similarity": "cosine_dot_product",
            "source_session_id": str(source_session_id),
            "source_snapshot_sha256": str(source_snapshot_sha256),
        }
        mismatches = [
            name for name, value in expected.items()
            if getattr(manifest, name) != value
        ]
        if mismatches:
            raise DenseRetrievalError(
                "dense_index_mismatch",
                f"dense index identity differs: {', '.join(sorted(mismatches))}",
            )
        try:
            vectors = np.load(root / "vectors.npy", allow_pickle=False)
            facts = []
            with (root / "facts.jsonl").open(encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        value = json.loads(line)
                        if not isinstance(value, dict):
                            raise TypeError("fact row must be an object")
                        facts.append(value)
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            raise DenseRetrievalError("dense_index_corrupt", "dense index data cannot be read safely") from exc
        if vectors.dtype != np.float32 or vectors.ndim != 2:
            raise DenseRetrievalError("dense_index_corrupt", "dense index vectors have an invalid dtype or shape")
        if vectors.shape != (manifest.fact_count, manifest.vector_dimension):
            raise DenseRetrievalError("dense_index_mismatch", "dense index vector shape differs from its manifest")
        if len(facts) != manifest.fact_count:
            raise DenseRetrievalError("dense_index_mismatch", "dense index fact count differs from its manifest")
        fact_ids = [str(fact.get("fact_id", "")) for fact in facts]
        if any(not value for value in fact_ids) or len(set(fact_ids)) != len(fact_ids):
            raise DenseRetrievalError("dense_index_corrupt", "dense index contains invalid fact IDs")
        if ordered_fact_digest(facts) != manifest.ordered_fact_digest:
            raise DenseRetrievalError("dense_index_mismatch", "dense index ordered fact digest differs")
        if not np.isfinite(vectors).all():
            raise DenseRetrievalError("dense_index_corrupt", "dense index contains NaN or infinity")
        if len(vectors):
            norms = np.linalg.norm(vectors, axis=1)
            if not np.allclose(norms, 1.0, rtol=1e-4, atol=1e-5):
                raise DenseRetrievalError("dense_index_corrupt", "dense index vectors are not normalized")
        return cls(np.ascontiguousarray(vectors), facts, manifest, provider)

    def search(
        self,
        query: str,
        *,
        top_k: int,
        scope: Any,
        predicates: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, Any]]:
        wanted = {
            re.sub(r"_+", "_", re.sub(r"[^A-Z0-9_]+", "_", str(value).strip().upper())).strip("_")
            for value in predicates or []
            if str(value).strip()
        }
        eligible: List[int] = []
        sessions = set(str(value) for value in scope.session_ids)
        for index, fact in enumerate(self.facts):
            fact_session = str(fact.get("session_id", ""))
            if fact_session not in sessions:
                continue
            if scope.mode == "example" and str(fact.get("example_id", "")) != str(scope.example_id):
                continue
            if wanted and str(fact.get("predicate", "")).upper() not in wanted:
                continue
            eligible.append(index)
        if not eligible or top_k < 1:
            return []
        query_vector = _normalized_matrix(
            self.provider.encode_query(query_text_v1(query)), rows=1
        )[0]
        if query_vector.shape[0] != self.manifest.vector_dimension:
            raise DenseRetrievalError("dense_backend_failure", "query embedding dimension does not match the index")
        candidate_vectors = self.vectors[np.asarray(eligible, dtype=np.int64)]
        similarities = candidate_vectors @ query_vector
        ranked = sorted(
            zip(eligible, similarities.tolist()),
            key=lambda item: (-float(item[1]), str(self.facts[item[0]].get("fact_id", ""))),
        )
        output: List[Dict[str, Any]] = []
        for row_index, similarity in ranked[:top_k]:
            row = dict(self.facts[row_index])
            row["_dense_similarity"] = float(similarity)
            row["_dense_index_identity"] = self.manifest.identity
            output.append(row)
        return output


def load_facts_snapshot(path: Path, *, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    try:
        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise TypeError("fact row must be an object")
                    row = dict(value)
                    if session_id and not row.get("session_id"):
                        row["session_id"] = session_id
                    rows.append(row)
    except (OSError, TypeError, json.JSONDecodeError) as exc:
        raise DenseRetrievalError("dense_index_unavailable", "fact snapshot cannot be read") from exc
    return rows


def ensure_dense_index(
    *,
    index_root: Path,
    session_id: str,
    facts: Sequence[Mapping[str, Any]],
    config: EmbeddingConfig,
    provider: Optional[EmbeddingProvider] = None,
    source_hash: Optional[str] = None,
) -> DenseFactIndex:
    rows = [dict(fact) for fact in facts]
    for row in rows:
        if not row.get("session_id"):
            row["session_id"] = session_id
    rows.sort(key=lambda row: (str(row.get("session_id", "")), str(row.get("example_id", "")), str(row.get("fact_id", ""))))
    digest = source_hash or snapshot_sha256(rows)
    embedder = provider or SentenceTransformerEmbedder(config)
    path = session_index_path(index_root, session_id)
    try:
        return DenseFactIndex.load(
            path,
            embedder,
            config=config,
            source_session_id=session_id,
            source_snapshot_sha256=digest,
        )
    except DenseRetrievalError:
        return DenseFactIndex.build(
            path,
            rows,
            embedder,
            config=config,
            source_session_id=session_id,
            source_snapshot_sha256=digest,
        )


def ensure_dense_index_from_snapshot(
    snapshot_path: Path,
    *,
    index_root: Path,
    session_id: str,
    config: EmbeddingConfig,
    provider: Optional[EmbeddingProvider] = None,
) -> DenseFactIndex:
    rows = load_facts_snapshot(snapshot_path, session_id=session_id)
    source_hash = hashlib.sha256(Path(snapshot_path).read_bytes()).hexdigest()
    return ensure_dense_index(
        index_root=index_root,
        session_id=session_id,
        facts=rows,
        config=config,
        provider=provider,
        source_hash=source_hash,
    )
