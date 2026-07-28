from compiled_memory import CompiledRelationship as LegacyCompiledRelationship
from experiments.graph_context import GraphSource as LegacyGraphSource
from experiments.retrieval_config import RetrievalConfig as LegacyRetrievalConfig
from memory_artifacts import MemoryScope as LegacyMemoryScope
from neo4j_graph import Neo4jGraph as LegacyNeo4jGraph
from scallop_validator import RuleParameters as LegacyRuleParameters
from validator_backend import LocalValidatorBackend as LegacyValidatorBackend

from neurosym.adapters.graph_source import GraphSource
from neurosym.adapters.kg_search import TOOL_VERSION
from neurosym.adapters.neo4j_graph import Neo4jGraph
from neurosym.adapters.validation_backend import LocalValidatorBackend
from neurosym.adapters.working_memory_tool import TOOL_VERSION as MEMORY_TOOL_VERSION
from neurosym.domain.compiled_memory import CompiledRelationship
from neurosym.domain.memory_artifacts import MemoryScope
from neurosym.domain.retrieval_config import RetrievalConfig
from neurosym.domain.validation_rules import RuleParameters


def test_retrieval_config_legacy_import_is_canonical_symbol() -> None:
    assert LegacyRetrievalConfig is RetrievalConfig


def test_legacy_imports_are_silent_canonical_symbol_reexports() -> None:
    assert LegacyCompiledRelationship is CompiledRelationship
    assert LegacyGraphSource is GraphSource
    assert LegacyMemoryScope is MemoryScope
    assert LegacyNeo4jGraph is Neo4jGraph
    assert LegacyRuleParameters is RuleParameters
    assert LegacyValidatorBackend is LocalValidatorBackend


def test_tool_contract_versions_remain_stable() -> None:
    assert TOOL_VERSION == "search_knowledge_graph.v3"
    assert MEMORY_TOOL_VERSION == "update_working_memory.v1"
