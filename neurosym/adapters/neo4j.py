from __future__ import annotations

from typing import Any, Dict, List, Optional


class Neo4jFactRepository:
    def __init__(self, graph: Any) -> None:
        self.graph = graph

    @property
    def session_id(self) -> str:
        return str(self.graph.session_id)

    def propose_facts(
        self, facts: List[Dict[str, Any]], *, session_id: Optional[str] = None
    ) -> Dict[str, List[Any]]:
        return self.graph.propose_facts(facts, session_id=session_id)

    def commit_facts(
        self, facts: List[Dict[str, Any]], *, session_id: Optional[str] = None
    ) -> int:
        return self.graph.commit_facts(facts, session_id=session_id)

    def validation_context_for_facts(
        self, facts: List[Dict[str, Any]], *, session_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        return self.graph.validation_context_for_facts(facts, session_id=session_id)

    def delete_fact(self, fact_id: str, *, session_id: Optional[str] = None) -> None:
        self.graph._delete_fact(fact_id, session_id=session_id)

    def record_decision(self, **values: Any) -> Dict[str, Any]:
        return self.graph._record_decision_ledger(**values)

    def reconcile_fact_decisions(self, *, session_id: Optional[str] = None) -> int:
        return self.graph.reconcile_fact_decisions(session_id=session_id)


class Neo4jConnection:
    def __init__(self, driver: Any, database: Optional[str] = None) -> None:
        self.driver = driver
        self.database = database

    def session(self) -> Any:
        if self.database:
            return self.driver.session(database=self.database)
        return self.driver.session()

    def close(self) -> None:
        self.driver.close()
