from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence


Fact = Dict[str, Any]


def _fact_with_decision(
    fact: Mapping[str, Any],
    *,
    status: str,
    validator: str,
    reason: str,
    replaces: Optional[str],
    rule_params_version: Optional[str],
) -> Fact:
    committed = dict(fact)
    decision = dict(committed.get("decision", {})) if isinstance(
        committed.get("decision"), dict
    ) else {}
    decision.update(
        {
            "status": status,
            "validator": validator,
            "reason": reason,
            "replaces": replaces,
        }
    )
    committed["decision"] = decision
    committed["rule_params_version"] = rule_params_version or ""
    return committed


def _first_conflicting_fact(
    existing_facts: Sequence[Mapping[str, Any]],
    candidate: Mapping[str, Any],
) -> Optional[Fact]:
    from neurosym.domain.compiled_memory import facts_temporally_overlap, sanitize_predicate

    subject = str(candidate.get("subject", ""))
    predicate = sanitize_predicate(str(candidate.get("predicate", "")))
    object_value = str(candidate.get("object", ""))
    for raw_fact in existing_facts:
        fact = dict(raw_fact)
        fact_subject = str(fact.get("subject", ""))
        fact_predicate = sanitize_predicate(str(fact.get("predicate", "")))
        fact_object = str(fact.get("object", ""))
        if not facts_temporally_overlap(fact, dict(candidate)):
            continue
        if fact_subject == subject and fact_predicate == predicate and fact_object == object_value:
            return fact
        if fact_subject == subject and fact_predicate == predicate and fact_object != object_value:
            return fact
        if (
            predicate == "PART_OF"
            and fact_predicate == "PART_OF"
            and fact_subject == object_value
            and fact_object == subject
        ):
            return fact
        if (
            predicate == "IS_ALIVE"
            and fact_predicate == "IS_ALIVE"
            and fact_subject == subject
            and {fact_object.lower(), object_value.lower()} == {"true", "false"}
        ):
            return fact
    return None


class FactWriteService:
    def __init__(self, repository: Any, validator: Any) -> None:
        self.repository = repository
        self.validator = validator

    def insert(
        self,
        facts: List[Fact],
        *,
        session_id: Optional[str],
        validate: bool,
        rule_params: Any,
        derived_from_session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        sid = session_id or self.repository.session_id
        proposal = self.repository.propose_facts(facts, session_id=session_id)
        if not validate:
            accepted_facts = [
                _fact_with_decision(
                    fact,
                    status="accept",
                    validator="none",
                    reason="Accepted without validation",
                    replaces=None,
                    rule_params_version=None,
                )
                for fact in proposal["new"]
            ]
            committed = self.repository.commit_facts(accepted_facts, session_id=session_id)
            ledger = [
                self.repository.record_decision(
                    candidate=fact,
                    session_id=sid,
                    decision="accept",
                    reason="Accepted without validation",
                    rule_params=rule_params,
                    committed=True,
                    derived_from_session_id=derived_from_session_id,
                )
                for fact in proposal["new"]
            ]
            return {
                "committed": committed,
                "conflicts": proposal["conflicts"],
                "replaced": [],
                "ledger": ledger,
                "rejected": [],
            }

        # Seed the validator with relevant committed session facts, plus facts
        # accepted earlier in this batch. This makes Scallop gate against the
        # persistent graph state instead of only the current insert batch.
        existing_facts = self.repository.validation_context_for_facts(
            proposal["new"], session_id=session_id
        )
        known_keys = {
            (
                str(fact.get("fact_id", "")),
                str(fact.get("session_id", sid)),
            )
            for fact in existing_facts
        }
        for fact in proposal["existing"]:
            key = (
                str(fact.get("fact_id", "")),
                str(fact.get("session_id", sid)),
            )
            if key not in known_keys:
                existing_facts.append(fact)
                known_keys.add(key)

        valid_facts: List[Fact] = []
        replaced: List[Dict[str, Any]] = []
        ledger: List[Dict[str, Any]] = []
        rejected: List[Dict[str, Any]] = []
        for fact in proposal["new"]:
            validation = self.validator.validate(
                existing_facts, fact, rule_params=rule_params
            )
            decision = validation.decision
            reason = validation.reason
            replace_id = validation.replace_fact_id
            if decision in {"accept", "replace"}:
                if decision == "replace" and replace_id:
                    # Delete the lower-confidence existing fact, then commit the new one.
                    self.repository.delete_fact(replace_id, session_id=session_id)
                    replaced.append({"removed_fact_id": replace_id, "reason": reason})
                    # Remove from local cache so subsequent facts see the updated state.
                    existing_facts = [
                        existing
                        for existing in existing_facts
                        if existing.get("fact_id") != replace_id
                    ]
                committed_fact = _fact_with_decision(
                    fact,
                    status=decision,
                    validator=self.validator.info.name,
                    reason=reason,
                    replaces=replace_id,
                    rule_params_version=validation.rule_params_version,
                )
                valid_facts.append(committed_fact)
                existing_facts.append(committed_fact)
                ledger.append(
                    {
                        "candidate": fact,
                        "decision": decision,
                        "reason": reason,
                        "replace_fact_id": replace_id,
                        "rejection_label": None,
                    }
                )
                continue

            label = (
                validation.rejection_label.to_dict()
                if validation.rejection_label
                else None
            )
            proposal["conflicts"].append(
                {
                    "candidate": fact,
                    "reason": reason,
                    "rejection_label": label,
                    "rule_params_version": validation.rule_params_version,
                }
            )
            rejected.append(
                {
                    "candidate": fact,
                    "reason": reason,
                    "rule_fired": (label or {}).get("code", "validator_reject"),
                    "rejection_label": label,
                    "rule_params_version": validation.rule_params_version,
                    "existing_conflicting_fact": _first_conflicting_fact(
                        existing_facts, fact
                    ),
                }
            )
            ledger.append(
                {
                    "candidate": fact,
                    "decision": decision,
                    "reason": reason,
                    "replace_fact_id": replace_id,
                    "rejection_label": label,
                }
            )

        committed = self.repository.commit_facts(valid_facts, session_id=session_id)
        persisted_ledger = [
            self.repository.record_decision(
                candidate=entry["candidate"],
                session_id=sid,
                decision=entry["decision"],
                reason=entry["reason"],
                rule_params=rule_params,
                rejection_label=entry["rejection_label"],
                replace_fact_id=entry["replace_fact_id"],
                committed=entry["decision"] in {"accept", "replace"},
                derived_from_session_id=derived_from_session_id,
            )
            for entry in ledger
        ]
        return {
            "committed": committed,
            "conflicts": proposal["conflicts"],
            "replaced": replaced,
            "ledger": persisted_ledger,
            "rejected": rejected,
        }
