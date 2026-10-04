from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple


@dataclass(frozen=True)
class RuleParameters:
    version: str = "rules.v1"
    functional_predicates: Tuple[str, ...] = (
        "CAPITAL_IS",
        "BORN_IN",
        "BIRTH_DATE",
        "DEATH_DATE",
        "DIED_IN",
        "FOUNDED_IN",
        "LOCATED_IN",
        "HAS_ISO_CODE",
        "HAS_GLOTTOCODE",
    )
    generic_objects: Tuple[str, ...] = (
        "unknown",
        "n/a",
        "none",
        "various",
        "multiple",
        "several",
        "many",
        "some",
        "thing",
        "entity",
    )
    confidence_weights: Dict[str, int] = field(
        default_factory=lambda: {
            "supported": 3,
            "uncertain": 2,
            "rejected": 1,
        }
    )
    provenance_weight: int = 1
    confidence_multiplier: int = 10

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "functional_predicates": list(self.functional_predicates),
            "generic_objects": list(self.generic_objects),
            "confidence_weights": dict(self.confidence_weights),
            "provenance_weight": self.provenance_weight,
            "confidence_multiplier": self.confidence_multiplier,
        }


DEFAULT_RULE_PARAMETERS = RuleParameters()
