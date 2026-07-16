from __future__ import annotations

from unittest.mock import patch

import pytest

from validator_backend import LocalValidatorBackend


def test_local_backend_fails_closed_when_actual_scallop_is_required() -> None:
    with patch("validator_backend.scallop_validator.scallopy", None):
        with pytest.raises(RuntimeError, match="actual Scallop is required"):
            LocalValidatorBackend(require_scallop=True)


def test_fallback_backend_is_explicitly_labeled() -> None:
    with patch("validator_backend.scallop_validator.scallopy", None):
        backend = LocalValidatorBackend(require_scallop=False)
    assert backend.info.name == "python_symbolic_fallback"
    assert backend.info.scallop_available is False
