"""Base mixin and Result helpers for analysis estimators."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from embedkit.utils.validation import validate_embeddings


class BaseAnalyzer:
    """Mixin enforcing fit(X) -> Result convention."""

    def _prepare(self, X, min_n: int = 2) -> np.ndarray:
        """Validate and convert to a float32 ndarray (see ``validate_embeddings``)."""
        return validate_embeddings(X, min_samples=min_n, stacklevel=4)


@dataclass(frozen=True)
class BaseResult:
    def to_dict(self) -> dict[str, Any]:
        def _convert(v):
            if isinstance(v, np.ndarray):
                return v.tolist()
            if isinstance(v, (np.floating, float)):
                return float(v)
            if isinstance(v, (np.integer, int)):
                return int(v)
            if isinstance(v, BaseResult):
                return v.to_dict()
            if isinstance(v, dict):
                return {k: _convert(val) for k, val in v.items()}
            if isinstance(v, list):
                return [_convert(i) for i in v]
            return v

        return {k: _convert(v) for k, v in asdict(self).items()}
