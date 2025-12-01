"""gempyor2 vectorized compartmental modeling package."""

from __future__ import annotations

from .parser import (
    DynamicRule,
    EngineConfig,
    ModelSpec,
    NormalizedIR,
    Structure,
    parse_model_dict,
    parse_model_yaml,
)

__all__ = [
    "DynamicRule",
    "EngineConfig",
    "ModelSpec",
    "NormalizedIR",
    "Structure",
    "parse_model_dict",
    "parse_model_yaml",
]

__version__ = "0.1.0"
