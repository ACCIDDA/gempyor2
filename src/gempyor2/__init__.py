"""gempyor2 vectorized compartmental modeling package."""

from __future__ import annotations

from .parser import (
    EngineConfig,
    DynamicRule,
    Structure,
    ModelSpec,
    NormalizedIR,
    parse_model_yaml,
    parse_model_dict,
)

__all__ = [
    "EngineConfig",
    "DynamicRule",
    "Structure",
    "ModelSpec",
    "NormalizedIR",
    "parse_model_yaml",
    "parse_model_dict",
]

__version__ = "0.1.0"
