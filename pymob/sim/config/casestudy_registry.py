from __future__ import annotations
from typing import Dict, Type
from pydantic import BaseModel

# Global registry mapping case‑study identifiers to their Pydantic config models.
_registry: Dict[str, Type[BaseModel]] = {}

def register_case_study_config(name: str, model_cls: Type[BaseModel]) -> None:
    """
    Register a Pydantic model for a case-study-specific configuration section.

    Parameters
    ----------
    name: str
        Identifier matching ``case_study.name`` (e.g. ``lotka_volterra_case_study``).
    model_cls: Type[BaseModel]
        The Pydantic model that defines the expected options for the section.
    """
    _registry[name] = model_cls

def get_case_study_model(name: str):
    """Return the registered model class for *name*, or ``None`` if not registered."""
    return _registry.get(name)
