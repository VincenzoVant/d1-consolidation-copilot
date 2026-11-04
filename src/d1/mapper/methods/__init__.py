"""The mapping methods, all behind one interface (base.py), found by name in METHOD_REGISTRY."""

from d1.mapper.methods import baselines  # noqa: F401  (importing a method's module registers it)
from d1.mapper.methods.base import METHOD_REGISTRY, AccountIn, MappingMethod, MethodOutput

__all__ = ["METHOD_REGISTRY", "AccountIn", "MappingMethod", "MethodOutput"]
