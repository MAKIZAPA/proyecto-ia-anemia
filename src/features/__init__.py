"""
Módulo de ingeniería de características, partición estratificada y escalado anti-leakage.
"""
from .builder import prepare_features, split_and_scale

__all__ = ["prepare_features", "split_and_scale"]
