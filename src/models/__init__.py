"""
Módulo de entrenamiento de los 3 modelos oficiales: Baseline Dummy, KNN y Random Forest.
"""
from .train import train_dummy, train_knn, train_random_forest, optimize_knn_k

__all__ = ["train_dummy", "train_knn", "train_random_forest", "optimize_knn_k"]
