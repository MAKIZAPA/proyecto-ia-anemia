"""
Módulo de evaluación diagnóstica, métricas de salud, validación cruzada y visualizaciones.
"""
from .metrics import evaluate_model, compute_cv_scores, plot_roc_curves, plot_confusion_matrices

__all__ = ["evaluate_model", "compute_cv_scores", "plot_roc_curves", "plot_confusion_matrices"]
