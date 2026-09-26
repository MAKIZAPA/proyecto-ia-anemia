"""
Métricas clínicas (Recall, Precisión, F1, ROC-AUC), Validación Cruzada y Matrices de Confusión.
"""

from typing import Dict, Any, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)

def evaluate_model(model: Any, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    """Calcula las métricas de rendimiento clínico sobre el conjunto de prueba independiente."""
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
    else:
        auc = 0.50
        
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1-Score": f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC": auc
    }

def compute_cv_scores(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42
) -> Tuple:
    """Calcula la validación cruzada estratificada de 5 pliegues para verificar estabilidad."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1')
    return float(scores.mean()), float(scores.std())

def plot_roc_curves(
    models_dict: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    output_path: str
):
    """Genera la curva ROC de los 3 modelos oficiales (Baseline Dummy, KNN, Random Forest)."""
    plt.figure(figsize=(7, 5), dpi=300)
    
    colors_map = {
        "Random Forest (100 árboles)": '#2E7D32',
        "K-Nearest Neighbors (k=17)": '#1565C0',
    }
    
    for name, model in models_dict.items():
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            auc_val = roc_auc_score(y_test, y_proba)
            plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.4f})", color=colors_map.get(name, '#333333'), linewidth=2.0)
            
    plt.plot([0, 1], [0, 1], 'k--', label='Baseline Dummy (AUC = 0.5000)', linewidth=1.5)
    plt.xlabel('Tasa de Falsos Positivos (1 - Especificidad)', fontsize=10)
    plt.ylabel('Tasa de Verdaderos Positivos (Sensibilidad / Recall)', fontsize=10)
    plt.title('Curvas ROC - Clasificación de Anemia Infantil (ENDES Perú)', fontsize=11, fontweight='bold', pad=10)
    plt.legend(loc='lower right', frameon=True, fontsize=9)
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_confusion_matrices(
    cm_knn: np.ndarray,
    cm_rf: np.ndarray,
    output_path: str,
    k: int = 17
):
    """Genera las matrices de confusión con recuentos y porcentajes clínicos."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), dpi=300)
    
    def _draw_single_cm(cm, ax, title, cmap):
        ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.set_title(title, fontsize=10.5, fontweight='bold', pad=8)
        tick_marks = np.arange(2)
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(['No Anemia', 'Anemia'], fontsize=9.5)
        ax.set_yticklabels(['No Anemia', 'Anemia'], fontsize=9.5)
        
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                val = cm[i, j]
                pct = val / np.sum(cm[i, :]) * 100
                txt = f"{val:,}\n({pct:.1f}%)"
                ax.text(j, i, txt, ha="center", va="center",
                        color="white" if val > thresh else "black",
                        fontsize=9, fontweight='semibold')
        ax.set_ylabel('Clase Real (Observada)', fontsize=9.5)
        ax.set_xlabel('Clase Predicha (Modelo)', fontsize=9.5)

    _draw_single_cm(cm_knn, axes[0], f'Matriz de Confusión - KNN (k={k})', plt.cm.Blues)
    _draw_single_cm(cm_rf, axes[1], 'Matriz de Confusión - Random Forest (100 árboles)', plt.cm.Greens)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
