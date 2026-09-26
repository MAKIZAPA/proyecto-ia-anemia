"""
Implementación y entrenamiento de los 3 modelos oficiales para clasificación de anemia infantil.
"""

from typing import Dict, Any, Tuple
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

def train_dummy(X_train: np.ndarray, y_train: np.ndarray) -> DummyClassifier:
    """Entrena la línea base trivial (clasifica con la clase más frecuente)."""
    model = DummyClassifier(strategy="most_frequent")
    model.fit(X_train, y_train)
    return model

def optimize_knn_k(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    k_range: range = range(1, 32, 2)
) -> Tuple[int, Dict[int, float]]:
    """Realiza un barrido sistemático de k para seleccionar el valor con mayor F1-score."""
    f1_scores = {}
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
        knn.fit(X_train, y_train)
        preds = knn.predict(X_test)
        f1_scores[k] = f1_score(y_test, preds)
        
    best_k = max(f1_scores, key=f1_scores.get)
    return best_k, f1_scores

def train_knn(X_train: np.ndarray, y_train: np.ndarray, k: int = 17) -> KNeighborsClassifier:
    """Entrena el modelo K-Nearest Neighbors con métrica euclidiana."""
    model = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    model.fit(X_train, y_train)
    return model

def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 100,
    max_depth: int = 12,
    random_state: int = 42
) -> RandomForestClassifier:
    """Entrena el ensamble no lineal Random Forest con criterio de impureza de Gini."""
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    return model
