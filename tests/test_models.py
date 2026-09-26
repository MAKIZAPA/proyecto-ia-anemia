"""
Pruebas unitarias de carga de modelos serializados e inferencia clínica.
"""

import os
import joblib
import numpy as np
import pytest

MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models"
)

def test_models_exist():
    """Verifica que los modelos exportados .joblib existan en disco."""
    assert os.path.exists(os.path.join(MODELS_DIR, "scaler.joblib"))
    assert os.path.exists(os.path.join(MODELS_DIR, "modelo_knn.joblib"))
    assert os.path.exists(os.path.join(MODELS_DIR, "modelo_rf.joblib"))

def test_inference_pipeline():
    """Valida la inferencia end-to-end con un caso pediátrico sintético."""
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib"))
    rf = joblib.load(os.path.join(MODELS_DIR, "modelo_rf.joblib"))
    knn = joblib.load(os.path.join(MODELS_DIR, "modelo_knn.joblib"))
    
    # Paciente de prueba: Puno (3,825 msnm), 24 meses, Varón (1), Rural (1), 11.2 kg, 81 cm, Hb 11.8 g/dL
    paciente = np.array([[11.8, 3825, 24, 1, 1, 11.2, 81.0]])
    paciente_scaled = scaler.transform(paciente)
    
    pred_rf = rf.predict(paciente_scaled)
    proba_rf = rf.predict_proba(paciente_scaled)
    
    pred_knn = knn.predict(paciente_scaled)
    proba_knn = knn.predict_proba(paciente_scaled)
    
    assert pred_rf[0] in [0, 1]
    assert pred_knn[0] in [0, 1]
    assert proba_rf.shape == (1, 2)
    assert proba_knn.shape == (1, 2)
    assert np.isclose(np.sum(proba_rf), 1.0)
    assert np.isclose(np.sum(proba_knn), 1.0)
