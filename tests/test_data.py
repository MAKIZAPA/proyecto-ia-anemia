"""
Pruebas de integridad del dataset oficial de la ENDES (INEI Perú).
"""

import os
import pytest
import pandas as pd

DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data/processed/anemia_infantil_peru_endes.csv"
)

EXPECTED_FEATURES = [
    'Hemoglobina_Observada',
    'Altitud_msnm',
    'Edad_Meses',
    'Sexo',
    'Area_Residencia',
    'Peso_kg',
    'Talla_cm'
]

def test_dataset_exists():
    """Verifica que el archivo procesado oficial exista en disco."""
    assert os.path.exists(DATA_PATH), f"El dataset no existe en: {DATA_PATH}"

def test_dataset_shape():
    """Verifica que la muestra analítica contenga exactamente los 10,340 infantes filtrados."""
    df = pd.read_csv(DATA_PATH)
    assert df.shape[0] == 10340, f"Se esperaban 10,340 registros, pero se encontraron {df.shape[0]}"

def test_columns_integrity():
    """Verifica la presencia de las 7 variables predictoras y la variable objetivo."""
    df = pd.read_csv(DATA_PATH)
    for col in EXPECTED_FEATURES:
        assert col in df.columns, f"Falta la columna predictora: {col}"
    assert 'Anemia' in df.columns, "Falta la columna objetivo 'Anemia'"

def test_no_null_values():
    """Verifica que no existan valores nulos en el conjunto preprocesado."""
    df = pd.read_csv(DATA_PATH)
    cols_to_check = EXPECTED_FEATURES + ['Anemia']
    null_counts = df[cols_to_check].isnull().sum().sum()
    assert null_counts == 0, f"Se encontraron {null_counts} valores nulos en los datos"

def test_altitude_adjustment_formula():
    """Verifica la fórmula oficial de ajuste de hemoglobina por altitud (NTS N° 134-MINSA)."""
    from src.preprocessing.cleaner import adjust_hemoglobin_altitude
    
    # Caso costa (< 1,000 msnm): sin ajuste
    adj_costa, hb_adj_costa, status_costa = adjust_hemoglobin_altitude(11.5, 150)
    assert adj_costa == 0.0
    assert hb_adj_costa == 11.5
    assert status_costa == 0
    
    # Caso sierra altoandina (Puno, 3,825 msnm): ajuste significativo
    adj_sierra, hb_adj_sierra, status_sierra = adjust_hemoglobin_altitude(11.8, 3825)
    assert adj_sierra > 0.15 # Factor MINSA positivo
    assert hb_adj_sierra < 11.8
