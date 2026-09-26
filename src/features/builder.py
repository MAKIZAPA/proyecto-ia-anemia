"""
Construcción y aislamiento de variables predictoras para evitar fuga de información (data leakage).
"""

from typing import Tuple, List
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS: List[str] = [
    'Hemoglobina_Observada',
    'Altitud_msnm',
    'Edad_Meses',
    'Sexo',
    'Area_Residencia',
    'Peso_kg',
    'Talla_cm'
]

TARGET_COLUMN: str = 'Anemia'

def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Extrae la matriz de características X (7 variables) y el vector objetivo y."""
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()
    return X, y

def split_and_scale(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.20,
    random_state: int = 42
) -> Tuple:
    """
    Particiona estratificadamente (80/20) y ajusta el StandardScaler ÚNICAMENTE
    con los datos de entrenamiento (fit en train, transform en train y test).
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler
