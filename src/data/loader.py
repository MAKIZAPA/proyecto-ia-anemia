"""
Cargador de microdatos del INEI ENDES Perú (Módulos RECH6 y RECH0).
"""

import os
import pandas as pd

DEFAULT_PROCESSED_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data/processed/anemia_infantil_peru_endes.csv"
)

DEFAULT_RAW_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data/raw"
)

def load_processed_data(file_path: str = DEFAULT_PROCESSED_PATH) -> pd.DataFrame:
    """Carga el dataset analítico procesado de 10,340 registros infantiles."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Archivo procesado no encontrado en: {file_path}")
    df = pd.read_csv(file_path)
    return df

def load_raw_endes(raw_dir: str = DEFAULT_RAW_DIR) -> tuple:
    """Carga los módulos crudos RECH6 (Salud infantil) y RECH0 (Vivienda y altitud)."""
    p_rech6 = os.path.join(raw_dir, "RECH6_2023.csv")
    p_rech0 = os.path.join(raw_dir, "RECH0_2023.csv")
    
    if not os.path.exists(p_rech6) or not os.path.exists(p_rech0):
        raise FileNotFoundError("Módulos crudos RECH6 y RECH0 no encontrados en data/raw/")
        
    df_rech6 = pd.read_csv(p_rech6, low_memory=False)
    df_rech0 = pd.read_csv(p_rech0, low_memory=False)
    return df_rech6, df_rech0
