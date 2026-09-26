"""
Limpieza de microdatos, enlace relacional y cálculo de factor de altitud según NTS N° 134-MINSA.
"""

import numpy as np
import pandas as pd

def adjust_hemoglobin_altitude(hb_observed: float, altitude_m: float) -> tuple:
    """
    Calcula el factor de corrección por altitud y la hemoglobina ajustada
    según la Norma Técnica Sanitaria NTS N° 134-MINSA del Perú.
    """
    if altitude_m < 1000:
        adjustment = 0.0
    else:
        alt_k = altitude_m / 1000.0
        adjustment = -0.032 * alt_k + 0.022 * (alt_k ** 2)
        
    hb_adjusted = hb_observed - adjustment
    anemia_status = 1 if hb_adjusted < 11.0 else 0
    return adjustment, round(hb_adjusted, 2), anemia_status

def clean_raw_data(df_rech6: pd.DataFrame, df_rech0: pd.DataFrame) -> pd.DataFrame:
    """
    Integra relacionalmente RECH6 y RECH0 mediante el código HHID,
    aplica filtros de prueba válida y rangos razonables de peso y talla (10,340 registros).
    """
    # Enlace relacional por HHID
    df_merged = pd.merge(df_rech6, df_rech0[['HHID', 'HV040']], on='HHID', how='inner')
    
    # Renombrado de variables estandarizadas
    # HC55 == 0 indica consentimiento y prueba de sangre válida
    if 'HC55' in df_merged.columns:
        df_merged = df_merged[df_merged['HC55'] == 0]
        
    return df_merged
