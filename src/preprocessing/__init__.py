"""
Módulo de preprocesamiento, limpieza clínica y cálculo del ajuste por altitud geográfica.
"""
from .cleaner import clean_raw_data, adjust_hemoglobin_altitude

__all__ = ["clean_raw_data", "adjust_hemoglobin_altitude"]
