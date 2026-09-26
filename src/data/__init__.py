"""
Módulo de carga y gestión de datos epidemiológicos oficiales de la ENDES (INEI Perú).
"""
from .loader import load_raw_endes, load_processed_data

__all__ = ["load_raw_endes", "load_processed_data"]
