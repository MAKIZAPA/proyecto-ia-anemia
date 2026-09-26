"""
Script de Limpieza y Procesamiento de Microdatos Oficiales de la ENDES (INEI Perú).
Genera el dataset limpio 'anemia_infantil_peru_endes.csv' a partir de los módulos RECH6 y RECH0.
"""

import os
import pandas as pd
import numpy as np

# Rutas relativas al proyecto
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

rech6_path = os.path.join(RAW_DIR, "RECH6_2023.csv")
rech0_path = os.path.join(RAW_DIR, "RECH0_2023.csv")
output_csv = os.path.join(PROCESSED_DIR, "anemia_infantil_peru_endes.csv")

print(":: [1/5] Cargando microdatos crudos del INEI...")
r6 = pd.read_csv(rech6_path, low_memory=False)
r0 = pd.read_csv(rech0_path, low_memory=False)

# Limpiar nombres de columnas
r6.columns = [c.strip().replace('ï»¿', '') for c in r6.columns]
r0.columns = [c.strip().replace('ï»¿', '') for c in r0.columns]

print(f" -> Registros en RECH6 (Módulo Anemia Niños): {r6.shape[0]}")
print(f" -> Registros en RECH0 (Módulo Hogar): {r0.shape[0]}")

# Normalizar clave de union HHID
r6['HHID'] = r6['HHID'].astype(str).str.strip()
r0['HHID'] = r0['HHID'].astype(str).str.strip()

print(":: [2/5] Fusionando información del niño con datos de altitud y geografía del hogar...")
merged = pd.merge(r6, r0[['HHID', 'HV040', 'HV024', 'HV025']], on='HHID', how='inner')
print(f" -> Total tras combinación relacional: {merged.shape[0]} casos")

print(":: [3/5] Filtrando datos válidos de los niños...")
# Nos quedamos con menores de 3 años (6 a 35 meses) y con prueba de hemoglobina bien tomada (HC55 == 0)
df_filtro = merged[(merged['HC55'] == 0) & (merged['HC1'] >= 6) & (merged['HC1'] <= 35)].copy()

# En la base del INEI la hemoglobina viene multiplicada por 10, así que la dividimos entre 10
df_filtro['Hemoglobina_Observada'] = df_filtro['HC53'] / 10.0
# Descartamos errores de lectura o valores fuera de rango normal (dejamos entre 4.0 y 20.0 g/dL)
df_filtro = df_filtro[(df_filtro['Hemoglobina_Observada'] >= 4.0) & (df_filtro['Hemoglobina_Observada'] <= 20.0)].copy()

# Convertimos peso a kilos (HC2 viene en hectogramos) y talla a centímetros (HC3 viene en mm)
df_filtro['Peso_kg'] = pd.to_numeric(df_filtro['HC2'], errors='coerce') / 10.0
df_filtro['Talla_cm'] = pd.to_numeric(df_filtro['HC3'], errors='coerce') / 10.0

# Descartamos valores vacíos o errores al anotar en la balanza (peso de 3 a 30 kg y talla de 45 a 120 cm)
df_filtro = df_filtro.dropna(subset=['Peso_kg', 'Talla_cm'])
df_filtro = df_filtro[(df_filtro['Peso_kg'] >= 3.0) & (df_filtro['Peso_kg'] <= 30.0)]
df_filtro = df_filtro[(df_filtro['Talla_cm'] >= 45.0) & (df_filtro['Talla_cm'] <= 120.0)].copy()

print(f" -> Casos válidos con medidas coherentes: {df_filtro.shape[0]}")

print(":: [4/5] Aplicando fórmula oficial MINSA / CDC de corrección por altitud...")
df_filtro['Altitud_msnm'] = df_filtro['HV040'].astype(float)
alt_km = df_filtro['Altitud_msnm'] / 1000.0

# Ecuación de Dallman / CDC adoptada en la NTS N° 134-MINSA
factor_ajuste = np.where(
    df_filtro['Altitud_msnm'] < 1000.0,
    0.0,
    -0.032 * alt_km + 0.022 * (alt_km ** 2)
)
df_filtro['Hemoglobina_Ajustada'] = np.round(df_filtro['Hemoglobina_Observada'] - factor_ajuste, 2)

# Definición del target diagnóstico según norma técnica peruana
df_filtro['Anemia'] = np.where(df_filtro['Hemoglobina_Ajustada'] < 11.0, 1, 0)

# Mapeo y recodificación de covariables
df_filtro['Edad_Meses'] = df_filtro['HC1'].astype(int)
df_filtro['Sexo'] = np.where(df_filtro['HC27'] == 1, 1, 0) # 1: Varon, 0: Mujer
df_filtro['Area_Residencia'] = np.where(df_filtro['HV025'] == 2, 1, 0) # 1: Rural, 0: Urbano

dept_map = {
    1: 'Amazonas', 2: 'Ancash', 3: 'Apurimac', 4: 'Arequipa', 5: 'Ayacucho',
    6: 'Cajamarca', 7: 'Callao', 8: 'Cusco', 9: 'Huancavelica', 10: 'Huanuco',
    11: 'Ica', 12: 'Junin', 13: 'La Libertad', 14: 'Lambayeque', 15: 'Lima',
    16: 'Loreto', 17: 'Madre de Dios', 18: 'Moquegua', 19: 'Pasco', 20: 'Piura',
    21: 'Puno', 22: 'San Martin', 23: 'Tacna', 24: 'Tumbes', 25: 'Ucayali'
}
df_filtro['Departamento'] = df_filtro['HV024'].map(dept_map)

# Seleccionar columnas finales para modelado e investigación
cols_finales = [
    'Hemoglobina_Observada',
    'Altitud_msnm',
    'Edad_Meses',
    'Sexo',
    'Area_Residencia',
    'Peso_kg',
    'Talla_cm',
    'Departamento',
    'Hemoglobina_Ajustada',
    'Anemia'
]
df_final = df_filtro[cols_finales].reset_index(drop=True)

print(":: [5/5] Exportando dataset procesado...")
df_final.to_csv(output_csv, index=False)
print(f" -> Guardado exitosamente en: {output_csv}")
print(f" -> Dimensiones finales: {df_final.shape[0]} filas x {df_final.shape[1]} columnas")
print("\nResumen epidemiológico de la muestra procesada:")
print(f" - Casos Sin Anemia (0): {(df_final['Anemia'] == 0).sum()} ({(df_final['Anemia'] == 0).mean()*100:.2f}%)")
print(f" - Casos Con Anemia (1): {(df_final['Anemia'] == 1).sum()} ({(df_final['Anemia'] == 1).mean()*100:.2f}%)")
print(f" - Altitud promedio: {df_final['Altitud_msnm'].mean():.1f} msnm (Máxima: {df_final['Altitud_msnm'].max()} msnm)")
