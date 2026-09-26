"""
Módulo de Inferencia y Triaje Rápido de Anemia Infantil
Permite diagnosticar a un niño en tiempo real utilizando los modelos entrenados.
"""

import sys
import os
import argparse
import joblib
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.joblib")
MODEL_PATH = os.path.join(MODELS_DIR, "modelo_rf.joblib")


def predecir_anemia(hb_observada: float, altitud_m: float, edad_meses: int,
                    sexo: int, area_rural: int, peso_kg: float, talla_cm: float):
    """
    Evalúa un caso pediátrico aplicando la corrección MINSA y la inferencia de Random Forest.
    """
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        raise FileNotFoundError("Los modelos serializados no existen. Ejecuta primero 'python src/pipeline.py'.")

    scaler = joblib.load(SCALER_PATH)
    modelo = joblib.load(MODEL_PATH)

    # 1. Corrección oficial por altitud (NTS N° 134-MINSA)
    if altitud_m < 1000.0:
        factor = 0.0
    else:
        alt_km = altitud_m / 1000.0
        factor = -0.032 * alt_km + 0.022 * (alt_km ** 2)

    hb_ajustada = round(hb_observada - factor, 2)
    diagnostico_minsa = "ANEMIA" if hb_ajustada < 11.0 else "SANO (Sin anemia)"

    # 2. Inferencia con Modelo de Machine Learning (Random Forest)
    cols = ['Hemoglobina_Observada', 'Altitud_msnm', 'Edad_Meses', 'Sexo', 'Area_Residencia', 'Peso_kg', 'Talla_cm']
    import pandas as pd
    paciente = pd.DataFrame([[hb_observada, altitud_m, edad_meses, sexo, area_rural, peso_kg, talla_cm]], columns=cols)
    paciente_scaled = scaler.transform(paciente)

    pred_ia = modelo.predict(paciente_scaled)[0]
    prob_ia = modelo.predict_proba(paciente_scaled)[0][1]

    # Presentación limpia de resultados
    print("\n" + "=" * 55)
    print("      REPORTE DE TRIAJE PEDIÁTRICO DE ANEMIA")
    print("        Facultad de Ingeniería (UNFV / MINSA)")
    print("=" * 55)
    print(f" -> Hemoglobina observada:  {hb_observada:.1f} g/dL")
    print(f" -> Altitud geográfica:    {altitud_m:.0f} msnm")
    print(f" -> Edad del menor:        {edad_meses} meses")
    print(f" -> Sexo:                  {'Varón' if sexo == 1 else 'Mujer'}")
    print(f" -> Entorno:               {'Rural' if area_rural == 1 else 'Urbano'}")
    print(f" -> Antropometría:         Peso: {peso_kg:.1f} kg | Talla: {talla_cm:.1f} cm")
    print("-" * 55)
    print(f" * Factor de ajuste MINSA: -{factor:.2f} g/dL")
    print(f" * Hemoglobina corregida:   {hb_ajustada:.2f} g/dL (Umbral clínico: 11.0 g/dL)")
    print(f" * Diagnóstico NTS MINSA:  {diagnostico_minsa}")
    print("-" * 55)
    print(f" [!] PREDICCIÓN IA:         {'ANEMIA DETECTADA' if pred_ia == 1 else 'INFANTE SANO'}")
    print(f" [!] Probabilidad Anemia:   {prob_ia * 100:.2f}%")
    print("=" * 55 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Triaje rápido de anemia infantil en tiempo real")
    parser.add_argument("--hb", type=float, default=11.8, help="Hemoglobina observada en sangre (g/dL)")
    parser.add_argument("--altitud", type=float, default=3825.0, help="Altitud en msnm de la comunidad")
    parser.add_argument("--edad", type=int, default=24, help="Edad en meses cumplidos (6 - 35)")
    parser.add_argument("--sexo", type=int, default=1, help="1: Varón, 0: Mujer")
    parser.add_argument("--rural", type=int, default=1, help="1: Rural, 0: Urbano")
    parser.add_argument("--peso", type=float, default=11.2, help="Peso corporal en kilogramos")
    parser.add_argument("--talla", type=float, default=81.0, help="Talla / estatura en centímetros")

    args = parser.parse_args()

    predecir_anemia(
        hb_observada=args.hb,
        altitud_m=args.altitud,
        edad_meses=args.edad,
        sexo=args.sexo,
        area_rural=args.rural,
        peso_kg=args.peso,
        talla_cm=args.talla
    )


if __name__ == "__main__":
    main()
