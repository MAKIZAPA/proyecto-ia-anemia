<div align="center">

  <h1>PROYECTO IA — ANEMIA INFANTIL EN EL PERÚ</h1>

  <p><b>Sistema Inteligente Basado en Modelos de Aprendizaje Automático para la Clasificación y Detección Oportuna de Anemia Infantil a partir de Microdatos de la ENDES</b></p>
  <p><i>Calibrado con 10,340 registros de infantes peruanos evaluados por el INEI, factor de ajuste por altitud andina (NTS N° 134-MINSA), optimización de Random Forest y KNN, alcanzando un recall clínico del 99.67%.</i></p>

  <p>
    <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-10b981?style=for-the-badge" alt="License" /></a>
    <img src="https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
    <img src="https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-Learn" />
    <img src="https://img.shields.io/badge/UNFV-FIEI-003366?style=for-the-badge" alt="UNFV FIEI" />
    <a href="notebooks/proyecto_clasificacion_anemia_colab.ipynb"><img src="https://img.shields.io/badge/Colab-Open%20Notebook-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white" alt="Open In Colab" /></a>
  </p>

  <br/>

  <sub>
    <a href="#problema-y-objetivo">Problema & Objetivo</a> •
    <a href="#características">Características</a> •
    <a href="#estructura-del-repositorio">Estructura</a> •
    <a href="#resultados-experimentales">Resultados</a> •
    <a href="#instalación">Instalación</a> •
    <a href="#ejecución-y-reproducibilidad">Reproducibilidad</a> •
    <a href="#autor--agradecimientos">Autor</a> •
    <a href="#licencia">Licencia</a>
  </sub>

  <br/><br/>
</div>

---

## Problema y Objetivo

- **Problema:** La anemia infantil por déficit de hierro afecta al 43.6% de los infantes peruanos de 6 a 35 meses a nivel nacional, superando el 70% en zonas altoandinas como Puno y Huancavelica. En las postas rurales del primer nivel de atención (I-1 a I-3), la ausencia de hematólogos y los errores al restar manualmente el ajuste por altitud con tablas impresas retrasan el inicio oportuno del tratamiento con sulfato ferroso.
- **Objetivo:** Desarrollar y validar experimentalmente un sistema inteligente de aprendizaje automático supervisado (Random Forest y KNN) calibrado con microdatos oficiales de la ENDES 2023, que clasifique automáticamente la anemia incorporando la corrección por altitud de la Norma Técnica de Salud NTS N° 134-MINSA para agilizar el triaje médico primario.

---

## Características

<table>
  <tr>
    <td width="50%">
      <img src="https://img.shields.io/badge/DATASET-ENDES%202023%20PERÚ-1F4E79?style=flat-square" alt="ENDES Perú" /><br/>
      <b>Microdatos Oficiales del INEI</b><br/>
      Muestra probabilística y estratificada de 10,340 niños peruanos con prueba válida de hemoglobina (HC55=0) y altitud registrada de los 25 departamentos del país.
    </td>
    <td width="50%">
      <img src="https://img.shields.io/badge/FISIOLOGÍA-NTS%20N°%20134--MINSA-C00000?style=flat-square" alt="MINSA" /><br/>
      <b>Ajuste Polinómico por Altitud</b><br/>
      Compensación matemática rigurosa por hipoxia ambiental para comunidades altoandinas hasta los 4,488 msnm, eliminando errores de diagnóstico manual en postas.
    </td>
  </tr>
  <tr>
    <td width="50%">
      <img src="https://img.shields.io/badge/ARQUITECTURA-ANTI--LEAKAGE-2E7D32?style=flat-square" alt="Anti-Leakage" /><br/>
      <b>Aislamiento Estricto de Datos</b><br/>
      Partición estratificada 80/20 con <code>StandardScaler</code> calibrado únicamente en el conjunto de entrenamiento, previniendo sobreajuste y fuga de información.
    </td>
    <td width="50%">
      <img src="https://img.shields.io/badge/EFICACIA%20CLÍNICA-99.67%25%20RECALL-7030A0?style=flat-square" alt="Recall Clínico" /><br/>
      <b>Máxima Sensibilidad Médica</b><br/>
      Random Forest detecta 612 de 614 infantes anémicos en el conjunto de prueba independiente (solo 2 falsos negativos), superando ampliamente al baseline trivial.
    </td>
  </tr>
  <tr>
    <td width="50%">
      <img src="https://img.shields.io/badge/OPTIMIZACIÓN-KNN%20(k=17)-1565C0?style=flat-square" alt="KNN" /><br/>
      <b>Frontera Métrica Óptima</b><br/>
      Barrido sistemático de vecinos más cercanos ($k \in [1, 31]$) que determina $k=17$ como la configuración métrica de menor error ($F_1 = 94.04\%$, $AUC = 0.9958$).
    </td>
    <td width="50%">
      <img src="https://img.shields.io/badge/DESPLIEGUE-MODELOS%20.JOBLIB-008080?style=flat-square" alt="Serialización" /><br/>
      <b>Prototipo Asistencial Ligero</b><br/>
      Modelos exportados en archivos binarios compactos listos para integrarse en software asistencial o ejecutarse directamente en Google Colab.
    </td>
  </tr>
</table>

---

## Estructura del Repositorio

El proyecto cumple estrictamente con el estándar de ingeniería de software propuesto en el sílabo académico oficial (Sección 29.1):

```text
PROYECTO-IA/
│
├── README.md                           # Documentación técnica principal y guía de reproducción
├── LICENSE                             # Licencia MIT de código abierto
├── requirements.txt                    # Dependencias científicas fijadas
│
├── data/
│   ├── raw/                            # Microdatos originales del INEI (RECH0_2023.csv, RECH6_2023.csv)
│   ├── processed/                      # Dataset analítico limpio (10,340 registros x 8 columnas)
│   └── README.md                       # Diccionario de datos y especificación técnica
│
├── notebooks/
│   ├── 01_data_understanding.ipynb     # Comprensión del dominio y variables pediátricas
│   ├── 02_eda.ipynb                    # Análisis exploratorio, distribuciones y altitud
│   ├── 03_preprocessing.ipynb          # Enlace relacional y escalado anti-leakage
│   ├── 04_modeling.ipynb               # Entrenamiento de Baseline, KNN y Random Forest
│   ├── 05_evaluation.ipynb             # Evaluación diagnóstica y validación clínica
│   └── proyecto_clasificacion_anemia_colab.ipynb # Cuaderno unificado listo para Google Colab
│
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py                   # Carga de datos crudos y procesados
│   │   └── clean_endes_peru.py         # Pipeline de limpieza y filtrado poblacional
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   └── cleaner.py                  # Función de ajuste por altitud NTS N° 134-MINSA
│   ├── features/
│   │   ├── __init__.py
│   │   └── builder.py                  # Selección de 7 variables y partición estratificada
│   ├── models/
│   │   ├── __init__.py
│   │   └── train.py                    # Algoritmos de entrenamiento y optimización de k
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py                  # Métricas de salud, curvas ROC y matrices de confusión
│   └── pipeline.py                     # Orquestador integral ejecutable por consola
│
├── tests/
│   ├── __init__.py
│   ├── test_data.py                    # Pruebas de integridad del dataset y fórmula MINSA
│   └── test_models.py                  # Pruebas unitarias de inferencia con modelos .joblib
│
├── models/
│   ├── scaler.joblib                   # StandardScaler calibrado en entrenamiento
│   ├── modelo_knn.joblib               # Modelo KNN óptimo (k=17)
│   └── modelo_rf.joblib                # Modelo Random Forest (100 árboles)
│
├── results/
│   ├── k_neighbors_optimization.png    # Gráfico de optimización de k en KNN
│   ├── confusion_matrices.png          # Matrices de confusión clínicas normalizadas
│   ├── roc_curves.png                  # Curvas ROC comparativas (Dummy, KNN, Random Forest)
│   ├── feature_importance.png          # Aporte de variables según impureza de Gini
│   ├── metrics_comparison.csv          # Tabla cuantitativa de métricas
│   └── feature_importance.csv          # Detalle porcentual por variable
│
└── docs/
    ├── PROYECTO_INVESTIGACION_IA.pdf   # Informe formal FIEI (29 secciones académicas)
    └── PROYECTO_INVESTIGACION_IA.docx  # Documento editable con formato APA 7 / MIC UNFV
```

---

## Resultados Experimentales

Evaluación comparativa sobre la partición de prueba independiente de **2,068 niños peruanos** (614 anémicos y 1,454 sanos):

| Experimento | Modelo | Accuracy (%) | Precisión (%) | Recall / Sensibilidad (%) | F1-Score (%) | ROC-AUC | 5-Fold CV F1 (%) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **E1** | **Baseline (Dummy)** | 70.31% | 0.00% | 0.00% | 0.00% | 0.5000 | 0.00% (± 0.00%) |
| **E2** | **K-Nearest Neighbors ($k=17$)** | 96.62% | 98.57% | 89.90% | 94.04% | 0.9958 | 93.49% (± 1.48%) |
| **E3** | **Random Forest (100 árboles)** | **99.90%** | **100.00%** | **99.67%** | **99.84%** | **0.9999** | **99.82% (± 0.08%)** |

### Importancia de Características (Gini)
1. **Hemoglobina observada:** $87.74\%$ (parámetro fisiológico de transporte de oxígeno)
2. **Altitud geográfica:** $7.12\%$ (compensación de hipoxia en los Andes peruanos)
3. **Talla:** $1.64\%$ (indicador de crecimiento y desnutrición crónica)
4. **Edad en meses:** $1.36\%$ (vulnerabilidad en los primeros 1,000 días de vida)
5. **Peso corporal:** $1.25\%$ (estado nutricional global)
6. **Sexo biológico:** $0.54\%$ (diferenciación antropométrica)
7. **Área de residencia:** $0.35\%$ (contexto habitacional rural/urbano)

---

## Instalación

El entorno requiere Python 3.10 o superior.

```bash
# 1. Clonar el repositorio
git clone https://github.com/makizapa/PROYECTO-IA-ANEMIA.git
cd PROYECTO-IA-ANEMIA

# 2. Crear entorno virtual
python3 -m venv .venv
source .venv/bin/activate

# 3. Instalar dependencias
pip install -r requirements.txt
```

*Opcional con `uv` (instalación ultrarrápida):*
```bash
uv pip install -r requirements.txt
```

---

## Ejecución y Reproducibilidad

### 1. Ejecutar el Pipeline Completo
Entrena los 3 modelos, ejecuta la validación cruzada y regenera los gráficos en `results/`:
```bash
python src/pipeline.py
```

### 2. Ejecutar Pruebas Unitarias
Valida la integridad de datos, fórmula MINSA e inferencia de modelos:
```bash
pytest tests/ -v
```

### 3. Ejecución en Google Colab
El proyecto incluye el cuaderno interactivo autoejecutable:
- Cuaderno unificado: `notebooks/proyecto_clasificacion_anemia_colab.ipynb`
- Al subir este repositorio a GitHub, puedes abrirlo directamente en Colab haciendo clic en la insignia superior o abriendo `https://colab.research.google.com/github/makizapa/PROYECTO-IA-ANEMIA/blob/main/notebooks/proyecto_clasificacion_anemia_colab.ipynb`.

### 4. Inferencia con Casos Nuevos
```python
import joblib
import numpy as np

# Cargar artefactos
scaler = joblib.load("models/scaler.joblib")
modelo_rf = joblib.load("models/modelo_rf.joblib")

# Caso pediátrico: Puno (3,825 msnm), Niño de 24 meses, Varón (1), Rural (1), Peso 11.2 kg, Talla 81.0 cm, Hb 11.8 g/dL
infante = np.array([[11.8, 3825, 24, 1, 1, 11.2, 81.0]])
infante_scaled = scaler.transform(infante)

prediccion = modelo_rf.predict(infante_scaled)[0]
probabilidad = modelo_rf.predict_proba(infante_scaled)[0][1]

print("Diagnóstico:", "ANEMIA" if prediccion == 1 else "SANO")
print(f"Probabilidad de Anemia: {probabilidad * 100:.2f}%")
```

---

## Autor & Agradecimientos

- **Autor:** Jimenez Villalva Franz José
- **Docente Asesor:** Dr. Ciro Rodríguez Rodríguez
- **Institución:** Universidad Nacional Federico Villarreal (UNFV)
- **Facultad:** Facultad de Ingeniería Electrónica e Informática (FIEI)
- **Asignatura:** Inteligencia Artificial
- **Año:** 2026

### Reconocimientos y Fuentes de Datos
- **Fuente Primaria:** Al **Instituto Nacional de Estadística e Informática (INEI)** del Perú por proveer en formato de datos abiertos los microdatos oficiales de la Encuesta Demográfica y de Salud Familiar (ENDES 2023).
- **Compilación de Microdatos en CSV:** Al investigador **César A. H. N.** ([`CesarAHN/anemia-endes-inei`](https://github.com/CesarAHN/anemia-endes-inei)) por disponibilizar la extracción inicial de los módulos `RECH0` y `RECH6` en formato tabular CSV para la comunidad científica.

---

<details>
<summary><b>Aviso Legal y Uso Responsable (Haz clic para expandir)</b></summary>

<br/>

Este proyecto ha sido desarrollado con fines estrictamente académicos y de investigación científica en ingeniería informática. Los modelos implementados constituyen una herramienta de soporte analítico complementario para la aceleración del triaje pediátrico primario en centros de salud rurales, y **no reemplazan el criterio médico profesional ni los protocolos diagnósticos emitidos por el Ministerio de Salud (MINSA)**.

</details>

---

## Licencia

Este proyecto está distribuido bajo la licencia **MIT**. Consulta el archivo [`LICENSE`](LICENSE) para mayores detalles.
