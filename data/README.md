# Diccionario de Datos y Ficha Técnica — INEI ENDES Perú 2023

## 1. Origen y Módulos Oficiales
Los datos utilizados en esta investigación provienen de la **Encuesta Demográfica y de Salud Familiar (ENDES)** ejecutada a nivel nacional por el **Instituto Nacional de Estadística e Informática (INEI)** del Perú.

Se integraron dos módulos censales mediante el identificador relacional del hogar (`HHID`):
- **`RECH6_2023.csv`**: Módulo de salud infantil y antropometría del menor de cinco años.
- **`RECH0_2023.csv`**: Módulo de características del hogar, vivienda y altitud geográfica en metros sobre el nivel del mar.

---

## 2. Criterios de Selección y Muestra Analítica
- **Población objetivo**: Infantes peruanos entre **6 y 35 meses de edad** (primeros 1,000 días de vida).
- **Criterio de inclusión**: Menores con prueba de hemoglobina completa y consentimiento informado (`HC55 == 0`).
- **Control de calidad y rangos válidos**:
  - Hemoglobina observada: $4.0 \le \text{Hb} \le 20.0$ g/dL.
  - Peso corporal: $3.0 \le \text{Peso} \le 30.0$ kg.
  - Talla / longitud: $45.0 \le \text{Talla} \le 120.0$ cm.
  - Altitud geográfica: $0 \le \text{Altitud} \le 5,000$ msnm.
- **Tamaño muestral final**: **10,340 niños y niñas** de los 25 departamentos del Perú.

---

## 3. Matriz de Variables

| Variable | Tipo de Dato | Unidad / Valores | Descripción Clínica / Geográfica |
| :--- | :--- | :--- | :--- |
| `Hemoglobina_Observada` | Float | g/dL (4.2 - 18.9) | Concentración hemática medida en sangre mediante hemoglobinómetro portátil (HemoCue). |
| `Altitud_msnm` | Integer | Metros (1 - 4,488) | Altitud sobre el nivel del mar de la comunidad o centro poblado del menor. |
| `Edad_Meses` | Integer | Meses (6 - 35) | Edad cumplida del infante al momento de la entrevista. |
| `Sexo` | Integer | 0: Femenino, 1: Masculino | Sexo biológico del niño o niña. |
| `Area_Residencia` | Integer | 0: Urbano, 1: Rural | Clasificación geográfica del entorno habitacional. |
| `Peso_kg` | Float | Kilogramos (4.5 - 24.8) | Masa corporal obtenida en balanza pediátrica calibrada. |
| `Talla_cm` | Float | Centímetros (50.0 - 110.0) | Longitud corporal medida en infantómetro de madera. |
| **`Anemia`** *(Target)* | Integer | **0: Sano, 1: Anémico** | Diagnóstico oficial tras aplicar el factor polinómico de ajuste por altitud (NTS N° 134-MINSA). |

---

## 4. Distribución de Clases
- **Clase 0 (No anémico / Sano)**: 7,270 infantes (70.31%)
- **Clase 1 (Anémico)**: 3,070 infantes (29.69%)
- **Tasa de prevalencia nacional en la muestra**: 29.69% (se incrementa significativamente por encima de los 3,000 msnm).

---

## 5. Licencia, Citas y Agradecimientos
Los microdatos abiertos del INEI Perú son de acceso público irrestricto con fines de investigación científica y docencia universitaria, respetando el anonimato y la confidencialidad de las familias peruanas encuestadas.

- **Fuente Oficial Primaria:** Instituto Nacional de Estadística e Informática (INEI Perú) — Encuesta Demográfica y de Salud Familiar (ENDES 2023).
- **Agradecimiento por Compilación en CSV:** Se agradece al investigador **César A. H. N.** ([`CesarAHN/anemia-endes-inei`](https://github.com/CesarAHN/anemia-endes-inei)) por poner a disposición de la comunidad los módulos brutos extraídos en formato CSV estructurado.
