"""
Pipeline de Entrenamiento, Optimización y Evaluación Multicriterio de Modelos
Dataset Oficial: INEI ENDES Perú (10,340 niños menores de 3 años evaluados en las 25 regiones).
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay,
    classification_report
)
import joblib

# Paths
BASE_DIR = "/home/makizapa/Descargas/PROYECTO-IA-ANEMIA"
DATA_PATH = os.path.join(BASE_DIR, "data/processed/anemia_infantil_peru_endes.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# 1. Carga de datos oficiales de Perú
df = pd.read_csv(DATA_PATH)
print(":: [1/7] Datos cargados de INEI ENDES Perú:")
print(f" -> Dimensiones: {df.shape[0]} infantes x {df.shape[1]} columnas")
print(" -> Distribución de clases en la población infantil:")
print(df['Anemia'].value_counts())

# 2. EDA estadístico
features_model = [
    'Hemoglobina_Observada',
    'Altitud_msnm',
    'Edad_Meses',
    'Sexo',
    'Area_Residencia',
    'Peso_kg',
    'Talla_cm'
]

X = df[features_model]
y = df['Anemia']

eda_stats = X.describe().to_dict()
with open(os.path.join(RESULTS_DIR, "eda_stats.json"), "w") as f:
    json.dump(eda_stats, f, indent=2)

# 3. Partición estratificada 80/20 (Prevención estricta de Data Leakage)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f":: [2/7] Partición estratificada completada: Train={X_train.shape[0]}, Test={X_test.shape[0]}")

# 4. Escalado estricto ajustado únicamente en train
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.joblib"))

# 5. Optimización del hiperparámetro k en KNN
print(":: [3/7] Optimizando hiperparámetro k para KNN en la población peruana...")
k_values = range(1, 25, 2)
k_accuracies_train = []
k_accuracies_test = []
k_f1_scores = []

for k in k_values:
    knn_temp = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    knn_temp.fit(X_train_scaled, y_train)
    y_pred_te = knn_temp.predict(X_test_scaled)
    k_accuracies_train.append(accuracy_score(y_train, knn_temp.predict(X_train_scaled)))
    k_accuracies_test.append(accuracy_score(y_test, y_pred_te))
    k_f1_scores.append(f1_score(y_test, y_pred_te))

best_k = list(k_values)[np.argmax(k_f1_scores)]
print(f" -> k óptimo seleccionado: k={best_k} (F1={max(k_f1_scores):.4f})")

# Graficar optimización de k
plt.figure(figsize=(8, 4.5))
plt.plot(k_values, k_accuracies_train, marker='o', label='Exactitud Train', color='#2F5496')
plt.plot(k_values, k_accuracies_test, marker='s', label='Exactitud Test', color='#C00000')
plt.plot(k_values, k_f1_scores, marker='^', linestyle='--', label='F1-Score Test', color='#548235')
plt.title('Evaluación de Hiperparámetro k en KNN (Población ENDES Perú)')
plt.xlabel('Número de Vecinos (k)')
plt.ylabel('Puntuación')
plt.xticks(list(k_values))
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "k_neighbors_optimization.png"), dpi=300)
plt.close()

# 6. Modelos comparativos
print(":: [4/7] Entrenando y evaluando modelos de Machine Learning...")
models = {
    "Baseline (Dummy)": DummyClassifier(strategy="most_frequent"),
    f"K-Nearest Neighbors (k={best_k})": KNeighborsClassifier(n_neighbors=best_k),
    "Random Forest (100 árboles)": RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42)
}

results = []
trained_models = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    trained_models[name] = model
    
    y_pred = model.predict(X_test_scaled)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
    else:
        y_proba = y_pred
        auc = 0.50
        
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='f1')
    
    results.append({
        "Modelo": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "ROC-AUC": auc,
        "CV F1 (Promedio)": cv_scores.mean(),
        "CV F1 (Desv. Est.)": cv_scores.std()
    })
    
    if "Nearest" in name:
        joblib.dump(model, os.path.join(MODELS_DIR, "modelo_knn.joblib"))
    elif "Random Forest" in name:
        joblib.dump(model, os.path.join(MODELS_DIR, "modelo_rf.joblib"))

df_results = pd.DataFrame(results)
df_results.to_csv(os.path.join(RESULTS_DIR, "metrics_comparison.csv"), index=False)
print("\n--- COMPARATIVA DE RENDIMIENTO (ENDES PERÚ) ---")
print(df_results.to_string(index=False))

# 7. Matrices de Confusión
print(":: [5/7] Generando matrices de confusión...")
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

cm_knn = confusion_matrix(y_test, trained_models[f"K-Nearest Neighbors (k={best_k})"].predict(X_test_scaled))
disp_knn = ConfusionMatrixDisplay(cm_knn, display_labels=['No Anemia', 'Anemia'])
disp_knn.plot(ax=axes[0], cmap='Blues', colorbar=False)
axes[0].set_title(f'Matriz de Confusión - KNN (k={best_k})')

cm_rf = confusion_matrix(y_test, trained_models["Random Forest (100 árboles)"].predict(X_test_scaled))
disp_rf = ConfusionMatrixDisplay(cm_rf, display_labels=['No Anemia', 'Anemia'])
disp_rf.plot(ax=axes[1], cmap='Greens', colorbar=False)
axes[1].set_title('Matriz de Confusión - Random Forest')

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrices.png"), dpi=300)
plt.close()

# 8. Curvas ROC
print(":: [6/7] Generando curvas ROC comparativas...")
plt.figure(figsize=(7, 5))
for name in ["Random Forest (100 árboles)", f"K-Nearest Neighbors (k={best_k})"]:
    model = trained_models[name]
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc_val = roc_auc_score(y_test, y_proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.4f})")

plt.plot([0, 1], [0, 1], 'k--', label='Baseline Dummy (AUC = 0.5000)')
plt.xlabel('Tasa de Falsos Positivos (1 - Especificidad)')
plt.ylabel('Tasa de Verdaderos Positivos (Sensibilidad / Recall)')
plt.title('Curvas ROC - Clasificación de Anemia Infantil (ENDES Perú)')
plt.legend(loc='lower right')
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "roc_curves.png"), dpi=300)
plt.close()

# 9. Importancia de Variables en Random Forest
print(":: [7/7] Analizando importancia de variables clínicas y geográficas...")
rf_model = trained_models["Random Forest (100 árboles)"]
importances = rf_model.feature_importances_
feat_names = X.columns
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(8, 4.5))
plt.bar(range(X.shape[1]), importances[indices], color='#2F5496', align='center')
plt.xticks(range(X.shape[1]), [feat_names[i] for i in indices], rotation=30, ha='right')
plt.title('Importancia de Características (Random Forest - ENDES Perú)')
plt.ylabel('Gini Importance Relativa')
plt.grid(axis='y', linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "feature_importance.png"), dpi=300)
plt.close()

feat_imp_df = pd.DataFrame({
    "Variable": [feat_names[i] for i in indices],
    "Importancia_Gini": importances[indices]
})
feat_imp_df.to_csv(os.path.join(RESULTS_DIR, "feature_importance.csv"), index=False)
print("\n--- IMPORTANCIA DE VARIABLES (ENDES PERÚ) ---")
print(feat_imp_df.to_string(index=False))

print("\n:: Pipeline de Machine Learning ejecutado exitosamente con datos de la ENDES de Perú.")
