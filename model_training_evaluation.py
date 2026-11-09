# 1. Importar las librerías

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from pathlib import Path
from time import time

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    average_precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from xgboost import XGBClassifier

from model_preprocessing import build_preprocessing_pipeline


# 2. Definir las rutas y configuraciones

CARPETA_PROYECTO = Path(__file__).resolve().parent

CARPETA_DATASETS = CARPETA_PROYECTO / 'outputs' / 'datasets'
CARPETA_MODELOS = CARPETA_PROYECTO / 'outputs' / 'models'
CARPETA_REPORTES = CARPETA_PROYECTO / 'outputs' / 'reports'

VARIABLE_OBJETIVO = 'Pago_atiempo'

RANDOM_STATE = 42
NUM_PARTICIONES = 5


# 3. Cargar los datasets

datos_train = pd.read_csv(
    CARPETA_DATASETS / 'dataset_train_raw.csv'
)

datos_test = pd.read_csv(
    CARPETA_DATASETS / 'dataset_test_raw.csv'
)

# Separar las características y la variable objetivo

X_train = datos_train.drop(columns=[VARIABLE_OBJETIVO])
X_test = datos_test.drop(columns=[VARIABLE_OBJETIVO])

y_train_original = datos_train[VARIABLE_OBJETIVO]
y_test_original = datos_test[VARIABLE_OBJETIVO]

# Verificar que las clases esperadas estén presentes

if y_train_original.isna().any() or y_test_original.isna().any():
    raise ValueError('La variable objetivo contiene valores nulos.')

if set(y_train_original.unique()) != {0, 1}:
    raise ValueError('La variable objetivo debe contener las clases 0 y 1.')

if set(y_test_original.unique()) != {0, 1}:
    raise ValueError('La variable objetivo de evaluación debe contener las clases 0 y 1.')

# Invertir temporalmente la codificación:
# 1 = incumplimiento
# 0 = pago a tiempo

y_train = (y_train_original == 0).astype(int)
y_test = (y_test_original == 0).astype(int)

# Asegurar que entrenamiento y evaluación tengan las mismas columnas

X_test = X_test.reindex(columns=X_train.columns)

print('Dimensiones de los datos:')
print('Entrenamiento:', X_train.shape)
print('Evaluación:', X_test.shape)

print('\nDistribución de incumplimientos:')
print('Entrenamiento:', y_train.value_counts().to_dict())
print('Evaluación:', y_test.value_counts().to_dict())


# 4. Definir la función para construir los modelos

def build_model(nombre_modelo, peso_clase):

    if nombre_modelo == 'Regresión Logística':

        modelo = LogisticRegression(
            class_weight='balanced',
            max_iter=2000,
            random_state=RANDOM_STATE
        )

    elif nombre_modelo == 'Árbol de Decisión':

        modelo = DecisionTreeClassifier(
            class_weight='balanced',
            max_depth=6,
            min_samples_leaf=10,
            random_state=RANDOM_STATE
        )

    elif nombre_modelo == 'Random Forest':

        modelo = RandomForestClassifier(
            n_estimators=300,
            class_weight='balanced',
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

    elif nombre_modelo == 'XGBoost':

        modelo = XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            scale_pos_weight=peso_clase,
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method='hist',
            device='cpu',
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

    else:
        raise ValueError('Modelo no reconocido.')

    return modelo


def build_training_pipeline(nombre_modelo, peso_clase):
    preprocesamiento = build_preprocessing_pipeline()
    return Pipeline(
        steps=preprocesamiento.steps + [
            ('modelo', build_model(nombre_modelo, peso_clase))
        ]
    )


# 5. Definir la función para resumir las métricas

def summarize_classification(y_real, y_predicho, probabilidades):

    matriz = confusion_matrix(
        y_real,
        y_predicho,
        labels=[0, 1]
    )

    tn, fp, fn, tp = matriz.ravel()

    especificidad = tn / (tn + fp) if (tn + fp) > 0 else 0

    metricas = {
        'Accuracy': accuracy_score(y_real, y_predicho),
        'Precision': precision_score(
            y_real, y_predicho, zero_division=0
        ),
        'Recall': recall_score(
            y_real, y_predicho, zero_division=0
        ),
        'F1-Score': f1_score(
            y_real, y_predicho, zero_division=0
        ),
        'Especificidad': especificidad,
        'ROC-AUC': roc_auc_score(y_real, probabilidades),
        'PR-AUC': average_precision_score(y_real, probabilidades)
    }

    return metricas, matriz


# 6. Configurar la validación cruzada

# Calcular el peso de la clase minoritaria para XGBoost

cantidad_incumplimientos = (y_train == 1).sum()
cantidad_pagos = (y_train == 0).sum()

peso_clase = cantidad_pagos / cantidad_incumplimientos

# Crear las cinco particiones estratificadas

validacion = StratifiedKFold(
    n_splits=NUM_PARTICIONES,
    shuffle=True,
    random_state=RANDOM_STATE
)

# Métricas que utilizaremos para comparar los modelos

metricas_evaluacion = {
    'Recall': 'recall',
    'Precision': 'precision',
    'F1': 'f1',
    'Accuracy': 'accuracy',
    'ROC_AUC': 'roc_auc',
    'PR_AUC': 'average_precision'
}


# 7. Entrenar y comparar los cuatro modelos

nombres_modelos = [
    'Regresión Logística',
    'Árbol de Decisión',
    'Random Forest',
    'XGBoost'
]

resultados = []

print('\nIniciando entrenamiento y validación cruzada...\n')

for nombre in nombres_modelos:

    print(f'Evaluando: {nombre}')

    modelo = build_training_pipeline(nombre, peso_clase)

    resultado_cv = cross_validate(
        modelo,
        X_train,
        y_train,
        cv=validacion,
        scoring=metricas_evaluacion,
        return_train_score=False,
        n_jobs=1
    )

    resultados.append({
        'Modelo': nombre,

        'Recall_CV': resultado_cv['test_Recall'].mean(),
        'Recall_Desv': resultado_cv['test_Recall'].std(),

        'Precision_CV': resultado_cv['test_Precision'].mean(),
        'Precision_Desv': resultado_cv['test_Precision'].std(),

        'F1_CV': resultado_cv['test_F1'].mean(),
        'F1_Desv': resultado_cv['test_F1'].std(),

        'Accuracy_CV': resultado_cv['test_Accuracy'].mean(),

        'ROC_AUC_CV': resultado_cv['test_ROC_AUC'].mean(),

        'PR_AUC_CV': resultado_cv['test_PR_AUC'].mean(),

        'Tiempo_entrenamiento': resultado_cv['fit_time'].mean(),

        'Tiempo_prediccion': resultado_cv['score_time'].mean()
    })

    print('Recall promedio:', round(resultado_cv['test_Recall'].mean(), 4))
    print('F1-Score promedio:', round(resultado_cv['test_F1'].mean(), 4))
    print('Evaluación completada.\n')


# 8. Crear la tabla comparativa

tabla_resultados = pd.DataFrame(resultados)

# Ordenar según Recall, F1-Score, estabilidad y tiempo de entrenamiento

tabla_resultados = tabla_resultados.sort_values(
    by=[
        'Recall_CV',
        'F1_CV',
        'Recall_Desv',
        'Tiempo_entrenamiento'
    ],
    ascending=[False, False, True, True]
).reset_index(drop=True)

print('\nTABLA COMPARATIVA DE MODELOS')
print(tabla_resultados.to_string(index=False))


# 9. Seleccionar el mejor modelo

mejor_nombre = tabla_resultados.loc[0, 'Modelo']

print('\nMejor modelo seleccionado:', mejor_nombre)

# Construir el modelo seleccionado

mejor_modelo = build_training_pipeline(mejor_nombre, peso_clase)

# Entrenar con todos los datos de entrenamiento

inicio_entrenamiento = time()

mejor_modelo.fit(X_train, y_train)

tiempo_entrenamiento_final = time() - inicio_entrenamiento


# 10. Evaluar el mejor modelo con los datos de prueba

inicio_prediccion = time()

predicciones = mejor_modelo.predict(X_test)

probabilidades = mejor_modelo.predict_proba(X_test)[:, 1]

tiempo_prediccion_final = time() - inicio_prediccion

metricas_finales, matriz_confusion = summarize_classification(
    y_test,
    predicciones,
    probabilidades
)

print('\nEVALUACIÓN FINAL DEL MEJOR MODELO')

for metrica, valor in metricas_finales.items():
    print(f'{metrica}: {valor:.4f}')

print('\nMatriz de confusión:')
print(matriz_confusion)

print('\nTiempo de entrenamiento final:', round(tiempo_entrenamiento_final, 4), 'segundos')
print('Tiempo de predicción:', round(tiempo_prediccion_final, 4), 'segundos')


# 11. Crear las carpetas de resultados

CARPETA_MODELOS.mkdir(parents=True, exist_ok=True)
CARPETA_REPORTES.mkdir(parents=True, exist_ok=True)

# Guardar los datos transformados para el monitoreo y el panel
limpieza = mejor_modelo.named_steps['limpieza']
preprocesador = mejor_modelo.named_steps['preprocesador']
X_train_limpio = limpieza.transform(X_train)
X_test_limpio = limpieza.transform(X_test)
X_train_transformado = pd.DataFrame(
    preprocesador.transform(X_train_limpio),
    columns=preprocesador.get_feature_names_out(),
    index=X_train.index
)
X_test_transformado = pd.DataFrame(
    preprocesador.transform(X_test_limpio),
    columns=preprocesador.get_feature_names_out(),
    index=X_test.index
)
dataset_train = X_train_transformado.assign(**{VARIABLE_OBJETIVO: y_train_original})
dataset_test = X_test_transformado.assign(**{VARIABLE_OBJETIVO: y_test_original})
dataset_train.to_csv(CARPETA_DATASETS / 'dataset_train.csv', index=False)
dataset_test.to_csv(CARPETA_DATASETS / 'dataset_test.csv', index=False)
joblib.dump(
    mejor_modelo[:-1],
    CARPETA_MODELOS / 'preprocesador.joblib'
)


# 12. Guardar la tabla comparativa y las métricas finales

tabla_resultados.to_csv(
    CARPETA_REPORTES / 'comparacion_modelos.csv',
    index=False
)

pd.DataFrame([metricas_finales]).to_csv(
    CARPETA_REPORTES / 'evaluacion_final.csv',
    index=False
)

# 12.1 Justificación de la selección del modelo

mejor = tabla_resultados.loc[0]
segundo = tabla_resultados.loc[1]
mas_estable = tabla_resultados.loc[tabla_resultados['Recall_Desv'].idxmin()]
mas_rapido = tabla_resultados.loc[tabla_resultados['Tiempo_entrenamiento'].idxmin()]
mas_lento = tabla_resultados.loc[tabla_resultados['Tiempo_entrenamiento'].idxmax()]

justificacion = f"""JUSTIFICACIÓN DEL MODELO SELECCIONADO: {mejor_nombre}

1. Desempeño: tiene el mayor recall en validación cruzada ({mejor['Recall_CV']:.3f}; el segundo,
   {segundo['Modelo']}, obtiene {segundo['Recall_CV']:.3f}), con F1 {mejor['F1_CV']:.3f} y PR-AUC
   {mejor['PR_AUC_CV']:.3f}. Se prioriza el recall porque en riesgo crediticio no detectar un
   incumplimiento cuesta más que revisar una solicitud buena, y la clase de incumplimiento es minoritaria (4,75 %).
2. Consistencia: la desviación del recall entre los {NUM_PARTICIONES} pliegues es {mejor['Recall_Desv']:.3f}
   (el modelo más estable es {mas_estable['Modelo']}, con {mas_estable['Recall_Desv']:.3f}).
3. Escalabilidad: entrena en {mejor['Tiempo_entrenamiento']:.2f} s y predice en
   {mejor['Tiempo_prediccion']:.2f} s por pliegue (el más rápido en entrenar es {mas_rapido['Modelo']},
   el más lento {mas_lento['Modelo']}).
4. Evaluación final en el conjunto de prueba: recall {metricas_finales['Recall']:.3f},
   precisión {metricas_finales['Precision']:.3f}, F1 {metricas_finales['F1-Score']:.3f},
   PR-AUC {metricas_finales['PR-AUC']:.3f}.
   Limitación: la precisión es baja, así que habrá muchas falsas alarmas; el umbral de decisión
   debe ajustarse con el negocio (el simulador de umbral de la app Streamlit sirve para eso).
"""

print(justificacion)
(CARPETA_REPORTES / 'justificacion_modelo.txt').write_text(justificacion, encoding='utf-8')


# 13. Generar gráfico comparativo de rendimiento

metricas_grafico = [
    'Recall_CV',
    'Precision_CV',
    'F1_CV',
    'PR_AUC_CV'
]

tabla_grafico = tabla_resultados.set_index('Modelo')[metricas_grafico]

tabla_grafico.plot(
    kind='bar',
    figsize=(11, 6)
)

plt.title('Comparación del rendimiento de los modelos')
plt.ylabel('Puntuación')
plt.xlabel('Modelo')
plt.ylim(0, 1)
plt.xticks(rotation=15)
plt.legend(title='Métricas')
plt.tight_layout()

plt.savefig(
    CARPETA_REPORTES / 'comparacion_metricas.png',
    dpi=300
)

plt.close()


# 14. Generar gráfico de consistencia

plt.figure(figsize=(9, 5))

plt.bar(
    tabla_resultados['Modelo'],
    tabla_resultados['Recall_CV'],
    yerr=tabla_resultados['Recall_Desv'],
    capsize=5
)

plt.title('Consistencia del Recall en validación cruzada')
plt.ylabel('Recall promedio')
plt.xlabel('Modelo')
plt.ylim(0, 1)
plt.xticks(rotation=15)
plt.tight_layout()

plt.savefig(
    CARPETA_REPORTES / 'consistencia_modelos.png',
    dpi=300
)

plt.close()


# 15. Generar matriz de confusión del mejor modelo

visualizacion = ConfusionMatrixDisplay(
    confusion_matrix=matriz_confusion,
    display_labels=['Pago a tiempo', 'Incumplimiento']
)

visualizacion.plot(
    values_format='d',
    cmap='Blues'
)

plt.title(f'Matriz de confusión: {mejor_nombre}')
plt.tight_layout()

plt.savefig(
    CARPETA_REPORTES / 'matriz_confusion.png',
    dpi=300
)

plt.close()

# 15.1 Generar la curva ROC del mejor modelo

tasa_falsos_positivos, tasa_verdaderos_positivos, _ = roc_curve(
    y_test,
    probabilidades
)

plt.figure(figsize=(7, 6))
plt.plot(
    tasa_falsos_positivos,
    tasa_verdaderos_positivos,
    label=f'{mejor_nombre} (AUC = {metricas_finales["ROC-AUC"]:.3f})'
)
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Clasificador aleatorio')
plt.title('Curva ROC del modelo seleccionado')
plt.xlabel('Tasa de falsos positivos')
plt.ylabel('Tasa de verdaderos positivos')
plt.xlim(0, 1)
plt.ylim(0, 1.05)
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig(
    CARPETA_REPORTES / 'curva_roc.png',
    dpi=300
)
plt.close()


# 16. Generar gráfico de tiempos de entrenamiento

plt.figure(figsize=(9, 5))

plt.bar(
    tabla_resultados['Modelo'],
    tabla_resultados['Tiempo_entrenamiento']
)

plt.title('Tiempo promedio de entrenamiento por modelo')
plt.ylabel('Tiempo en segundos')
plt.xlabel('Modelo')
plt.xticks(rotation=15)
plt.tight_layout()

plt.savefig(
    CARPETA_REPORTES / 'tiempos_entrenamiento.png',
    dpi=300
)

plt.close()


# 17. Guardar el mejor modelo

joblib.dump(
    mejor_modelo,
    CARPETA_MODELOS / 'mejor_modelo.joblib'
)


# 18. Mostrar el resumen final

print('Proceso de modelado y evaluación completado:')

print('Modelo seleccionado:', mejor_nombre)
print('Recall final:', round(metricas_finales['Recall'], 4))
print('F1-Score final:', round(metricas_finales['F1-Score'], 4))

print('\nArchivos generados:')
print('- outputs/models/mejor_modelo.joblib')
print('- outputs/reports/comparacion_modelos.csv')
print('- outputs/reports/evaluacion_final.csv')
print('- outputs/reports/comparacion_metricas.png')
print('- outputs/reports/consistencia_modelos.png')
print('- outputs/reports/matriz_confusion.png')
print('- outputs/reports/curva_roc.png')
print('- outputs/reports/tiempos_entrenamiento.png')

print('\nTodos los procesos finalizaron correctamente.')

from model_preprocessing import CreditDataCleaner

limpiador = CreditDataCleaner()

X_train_limpio = limpiador.fit_transform(X_train)

print('\nCOMPROBACIÓN DEL PREPROCESAMIENTO')

print('Dimensiones originales:', X_train.shape)
print('Dimensiones después de limpiar:', X_train_limpio.shape)

print('\nValores de tipo_credito:')
print(X_train_limpio['tipo_credito'].value_counts(dropna=False))

print('\nValores mínimos de variables monetarias:')

for columna in [
    'capital_prestado',
    'salario_cliente',
    'total_otros_prestamos',
    'saldo_mora',
    'saldo_total',
    'saldo_mora_codeudor',
    'promedio_ingresos_datacredito'
]:
    print(columna, ':', X_train_limpio[columna].min())

    