# 1. Importar las librerías

import numpy as np
import pandas as pd
import joblib
import sys

from pathlib import Path
from datetime import datetime

from scipy.stats import ks_2samp, chi2_contingency
from scipy.spatial.distance import jensenshannon
from mlops_pipeline import model_preprocessing

# Supports loading models serialized before model_preprocessing moved into the package.
sys.modules.setdefault('model_preprocessing', model_preprocessing)


# 2. Definir las rutas

CARPETA_PROYECTO = Path(__file__).resolve().parent.parent

CARPETA_DATASETS = CARPETA_PROYECTO / 'outputs' / 'datasets'
CARPETA_MODELOS = CARPETA_PROYECTO / 'outputs' / 'models'
CARPETA_REPORTES = CARPETA_PROYECTO / 'outputs' / 'reports'

VARIABLE_OBJETIVO = 'Pago_atiempo'


# 3. Cargar los datos y el modelo

datos_train_raw = pd.read_csv(
    CARPETA_DATASETS / 'dataset_train_raw.csv'
)

datos_test_raw = pd.read_csv(
    CARPETA_DATASETS / 'dataset_test_raw.csv'
)

datos_train = pd.read_csv(
    CARPETA_DATASETS / 'dataset_train.csv'
)

datos_test = pd.read_csv(
    CARPETA_DATASETS / 'dataset_test.csv'
)

mejor_modelo = joblib.load(
    CARPETA_MODELOS / 'mejor_modelo.joblib'
)

# Separar las características de la variable objetivo

X_train_raw = datos_train_raw.drop(columns=[VARIABLE_OBJETIVO])
X_test_raw = datos_test_raw.drop(columns=[VARIABLE_OBJETIVO])
X_train = datos_train.drop(columns=[VARIABLE_OBJETIVO])
X_test = datos_test.drop(columns=[VARIABLE_OBJETIVO])

y_test_original = datos_test_raw[VARIABLE_OBJETIVO]

# Transformar las etiquetas para que 1 represente incumplimiento

y_test = (y_test_original == 0).astype(int)

# Comprobar que las columnas coincidan

if list(X_train.columns) != list(X_test.columns):
    raise ValueError('Los conjuntos de datos tienen columnas diferentes.')


# 4. Generar las predicciones

predicciones = mejor_modelo.predict(X_test_raw)

probabilidades = mejor_modelo.predict_proba(X_test_raw)[:, 1]

# Crear una tabla con los datos y sus pronósticos

tabla_monitoreo = X_test.copy()

tabla_monitoreo['Incumplimiento_real'] = y_test.values
tabla_monitoreo['Prediccion_incumplimiento'] = predicciones
tabla_monitoreo['Probabilidad_incumplimiento'] = probabilidades

tabla_monitoreo['Fecha_monitoreo'] = datetime.now().strftime(
    '%Y-%m-%d %H:%M:%S'
)


# 5. Definir funciones para medir Data Drift

def calcular_psi(distribucion_referencia, distribucion_actual):

    epsilon = 0.000001

    referencia = np.array(distribucion_referencia) + epsilon
    actual = np.array(distribucion_actual) + epsilon

    return np.sum(
        (actual - referencia) * np.log(actual / referencia)
    )


def calcular_divergencia_js(distribucion_referencia, distribucion_actual):

    epsilon = 0.000001

    referencia = np.array(distribucion_referencia) + epsilon
    actual = np.array(distribucion_actual) + epsilon

    referencia = referencia / referencia.sum()
    actual = actual / actual.sum()

    return jensenshannon(
        referencia,
        actual,
        base=2
    ) ** 2


# 6. Comparar las distribuciones de los datos

resultados_drift = []

for columna in X_train.columns:

    referencia = X_train[columna].dropna()
    actual = X_test[columna].dropna()

    # Identificar variables binarias (0 y 1)

    es_binaria = set(referencia.unique()).issubset({0, 1})

    if es_binaria:

        categorias = [0, 1]

        conteo_referencia = referencia.value_counts().reindex(
            categorias, fill_value=0
        )

        conteo_actual = actual.value_counts().reindex(
            categorias, fill_value=0
        )

        # Distribuciones porcentuales

        distribucion_referencia = (
            conteo_referencia / conteo_referencia.sum()
        )

        distribucion_actual = (
            conteo_actual / conteo_actual.sum()
        )

        # Prueba Chi-cuadrado

        tabla_contingencia = np.array([
            conteo_referencia.values,
            conteo_actual.values
        ])

        chi2, valor_p, _, _ = chi2_contingency(
            tabla_contingencia
        )

        ks_statistic = np.nan

        tipo_variable = 'Categórica binaria'

    else:

        # Prueba Kolmogorov-Smirnov

        ks_statistic, valor_p = ks_2samp(
            referencia,
            actual
        )

        # Crear intervalos según la distribución de referencia

        limites = np.unique(
            np.quantile(
                referencia,
                np.linspace(0, 1, 11)
            )
        )

        if len(limites) < 2:
            limites = np.array([
                referencia.min() - 1,
                referencia.max() + 1
            ])

        limites[0] = -np.inf
        limites[-1] = np.inf

        conteo_referencia = pd.cut(
            referencia,
            bins=limites
        ).value_counts(sort=False)

        conteo_actual = pd.cut(
            actual,
            bins=limites
        ).value_counts(sort=False)

        distribucion_referencia = (
            conteo_referencia / conteo_referencia.sum()
        )

        distribucion_actual = (
            conteo_actual / conteo_actual.sum()
        )

        tipo_variable = 'Numérica'

    # Calcular PSI y Jensen-Shannon

    psi = calcular_psi(
        distribucion_referencia.values,
        distribucion_actual.values
    )

    divergencia_js = calcular_divergencia_js(
        distribucion_referencia.values,
        distribucion_actual.values
    )

    resultados_drift.append({
        'Variable': columna,
        'Tipo': tipo_variable,
        'KS': ks_statistic,
        'Chi_Cuadrado': chi2 if es_binaria else np.nan,
        'Valor_p': valor_p,
        'PSI': psi,
        'Jensen_Shannon': divergencia_js
    })


# 7. Crear el reporte de Data Drift

reporte_drift = pd.DataFrame(resultados_drift)

reporte_drift = reporte_drift.sort_values(
    by='PSI',
    ascending=False
).reset_index(drop=True)


# 8. Identificar posibles alertas

def interpretar_psi(valor):

    if valor < 0.10:
        return 'Sin cambio significativo según PSI'

    elif valor < 0.25:
        return 'Cambio moderado'

    else:
        return 'Cambio importante'


reporte_drift['Estado_PSI'] = reporte_drift['PSI'].apply(
    interpretar_psi
)



# 9. Guardar los resultados

CARPETA_REPORTES.mkdir(
    parents=True,
    exist_ok=True
)

# Registrar la fecha del monitoreo

reporte_drift['Fecha_monitoreo'] = datetime.now().strftime(
    '%Y-%m-%d %H:%M:%S'
)

# Guardar el reporte actual

reporte_drift.to_csv(
    CARPETA_REPORTES / 'reporte_data_drift.csv',
    index=False
)

# Guardar el historial de monitoreo

RUTA_HISTORIAL = CARPETA_REPORTES / 'historial_data_drift.csv'

if RUTA_HISTORIAL.exists():

    historial = pd.read_csv(RUTA_HISTORIAL)

    historial = pd.concat(
        [historial, reporte_drift],
        ignore_index=True
    )

else:

    historial = reporte_drift.copy()

historial.to_csv(
    RUTA_HISTORIAL,
    index=False
)

# Guardar los datos monitoreados

tabla_monitoreo.to_csv(
    CARPETA_REPORTES / 'datos_monitoreo.csv',
    index=False
)



# 10. Mostrar el resumen

print('\nMONITOREO DEL MODELO COMPLETADO')

print('\nCantidad de registros monitoreados:')
print(len(tabla_monitoreo))

print('\nDistribución de las predicciones:')
print(
    tabla_monitoreo['Prediccion_incumplimiento']
    .value_counts()
    .to_dict()
)

print('\nRESULTADOS DE DATA DRIFT:')
print(reporte_drift.to_string(index=False))

print('\nArchivos generados:')
print('- outputs/reports/datos_monitoreo.csv')
print('- outputs/reports/reporte_data_drift.csv')

print('\nProceso de monitoreo finalizado correctamente.')
