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

# Compatibilidad con modelos guardados anteriormente.
sys.modules.setdefault('model_preprocessing', model_preprocessing)

# 2. Definir las rutas
CARPETA_PROYECTO = Path(__file__).resolve().parent.parent
CARPETA_DATASETS = CARPETA_PROYECTO / 'outputs' / 'datasets'
CARPETA_MODELOS = CARPETA_PROYECTO / 'outputs' / 'models'
CARPETA_REPORTES = CARPETA_PROYECTO / 'outputs' / 'reports'
VARIABLE_OBJETIVO = 'Pago_atiempo'
VARIABLE_FECHA = 'fecha_prestamo'

# 3. Cargar los datos y el modelo
datos_train_raw = pd.read_csv(
    CARPETA_DATASETS / 'dataset_train_raw.csv'
)

datos_test_raw = pd.read_csv(
    CARPETA_DATASETS / 'dataset_test_raw.csv'
)

mejor_modelo = joblib.load(
    CARPETA_MODELOS / 'mejor_modelo.joblib'
)

# Unir los datos originales para recuperar todos los registros.

datos_completos = pd.concat(
    [datos_train_raw, datos_test_raw],
    ignore_index=True
)

# Convertir la fecha a formato datetime.

datos_completos[VARIABLE_FECHA] = pd.to_datetime(
    datos_completos[VARIABLE_FECHA],
    errors='coerce'
)

# Excluir registros sin fecha o sin variable objetivo.

datos_completos = datos_completos.dropna(
    subset=[VARIABLE_FECHA, VARIABLE_OBJETIVO]
).copy()

if datos_completos.empty:
    raise ValueError('No hay registros válidos para realizar el monitoreo.')

if len(datos_completos[VARIABLE_FECHA].unique()) < 2:
    raise ValueError('Se necesitan al menos dos fechas diferentes para comparar períodos.')

# 4. Separar los datos en períodos temporales

datos_completos = datos_completos.sort_values(
    VARIABLE_FECHA
).reset_index(drop=True)

# Utilizar aproximadamente el 75 % de los registros más antiguos
# como referencia y el 25 % más reciente como período actual.

indice_corte = int(len(datos_completos) * 0.75)

fecha_corte = datos_completos.loc[
    indice_corte, VARIABLE_FECHA
]

datos_referencia = datos_completos[
    datos_completos[VARIABLE_FECHA] < fecha_corte
].copy()

datos_actuales = datos_completos[
    datos_completos[VARIABLE_FECHA] >= fecha_corte
].copy()

if datos_referencia.empty or datos_actuales.empty:
    raise ValueError('No fue posible separar los datos en dos períodos temporales.')

print('\nPERÍODOS DE MONITOREO')
print(
    'Referencia:',
    datos_referencia[VARIABLE_FECHA].min().date(),
    'a',
    datos_referencia[VARIABLE_FECHA].max().date()
)
print(
    'Período actual:',
    datos_actuales[VARIABLE_FECHA].min().date(),
    'a',
    datos_actuales[VARIABLE_FECHA].max().date()
)

print('\nCantidad de registros:')
print('Referencia:', len(datos_referencia))
print('Período actual:', len(datos_actuales))


# Separar las características de la variable objetivo.

X_referencia_raw = datos_referencia.drop(
    columns=[VARIABLE_OBJETIVO]
)

X_actual_raw = datos_actuales.drop(
    columns=[VARIABLE_OBJETIVO]
)

y_actual_original = datos_actuales[VARIABLE_OBJETIVO]

# Transformar las etiquetas para que 1 represente incumplimiento.

y_actual = (y_actual_original == 0).astype(int)


# 5. Aplicar el preprocesamiento del modelo

# Se utiliza el mismo preprocesamiento aprendido durante
# el entrenamiento, sin volver a ajustarlo.

preprocesador_modelo = mejor_modelo[:-1]

X_referencia = preprocesador_modelo.transform(
    X_referencia_raw
)

X_actual = preprocesador_modelo.transform(
    X_actual_raw
)

nombres_variables = mejor_modelo.named_steps[
    'preprocesador'
].get_feature_names_out()

X_referencia = pd.DataFrame(
    X_referencia,
    columns=nombres_variables
)

X_actual = pd.DataFrame(
    X_actual,
    columns=nombres_variables
)


# 6. Generar las predicciones del período actual

predicciones = mejor_modelo.predict(X_actual_raw)

probabilidades = mejor_modelo.predict_proba(X_actual_raw)[:, 1]

# Crear una tabla con los datos y sus pronósticos.

tabla_monitoreo = X_actual.copy()

tabla_monitoreo[VARIABLE_FECHA] = (
    datos_actuales[VARIABLE_FECHA].values
)

tabla_monitoreo['Incumplimiento_real'] = y_actual.values

tabla_monitoreo['Prediccion_incumplimiento'] = predicciones

tabla_monitoreo['Probabilidad_incumplimiento'] = probabilidades

tabla_monitoreo['Fecha_monitoreo'] = datetime.now().strftime(
    '%Y-%m-%d %H:%M:%S'
)

# Guardar también los datos del período de referencia
# para utilizar los mismos registros en el dashboard.

tabla_referencia = X_referencia.copy()

tabla_referencia[VARIABLE_FECHA] = (
    datos_referencia[VARIABLE_FECHA].values
)

# 7. Definir funciones para medir Data Drift

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


# 8. Comparar las distribuciones de los datos

resultados_drift = []

for columna in X_referencia.columns:

    referencia = X_referencia[columna].dropna()
    actual = X_actual[columna].dropna()

    # Evitar cálculos cuando no hay suficientes datos.

    if referencia.empty or actual.empty:
        continue

    # Identificar variables binarias.

    valores = set(referencia.unique()).union(
        set(actual.unique())
    )

    es_binaria = valores.issubset({0, 1})

    chi2 = np.nan

    if es_binaria:

        categorias = [0, 1]

        conteo_referencia = referencia.value_counts().reindex(
            categorias,
            fill_value=0
        )

        conteo_actual = actual.value_counts().reindex(
            categorias,
            fill_value=0
        )

        # Distribuciones porcentuales.

        distribucion_referencia = (
            conteo_referencia / conteo_referencia.sum()
        )

        distribucion_actual = (
            conteo_actual / conteo_actual.sum()
        )

        # Prueba Chi-cuadrado.

        tabla_contingencia = np.array([
            conteo_referencia.values,
            conteo_actual.values
        ])

        # Eliminar categorías sin observaciones en ambos períodos.

        tabla_contingencia = tabla_contingencia[
            :, tabla_contingencia.sum(axis=0) > 0
        ]

        if tabla_contingencia.shape[1] >= 2:
            chi2, valor_p, _, _ = chi2_contingency(
                tabla_contingencia
            )
        else:
            valor_p = np.nan

        ks_statistic = np.nan

        tipo_variable = 'Binaria'

    else:

        # Prueba Kolmogorov-Smirnov.

        ks_statistic, valor_p = ks_2samp(
            referencia,
            actual
        )

        # Crear intervalos según la distribución de referencia.

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

        # Distribuciones porcentuales.

        distribucion_referencia = (
            conteo_referencia / conteo_referencia.sum()
        )

        distribucion_actual = (
            conteo_actual / conteo_actual.sum()
        )

        tipo_variable = 'Numérica'

    # Calcular PSI y Jensen-Shannon.

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
        'Chi_Cuadrado': chi2,
        'Valor_p': valor_p,
        'PSI': psi,
        'Jensen_Shannon': divergencia_js
    })


# 9. Crear el reporte de Data Drift

reporte_drift = pd.DataFrame(resultados_drift)

if reporte_drift.empty:
    raise ValueError('No se pudieron calcular las métricas de Data Drift.')

reporte_drift = reporte_drift.sort_values(
    by='PSI',
    ascending=False
).reset_index(drop=True)


# 10. Identificar posibles alertas

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


# 11. Guardar los resultados

CARPETA_REPORTES.mkdir(
    parents=True,
    exist_ok=True
)

fecha_monitoreo = datetime.now().strftime(
    '%Y-%m-%d %H:%M:%S'
)

reporte_drift['Fecha_monitoreo'] = fecha_monitoreo

reporte_drift['Inicio_referencia'] = (
    datos_referencia[VARIABLE_FECHA].min().date()
)

reporte_drift['Fin_referencia'] = (
    datos_referencia[VARIABLE_FECHA].max().date()
)

reporte_drift['Inicio_actual'] = (
    datos_actuales[VARIABLE_FECHA].min().date()
)

reporte_drift['Fin_actual'] = (
    datos_actuales[VARIABLE_FECHA].max().date()
)

# Guardar el reporte actual.

reporte_drift.to_csv(
    CARPETA_REPORTES / 'reporte_data_drift.csv',
    index=False
)

# Guardar el historial de monitoreo.

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

# Guardar los datos del período de referencia.

tabla_referencia.to_csv(
    CARPETA_REPORTES / 'datos_referencia.csv',
    index=False
)

# Guardar los datos del período actual.

tabla_monitoreo.to_csv(
    CARPETA_REPORTES / 'datos_monitoreo.csv',
    index=False
)

# 12. Mostrar el resumen

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
print('- outputs/reports/historial_data_drift.csv')
print('- outputs/reports/datos_referencia.csv')

print('\nProceso de monitoreo finalizado correctamente.')
