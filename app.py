
# 1. Importar las librerías

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path


# 2. Definir las rutas

CARPETA_PROYECTO = Path(__file__).resolve().parent

CARPETA_DATASETS = CARPETA_PROYECTO / 'outputs' / 'datasets'
CARPETA_REPORTES = CARPETA_PROYECTO / 'outputs' / 'reports'


# 3. Configurar la aplicación

st.set_page_config(
    page_title='Monitoreo del modelo de crédito',
    page_icon='📊',
    layout='wide'
)

st.title('Monitoreo del modelo de riesgo crediticio')

st.write(
    'Panel de seguimiento de Data Drift y estabilidad '
    'de las variables utilizadas por el modelo.'
)


# 4. Cargar los archivos

try:

    datos_train = pd.read_csv(
        CARPETA_DATASETS / 'dataset_train.csv'
    )

    datos_monitoreo = pd.read_csv(
        CARPETA_REPORTES / 'datos_monitoreo.csv'
    )

    reporte_drift = pd.read_csv(
        CARPETA_REPORTES / 'reporte_data_drift.csv'
    )

except FileNotFoundError as error:

    st.error(f'No se encontró uno de los archivos: {error}')
    st.stop()


# Cargar el historial si existe

RUTA_HISTORIAL = CARPETA_REPORTES / 'historial_data_drift.csv'

if RUTA_HISTORIAL.exists():

    historial = pd.read_csv(RUTA_HISTORIAL)

    historial['Fecha_monitoreo'] = pd.to_datetime(
        historial['Fecha_monitoreo']
    )

else:

    historial = pd.DataFrame()


# 5. Mostrar indicadores generales

st.header('Resumen del monitoreo')

total_variables = len(reporte_drift)

cambios_importantes = (
    reporte_drift['PSI'] >= 0.25
).sum()

cambios_moderados = (
    (reporte_drift['PSI'] >= 0.10) &
    (reporte_drift['PSI'] < 0.25)
).sum()

diferencias_estadisticas = (
    reporte_drift['Valor_p'] < 0.05
).sum()

col1, col2, col3, col4 = st.columns(4)

col1.metric('Variables analizadas', total_variables)

col2.metric('Cambios importantes', int(cambios_importantes))

col3.metric('Cambios moderados', int(cambios_moderados))

col4.metric('Diferencias estadísticas', int(diferencias_estadisticas))


# 6. Mostrar alertas

st.header('Estado del Data Drift')

if cambios_importantes > 0:

    st.error(
        'Alerta: se detectaron variables con cambios importantes. '
        'Se recomienda investigar su comportamiento.'
    )

elif cambios_moderados > 0:

    st.warning(
        'Se detectaron cambios moderados. '
        'Se recomienda realizar seguimiento.'
    )

else:

    st.success(
        'No se detectaron cambios importantes según los umbrales PSI.'
    )

if diferencias_estadisticas > 0:

    st.info(
        f'{diferencias_estadisticas} variables presentan diferencias '
        'estadísticas según su prueba correspondiente. '
        'Revisar su magnitud antes de generar una alerta.'
    )


# 7. Comparación de distribuciones

st.header('Distribución histórica vs. actual')

variables_disponibles = [
    columna for columna in datos_train.columns
    if columna in datos_monitoreo.columns
    and columna != 'Pago_atiempo'
]

variable_seleccionada = st.selectbox(
    'Selecciona una variable:',
    variables_disponibles
)

referencia = datos_train[variable_seleccionada].dropna()

actual = datos_monitoreo[variable_seleccionada].dropna()

if set(referencia.unique()).issubset({0, 1}):

    distribucion_referencia = (
        referencia.value_counts(normalize=True)
        .reindex([0, 1], fill_value=0)
    )

    distribucion_actual = (
        actual.value_counts(normalize=True)
        .reindex([0, 1], fill_value=0)
    )

    comparacion = pd.DataFrame({
        'Histórica': distribucion_referencia,
        'Actual': distribucion_actual
    })

    comparacion.index = ['0', '1']

else:

    minimo = min(referencia.min(), actual.min())
    maximo = max(referencia.max(), actual.max())

    if minimo == maximo:

        st.info('La variable presenta un único valor en ambos conjuntos.')

        comparacion = None

    else:

        limites = np.linspace(minimo, maximo, 11)

        conteo_referencia, _ = np.histogram(
            referencia,
            bins=limites
        )

        conteo_actual, _ = np.histogram(
            actual,
            bins=limites
        )

        porcentajes_referencia = (
            conteo_referencia / len(referencia) * 100
        )

        porcentajes_actual = (
            conteo_actual / len(actual) * 100
        )

        etiquetas = [
            f'{limites[i]:.1f} - {limites[i+1]:.1f}'
            for i in range(len(limites) - 1)
        ]

        comparacion = pd.DataFrame({
            'Histórica': porcentajes_referencia,
            'Actual': porcentajes_actual
        }, index=etiquetas)

if comparacion is not None:

    st.bar_chart(comparacion)

    st.caption(
        'Comparación porcentual de las distribuciones '
        'de referencia y actuales.'
    )


# 8. Tabla de métricas por variable

st.header('Métricas de Data Drift')

columnas_tabla = [
    'Variable',
    'Tipo',
    'KS',
    'Chi_Cuadrado',
    'Valor_p',
    'PSI',
    'Jensen_Shannon',
    'Estado_PSI'
]

st.dataframe(
    reporte_drift[columnas_tabla],
    use_container_width=True,
    hide_index=True
)


# 9. Análisis temporal

st.header('Evolución del Data Drift')

if not historial.empty:

    fecha_maxima = historial.groupby(
        'Fecha_monitoreo'
    )['PSI'].max()

    st.subheader('Evolución del PSI máximo')

    st.line_chart(fecha_maxima)

    variable_temporal = st.selectbox(
        'Selecciona una variable para revisar su evolución:',
        sorted(historial['Variable'].unique())
    )

    evolucion = historial[
        historial['Variable'] == variable_temporal
    ].sort_values('Fecha_monitoreo')

    st.line_chart(
        evolucion.set_index('Fecha_monitoreo')['PSI']
    )

else:

    st.info(
        'Todavía no existe un historial de monitoreo. '
        'Ejecuta model_monitoring.py para registrar el primer resultado.'
    )


# 10. Recomendaciones automáticas

st.header('Recomendaciones')

if cambios_importantes > 0:

    st.error(
        'Se recomienda investigar las variables con PSI igual o '
        'superior a 0.25, revisar la calidad de los datos y evaluar '
        'la necesidad de actualizar el modelo.'
    )

elif cambios_moderados > 0:

    st.warning(
        'Se recomienda revisar las variables con PSI entre 0.10 y 0.25 '
        'y mantener un seguimiento en las próximas ejecuciones.'
    )

else:

    st.success(
        'Las distribuciones se mantienen estables según PSI. '
        'Se recomienda continuar con el monitoreo periódico.'
    )

if diferencias_estadisticas > 0:

    st.info(
        'Revisar las variables con diferencias estadísticas para '
        'determinar si el cambio tiene relevancia práctica.'
    )

st.caption(
    'Nota: detectar Data Drift no significa necesariamente que '
    'el rendimiento predictivo haya disminuido. Para decidir '
    'un reentrenamiento se debe evaluar también el desempeño '
    'con resultados reales.'
)
