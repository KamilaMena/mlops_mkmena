"""Dashboard de monitoreo de riesgo crediticio."""

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ---------------------------------------------------------------------
# 1. Rutas y configuración
# ---------------------------------------------------------------------

CARPETA_PROYECTO = Path(__file__).resolve().parent
CARPETA_REPORTES = CARPETA_PROYECTO / "outputs" / "reports"

COLOR_REFERENCIA = "#5276A7"
COLOR_ACTUAL = "#E58A45"

COLORES_ESTADO = {
    "Estable": "#278568",
    "Moderado": "#D99A32",
    "Importante": "#C94C4C",
}


st.set_page_config(
    page_title="Monitoreo de riesgo crediticio",
    page_icon="📊",
    layout="wide",
)


# ---------------------------------------------------------------------
# 2. Estilos
# ---------------------------------------------------------------------

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .encabezado {
        padding: 30px;
        border-radius: 14px;
        background: linear-gradient(115deg, #172B46, #245B68);
        color: white;
        margin-bottom: 25px;
    }

    .encabezado h1 {
        color: white;
        font-size: 2rem;
        margin-bottom: 8px;
    }

    .encabezado p {
        color: #DCE8EE;
        margin-bottom: 0;
        font-size: 1rem;
    }

    .estado {
        padding: 17px 20px;
        border-radius: 10px;
        margin: 12px 0 22px 0;
        border-left: 5px solid var(--color);
        background: color-mix(in srgb, var(--color) 10%, transparent);
    }

    .estado strong {
        font-size: 1.05rem;
    }

    .estado p {
        margin: 5px 0 0 0;
    }

    .recomendacion {
        padding: 15px 18px;
        margin-bottom: 12px;
        border-left: 4px solid var(--color);
        border-radius: 8px;
        background: color-mix(in srgb, var(--color) 9%, transparent);
    }

    .recomendacion strong {
        display: block;
        margin-bottom: 5px;
    }

    h2, h3 {
        color: inherit;
        font-weight: 650;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------
# 3. Funciones auxiliares
# ---------------------------------------------------------------------

@st.cache_data
def cargar_csv(ruta, fecha_modificacion):
    return pd.read_csv(ruta)


def cargar_archivo(nombre):
    ruta = CARPETA_REPORTES / nombre

    if not ruta.exists():
        st.error(
            f"No se encontró el archivo `{nombre}`. "
            "Ejecuta primero `model_monitoring.py`."
        )
        st.stop()

    return cargar_csv(str(ruta), ruta.stat().st_mtime)


def clasificar_psi(valor):
    if valor >= 0.25:
        return "Importante"
    elif valor >= 0.10:
        return "Moderado"
    return "Estable"


def es_binaria(serie):
    valores = set(serie.dropna().unique())
    return bool(valores) and valores.issubset({0, 1})


def estilo_grafico(fig, altura=400):
    fig.update_layout(
        height=altura,
        margin=dict(l=15, r=15, t=55, b=15),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial, sans-serif"),
        title=dict(x=0),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            x=0,
        ),
    )

    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(128,128,128,0.15)",
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(128,128,128,0.15)",
    )

    return fig


def mostrar_recomendacion(titulo, mensaje, estado):
    color = COLORES_ESTADO[estado]

    st.markdown(
        f"""
        <div class="recomendacion" style="--color:{color}">
            <strong>{titulo}</strong>
            {mensaje}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------
# 4. Cargar resultados del monitoreo
# ---------------------------------------------------------------------

datos_referencia = cargar_archivo("datos_referencia.csv")
datos_actuales = cargar_archivo("datos_monitoreo.csv")
reporte = cargar_archivo("reporte_data_drift.csv")
historial = cargar_archivo("historial_data_drift.csv")

reporte["Estado"] = reporte["PSI"].apply(clasificar_psi)

historial["Fecha_monitoreo"] = pd.to_datetime(
    historial["Fecha_monitoreo"],
    errors="coerce",
)

fecha_monitoreo = (
    str(reporte["Fecha_monitoreo"].iloc[0])
    if not reporte.empty
    else "No disponible"
)

n_registros = len(datos_actuales)
n_variables = len(reporte)

n_importantes = (reporte["Estado"] == "Importante").sum()
n_moderadas = (reporte["Estado"] == "Moderado").sum()

if n_importantes > 0:
    estado_general = "Importante"
elif n_moderadas > 0:
    estado_general = "Moderado"
else:
    estado_general = "Estable"


# ---------------------------------------------------------------------
# 5. Encabezado
# ---------------------------------------------------------------------

st.markdown(
    """
    <div class="encabezado">
        <h1>Monitoreo de riesgo crediticio</h1>
        <p>
            Seguimiento de la estabilidad de los datos y detección
            de cambios en las distribuciones del modelo.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns(3)

col1.metric("Registros del período actual", f"{n_registros:,}")
col2.metric("Variables analizadas", n_variables)
col3.metric("Estado general", estado_general)

mensajes_estado = {
    "Estable": "No se detectan cambios importantes en las distribuciones.",
    "Moderado": (
        f"{n_moderadas} variable(s) presentan cambios moderados. "
        "Se recomienda mantener su seguimiento."
    ),
    "Importante": (
        f"{n_importantes} variable(s) presentan cambios importantes. "
        "Se recomienda investigar sus causas."
    ),
}

st.markdown(
    f"""
    <div class="estado" style="--color:{COLORES_ESTADO[estado_general]}">
        <strong>Estado del monitoreo: {estado_general}</strong>
        <p>{mensajes_estado[estado_general]}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption(f"Último monitoreo: {fecha_monitoreo}")


# ---------------------------------------------------------------------
# 6. Secciones del dashboard
# ---------------------------------------------------------------------

tab_resumen, tab_distribuciones, tab_metricas, tab_evolucion = st.tabs(
    [
        "01 · Resumen",
        "02 · Distribuciones",
        "03 · Métricas de Data Drift",
        "04 · Evolución y recomendaciones",
    ]
)


# ---------------------------------------------------------------------
# 7. Resumen
# ---------------------------------------------------------------------

with tab_resumen:

    st.subheader("Períodos comparados")

    fecha_inicio_ref = reporte["Inicio_referencia"].iloc[0]
    fecha_fin_ref = reporte["Fin_referencia"].iloc[0]
    fecha_inicio_act = reporte["Inicio_actual"].iloc[0]
    fecha_fin_act = reporte["Fin_actual"].iloc[0]

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Período de referencia**")
        st.write(f"{fecha_inicio_ref} — {fecha_fin_ref}")
        st.caption(
            f"{len(datos_referencia):,} registros utilizados"
        )

    with c2:
        st.markdown("**Período actual**")
        st.write(f"{fecha_inicio_act} — {fecha_fin_act}")
        st.caption(
            f"{len(datos_actuales):,} registros utilizados"
        )

    st.divider()

    st.subheader("Variables con mayor Data Drift")

    top = reporte.sort_values(
        "PSI",
        ascending=True,
    ).tail(12)

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=top["PSI"],
            y=top["Variable"],
            orientation="h",
            marker_color=[
                COLORES_ESTADO[estado]
                for estado in top["Estado"]
            ],
            text=top["PSI"].round(3),
            textposition="outside",
        )
    )

    fig.add_vline(
        x=0.10,
        line_dash="dash",
        line_color="#D99A32",
        annotation_text="Moderado",
    )

    fig.add_vline(
        x=0.25,
        line_dash="dash",
        line_color="#C94C4C",
        annotation_text="Importante",
    )

    fig.update_layout(
        title="Índice de estabilidad poblacional (PSI)",
        xaxis_title="PSI",
        yaxis_title="Variable",
    )

    st.plotly_chart(
        estilo_grafico(fig, 480),
        use_container_width=True,
    )


# ---------------------------------------------------------------------
# 8. Comparación de distribuciones
# ---------------------------------------------------------------------

with tab_distribuciones:

    st.subheader("Comparación de distribuciones")

    variables_disponibles = [
        variable
        for variable in reporte["Variable"]
        if (
            variable in datos_referencia.columns
            and variable in datos_actuales.columns
        )
    ]

    if not variables_disponibles:
        st.warning(
            "No hay variables disponibles para comparar."
        )

    else:

        variables_ordenadas = (
            reporte.sort_values("PSI", ascending=False)["Variable"]
            .tolist()
        )

        variables_ordenadas = [
            variable
            for variable in variables_ordenadas
            if variable in variables_disponibles
        ]

        variable = st.selectbox(
            "Selecciona una variable",
            variables_ordenadas,
        )

        fila = reporte[
            reporte["Variable"] == variable
        ].iloc[0]

        referencia = datos_referencia[variable].dropna()
        actual = datos_actuales[variable].dropna()

        estado = fila["Estado"]

        st.markdown(
            f"**Estado de la variable:** {estado}"
        )

        m1, m2, m3 = st.columns(3)

        m1.metric("PSI", f"{fila['PSI']:.4f}")
        m2.metric(
            "Jensen-Shannon",
            f"{fila['Jensen_Shannon']:.4f}",
        )

        if pd.notna(fila["KS"]):
            m3.metric("KS", f"{fila['KS']:.4f}")
        else:
            m3.metric(
                "Chi-cuadrado",
                f"{fila['Chi_Cuadrado']:.4f}",
            )

        if referencia.empty or actual.empty:
            st.warning(
                "No hay suficientes datos para representar esta variable."
            )

        elif es_binaria(referencia):

            comparacion = pd.DataFrame({
                "Valor": ["0", "1", "0", "1"],
                "Período": [
                    "Referencia",
                    "Referencia",
                    "Actual",
                    "Actual",
                ],
                "Porcentaje": (
                    list(
                        referencia.value_counts(normalize=True)
                        .reindex([0, 1], fill_value=0) * 100
                    )
                    +
                    list(
                        actual.value_counts(normalize=True)
                        .reindex([0, 1], fill_value=0) * 100
                    )
                ),
            })

            fig = px.bar(
                comparacion,
                x="Valor",
                y="Porcentaje",
                color="Período",
                barmode="group",
                color_discrete_map={
                    "Referencia": COLOR_REFERENCIA,
                    "Actual": COLOR_ACTUAL,
                },
                text_auto=".1f",
                title=f"Distribución de {variable}",
            )

            fig.update_layout(
                yaxis_title="Porcentaje de registros (%)",
                xaxis_title="Valor",
            )

        else:

            minimo = min(referencia.min(), actual.min())
            maximo = max(referencia.max(), actual.max())

            if minimo == maximo:
                st.info(
                    "La variable presenta el mismo valor en ambos períodos."
                )
                fig = None

            else:

                fig = go.Figure()

                fig.add_trace(
                    go.Histogram(
                        x=referencia,
                        name="Referencia",
                        opacity=0.65,
                        marker_color=COLOR_REFERENCIA,
                        histnorm="percent",
                        nbinsx=25,
                    )
                )

                fig.add_trace(
                    go.Histogram(
                        x=actual,
                        name="Actual",
                        opacity=0.65,
                        marker_color=COLOR_ACTUAL,
                        histnorm="percent",
                        nbinsx=25,
                    )
                )

                fig.update_layout(
                    title=f"Distribución de {variable}",
                    barmode="overlay",
                    xaxis_title="Valor de la variable",
                    yaxis_title="Porcentaje de registros (%)",
                )

        if fig is not None:
            st.plotly_chart(
                estilo_grafico(fig, 430),
                use_container_width=True,
            )


# ---------------------------------------------------------------------
# 9. Tabla de métricas
# ---------------------------------------------------------------------

with tab_metricas:

    st.subheader("Resultados de las métricas de Data Drift")

    st.write(
        "Comparación estadística entre el período de referencia "
        "y el período actual."
    )

    columnas = [
        "Variable",
        "Tipo",
        "KS",
        "Chi_Cuadrado",
        "Valor_p",
        "PSI",
        "Jensen_Shannon",
        "Estado",
    ]

    tabla = reporte[columnas].copy()

    tabla = tabla.sort_values(
        "PSI",
        ascending=False,
    )

    tabla = tabla.rename(
        columns={
            "Chi_Cuadrado": "Chi-cuadrado",
            "Valor_p": "Valor p",
            "Jensen_Shannon": "Jensen-Shannon",
        }
    )

    st.dataframe(
        tabla,
        use_container_width=True,
        hide_index=True,
        column_config={
            "KS": st.column_config.NumberColumn(format="%.4f"),
            "Chi-cuadrado": st.column_config.NumberColumn(
                format="%.4f"
            ),
            "Valor p": st.column_config.NumberColumn(
                format="%.4f"
            ),
            "PSI": st.column_config.NumberColumn(format="%.4f"),
            "Jensen-Shannon": st.column_config.NumberColumn(
                format="%.4f"
            ),
        },
    )

    st.caption(
        "Los valores de KS se utilizan para variables numéricas "
        "y Chi-cuadrado para variables binarias."
    )


# ---------------------------------------------------------------------
# 10. Evolución temporal y recomendaciones
# ---------------------------------------------------------------------

with tab_evolucion:

    st.subheader("Evolución temporal del Data Drift")

    historial_valido = historial.dropna(
        subset=["Fecha_monitoreo", "PSI"]
    ).copy()

    resumen_temporal = (
        historial_valido
        .groupby("Fecha_monitoreo")["PSI"]
        .agg(
            PSI_maximo="max",
            PSI_promedio="mean",
        )
        .reset_index()
        .sort_values("Fecha_monitoreo")
    )

    if resumen_temporal["Fecha_monitoreo"].nunique() < 2:

        st.info(
            "Se necesita más de una ejecución del monitoreo "
            "para mostrar la evolución temporal del PSI. "
            "Por ahora se presenta el resultado de la ejecución actual."
        )

    else:

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=resumen_temporal["Fecha_monitoreo"],
                y=resumen_temporal["PSI_maximo"],
                mode="lines+markers",
                name="PSI máximo",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=resumen_temporal["Fecha_monitoreo"],
                y=resumen_temporal["PSI_promedio"],
                mode="lines+markers",
                name="PSI promedio",
            )
        )

        fig.add_hline(
            y=0.10,
            line_dash="dash",
            line_color="#D99A32",
            annotation_text="Moderado",
        )

        fig.add_hline(
            y=0.25,
            line_dash="dash",
            line_color="#C94C4C",
            annotation_text="Importante",
        )

        fig.update_layout(
            title="Comportamiento del PSI a través de las ejecuciones",
            xaxis_title="Fecha de monitoreo",
            yaxis_title="PSI",
        )

        st.plotly_chart(
            estilo_grafico(fig, 400),
            use_container_width=True,
        )

    st.divider()

    st.subheader("Recomendaciones")

    criticas = reporte[
        reporte["Estado"] == "Importante"
    ].sort_values("PSI", ascending=False)

    moderadas = reporte[
        reporte["Estado"] == "Moderado"
    ].sort_values("PSI", ascending=False)

    if not criticas.empty:

        variables = ", ".join(
            criticas["Variable"].head(6).tolist()
        )

        mostrar_recomendacion(
            "Investigar cambios importantes",
            (
                f"Las siguientes variables presentan cambios importantes: "
                f"{variables}. Se recomienda revisar las fuentes de datos, "
                "los procesos de captura y los cambios en el perfil de los "
                "clientes. También conviene evaluar el desempeño del modelo "
                "antes de considerar un reentrenamiento."
            ),
            "Importante",
        )

    if not moderadas.empty:

        variables = ", ".join(
            moderadas["Variable"].head(6).tolist()
        )

        mostrar_recomendacion(
            "Mantener seguimiento",
            (
                f"Se identificaron cambios moderados en: {variables}. "
                "Se recomienda observar su comportamiento en los "
                "siguientes monitoreos."
            ),
            "Moderado",
        )

    if criticas.empty and moderadas.empty:

        mostrar_recomendacion(
            "Mantener el monitoreo",
            (
                "Las variables analizadas presentan distribuciones "
                "estables según los umbrales de PSI establecidos."
            ),
            "Estable",
        )

    st.info(
        "La detección de Data Drift indica cambios en los datos de entrada, "
        "pero no demuestra por sí sola que el modelo haya perdido precisión. "
        "Es necesario contrastar estos resultados con el desempeño real "
        "del modelo."
    )