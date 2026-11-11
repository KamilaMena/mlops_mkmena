"""Panel de monitoreo del modelo de riesgo crediticio (Streamlit).

Requiere haber ejecutado antes los módulos de preparación y monitoreo de
mlops_pipeline (genera los archivos dentro de outputs/).
Dependencias adicionales: plotly.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ---------------------------------------------------------------------------
# 1. Rutas y constantes
# ---------------------------------------------------------------------------

CARPETA_PROYECTO = Path(__file__).resolve().parent
CARPETA_DATASETS = CARPETA_PROYECTO / "outputs" / "datasets"
CARPETA_REPORTES = CARPETA_PROYECTO / "outputs" / "reports"

OBJETIVO = "Pago_atiempo"

TINTA = "#16263B"
TEAL = "#0F766E"
VERDE = "#2E8B63"
AMBAR = "#D98E04"
ROJO = "#C8372D"
COLOR_HIST = "#5B7DB1"
COLOR_ACTUAL = "#E07A3F"

COLOR_ESTADO = {"Estable": VERDE, "Moderado": AMBAR, "Crítico": ROJO}
ICONO_ESTADO = {"Estable": "🟢", "Moderado": "🟡", "Crítico": "🔴"}


# ---------------------------------------------------------------------------
# 2. Configuración de la página y estilos
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Monitoreo de riesgo crediticio",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"], .stMarkdown, button, input, select, textarea {{
        font-family: 'Plus Jakarta Sans', sans-serif;
    }}

    .block-container {{ padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1400px; }}

    .cabecera {{
        background: linear-gradient(120deg, {TINTA} 0%, #1F3A5C 60%, {TEAL} 130%);
        border-radius: 18px;
        padding: 26px 32px;
        margin-bottom: 22px;
        color: #fff;
    }}
    .cabecera h1 {{ color: #fff; font-size: 1.9rem; font-weight: 800; margin: 0 0 4px 0; padding: 0; }}
    .cabecera p {{ color: rgba(255,255,255,.78); margin: 0; font-size: .98rem; }}
    .cabecera .meta {{ margin-top: 12px; font-size: .82rem; color: rgba(255,255,255,.65); }}

    .estado {{
        border-radius: 14px; padding: 16px 22px; margin: 6px 0 18px 0;
        display: flex; align-items: center; gap: 16px;
        border: 1px solid var(--acento);
        background: color-mix(in srgb, var(--acento) 12%, transparent);
    }}
    .estado .punto {{
        width: 18px; height: 18px; border-radius: 50%; background: var(--acento);
        box-shadow: 0 0 0 6px color-mix(in srgb, var(--acento) 25%, transparent);
        flex-shrink: 0;
    }}
    .estado .texto b {{ font-size: 1.05rem; }}
    .estado .texto span {{ display: block; font-size: .88rem; opacity: .8; margin-top: 2px; }}

    .pill {{
        display: inline-block; padding: 3px 12px; border-radius: 999px;
        font-size: .78rem; font-weight: 700; color: #fff; background: var(--acento);
    }}

    .reco {{
        border-radius: 12px; padding: 12px 16px; margin-bottom: 10px;
        border-left: 5px solid var(--acento);
        background: color-mix(in srgb, var(--acento) 10%, transparent);
    }}
    .reco b {{ display: block; margin-bottom: 2px; }}

    .stTabs [data-baseweb="tab-list"] {{ gap: 6px; }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 10px 10px 0 0; padding: 8px 18px; font-weight: 600;
    }}
    .stTabs [aria-selected="true"] {{ color: {TEAL}; }}

    h2, h3 {{ font-weight: 700; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# 3. Funciones auxiliares
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def leer_csv(ruta: str, firma: float) -> pd.DataFrame:
    """Lee un CSV y actualiza la caché cuando cambia el archivo."""
    return pd.read_csv(ruta)


def cargar(ruta: Path, obligatorio: bool = True):
    """Carga un archivo CSV y muestra un mensaje si falta uno obligatorio."""
    if not ruta.exists():
        if obligatorio:
            st.error(
                f"No se encontró `{ruta.relative_to(CARPETA_PROYECTO)}`. "
                "Ejecuta primero los módulos de preparación y monitoreo "
                "de mlops_pipeline."
            )
            st.stop()
        return None
    return leer_csv(str(ruta), ruta.stat().st_mtime)


def estado_psi(valor, umbral_mod, umbral_crit):
    """Clasifica el nivel de drift según el PSI."""
    if valor >= umbral_crit:
        return "Crítico"
    if valor >= umbral_mod:
        return "Moderado"
    return "Estable"


def es_binaria(serie: pd.Series) -> bool:
    """Indica si una variable contiene únicamente los valores 0 y 1."""
    valores = set(serie.dropna().unique())
    return bool(valores) and valores.issubset({0, 1})


def estilo_plotly(fig, alto=380):
    """Aplica un estilo visual común a los gráficos."""
    fig.update_layout(
        height=alto,
        margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Plus Jakarta Sans, sans-serif"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        title=dict(x=0, font=dict(size=16)),
    )
    fig.update_xaxes(gridcolor="rgba(128,128,128,.18)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,.18)", zeroline=False)
    return fig


# ---------------------------------------------------------------------------
# 4. Carga de datos
# ---------------------------------------------------------------------------

datos_train = cargar(CARPETA_DATASETS / "dataset_train.csv")
datos_monitoreo = cargar(CARPETA_REPORTES / "datos_monitoreo.csv")
reporte = cargar(CARPETA_REPORTES / "reporte_data_drift.csv")
historial = cargar(
    CARPETA_REPORTES / "historial_data_drift.csv",
    obligatorio=False,
)

if historial is not None and not historial.empty:
    historial["Fecha_monitoreo"] = pd.to_datetime(historial["Fecha_monitoreo"])
else:
    historial = pd.DataFrame()

fecha_ultima = (
    str(reporte["Fecha_monitoreo"].iloc[0])
    if "Fecha_monitoreo" in reporte.columns and len(reporte)
    else "sin fecha"
)


# ---------------------------------------------------------------------------
# 5. Configuración de filtros y umbrales
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## ⚙️ Configuración")

    st.markdown("**Umbrales de alerta (PSI)**")
    umbral_mod = st.slider(
        "Cambio moderado desde", 0.05, 0.30, 0.10, 0.01
    )
    umbral_crit = st.slider(
        "Cambio crítico desde", 0.15, 0.60, 0.25, 0.01
    )
    if umbral_crit <= umbral_mod:
        st.warning("El umbral crítico debe ser mayor que el moderado.")
        umbral_crit = umbral_mod + 0.05

    alfa = st.select_slider(
        "Nivel de significancia (valor p)",
        options=[0.01, 0.05, 0.10],
        value=0.05,
    )

    st.divider()
    st.markdown("**Filtros de variables**")
    tipos = sorted(reporte["Tipo"].dropna().unique())
    tipos_sel = st.multiselect("Tipo de variable", tipos, default=tipos)
    estados_sel = st.multiselect(
        "Estado",
        ["Estable", "Moderado", "Crítico"],
        default=["Estable", "Moderado", "Crítico"],
    )
    busqueda = st.text_input(
        "Buscar variable", placeholder="p. ej. puntaje"
    )

    st.divider()
    st.caption(f"Último monitoreo: **{fecha_ultima}**")
    if st.button("🔄 Recargar archivos", use_container_width=True):
        st.cache_data.clear()
        st.rerun()


# Se recalculan las etiquetas para que respondan a los umbrales elegidos.
reporte = reporte.copy()
reporte["Estado"] = reporte["PSI"].apply(
    lambda valor: estado_psi(valor, umbral_mod, umbral_crit)
)
reporte["Significativo"] = reporte["Valor_p"] < alfa

reporte_f = reporte[
    reporte["Tipo"].isin(tipos_sel)
    & reporte["Estado"].isin(estados_sel)
    & reporte["Variable"].str.contains(busqueda, case=False, na=False)
]

n_total = len(reporte)
n_crit = int((reporte["Estado"] == "Crítico").sum())
n_mod = int((reporte["Estado"] == "Moderado").sum())

if n_crit > 0:
    nivel = "Crítico"
elif n_mod > 0:
    nivel = "Moderado"
else:
    nivel = "Estable"


# ---------------------------------------------------------------------------
# 6. Encabezado y alerta general
# ---------------------------------------------------------------------------

st.markdown(
    f"""
    <div class="cabecera">
        <h1>🛡️ Monitoreo del modelo de riesgo crediticio</h1>
        <p>Seguimiento de la estabilidad de las variables y detección de data drift.</p>
        <div class="meta">Último monitoreo: {fecha_ultima} &nbsp;|&nbsp;
        {len(datos_monitoreo):,} registros monitoreados &nbsp;|&nbsp;
        {n_total} variables analizadas</div>
    </div>
    """,
    unsafe_allow_html=True,
)

mensajes_estado = {
    "Crítico": (
        f"{n_crit} variable(s) superan el umbral crítico de PSI "
        f"({umbral_crit:.2f}). Revisa la pestaña Recomendaciones."
    ),
    "Moderado": (
        f"{n_mod} variable(s) muestran cambios moderados. "
        "Mantén seguimiento en la próxima ejecución."
    ),
    "Estable": (
        "Todas las variables se mantienen dentro de los umbrales definidos."
    ),
}

st.markdown(
    f"""
    <div class="estado" style="--acento:{COLOR_ESTADO[nivel]}">
        <div class="punto"></div>
        <div class="texto"><b>Estado general: {nivel}</b>
        <span>{mensajes_estado[nivel]}</span></div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# 7. Secciones principales del dashboard
# ---------------------------------------------------------------------------

tab_resumen, tab_dist, tab_metricas, tab_tiempo, tab_reco = st.tabs(
    [
        "📊 Resumen",
        "📈 Distribuciones",
        "🧮 Métricas por variable",
        "⏱️ Evolución temporal",
        "💡 Recomendaciones",
    ]
)


# ------------------------------ RESUMEN ------------------------------------

with tab_resumen:
    st.subheader("Variables con mayor cambio en su distribución")

    top = reporte.sort_values("PSI", ascending=True).tail(15)
    barras = go.Figure(
        go.Bar(
            x=top["PSI"],
            y=top["Variable"],
            orientation="h",
            marker_color=[COLOR_ESTADO[estado] for estado in top["Estado"]],
            text=top["PSI"].round(3),
            textposition="outside",
            hovertemplate="%{y}<br>PSI: %{x:.4f}<extra></extra>",
        )
    )
    barras.add_vline(
        x=umbral_mod,
        line_dash="dash",
        line_color=AMBAR,
        annotation_text="Moderado",
        annotation_position="top",
    )
    barras.add_vline(
        x=umbral_crit,
        line_dash="dash",
        line_color=ROJO,
        annotation_text="Crítico",
        annotation_position="top",
    )
    barras.update_layout(title="Variables ordenadas por PSI")
    st.plotly_chart(
        estilo_plotly(barras, 600),
        use_container_width=True,
    )


# --------------------------- DISTRIBUCIONES --------------------------------

with tab_dist:
    comunes = [
        columna
        for columna in datos_train.columns
        if columna in datos_monitoreo.columns and columna != OBJETIVO
    ]
    orden = [
        variable
        for variable in reporte.sort_values(
            "PSI", ascending=False
        )["Variable"]
        if variable in comunes
    ]

    if not orden:
        st.warning(
            "No hay variables comunes entre los datos históricos y los "
            "datos monitoreados para comparar sus distribuciones."
        )
    else:
        variable = st.selectbox(
            "Variable a inspeccionar (ordenadas por PSI)",
            orden,
        )

        fila = reporte[reporte["Variable"] == variable].iloc[0]
        ref = datos_train[variable].dropna()
        act = datos_monitoreo[variable].dropna()

        st.markdown(
            f"Estado: <span class='pill' "
            f"style='--acento:{COLOR_ESTADO[fila['Estado']]}'>"
            f"{fila['Estado']}</span>",
            unsafe_allow_html=True,
        )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("PSI", f"{fila['PSI']:.4f}")
        m2.metric(
            "Jensen-Shannon",
            f"{fila['Jensen_Shannon']:.4f}",
        )
        if pd.notna(fila["KS"]):
            m3.metric("Estadístico KS", f"{fila['KS']:.4f}")
        else:
            m3.metric(
                "Chi-cuadrado",
                f"{fila['Chi_Cuadrado']:.3f}",
            )
        m4.metric("Valor p", f"{fila['Valor_p']:.4f}")

        if ref.empty or act.empty:
            st.warning(
                "No hay suficientes valores válidos para comparar "
                "esta variable."
            )
        elif es_binaria(ref):
            comp = pd.DataFrame(
                {
                    "Valor": ["0", "1"] * 2,
                    "Conjunto": ["Histórico"] * 2 + ["Actual"] * 2,
                    "Porcentaje": list(
                        ref.value_counts(normalize=True).reindex(
                            [0, 1], fill_value=0
                        ) * 100
                    )
                    + list(
                        act.value_counts(normalize=True).reindex(
                            [0, 1], fill_value=0
                        ) * 100
                    ),
                }
            )
            fig = px.bar(
                comp,
                x="Valor",
                y="Porcentaje",
                color="Conjunto",
                barmode="group",
                color_discrete_map={
                    "Histórico": COLOR_HIST,
                    "Actual": COLOR_ACTUAL,
                },
                text_auto=".1f",
                title=f"{variable}: distribución histórica vs. actual",
            )
            st.plotly_chart(
                estilo_plotly(fig, 420),
                use_container_width=True,
            )
        else:
            lo = float(min(ref.min(), act.min()))
            hi = float(max(ref.max(), act.max()))

            if lo == hi:
                st.info(
                    "La variable tiene un único valor en ambos conjuntos."
                )
            else:
                bins = 25
                fig = go.Figure()
                for nombre, serie, color in [
                    ("Histórico", ref, COLOR_HIST),
                    ("Actual", act, COLOR_ACTUAL),
                ]:
                    fig.add_trace(
                        go.Histogram(
                            x=serie,
                            name=nombre,
                            opacity=0.6,
                            marker_color=color,
                            histnorm="percent",
                            nbinsx=bins,
                            xbins=dict(
                                start=lo,
                                end=hi,
                                size=(hi - lo) / bins,
                            ),
                        )
                    )
                fig.update_layout(
                    barmode="overlay",
                    xaxis_range=[lo, hi],
                    title=f"{variable}: distribución histórica vs. actual",
                    yaxis_title="% de registros",
                )
                st.plotly_chart(
                    estilo_plotly(fig, 420),
                    use_container_width=True,
                )

                box = go.Figure()
                box.add_trace(
                    go.Box(
                        x=ref,
                        name="Histórico",
                        marker_color=COLOR_HIST,
                        boxmean=True,
                    )
                )
                box.add_trace(
                    go.Box(
                        x=act,
                        name="Actual",
                        marker_color=COLOR_ACTUAL,
                        boxmean=True,
                    )
                )
                box.update_layout(
                    xaxis_range=[lo, hi],
                    title="Boxplot comparativo",
                    showlegend=False,
                )
                st.plotly_chart(
                    estilo_plotly(box, 260),
                    use_container_width=True,
                )


# ------------------------------ MÉTRICAS -----------------------------------

with tab_metricas:
    st.markdown(
        f"Mostrando **{len(reporte_f)}** de {n_total} variables "
        "según los filtros de la barra lateral."
    )

    columnas_tabla = [
        "Variable",
        "Tipo",
        "Estado",
        "PSI",
        "Jensen_Shannon",
        "KS",
        "Chi_Cuadrado",
        "Valor_p",
        "Significativo",
    ]
    tabla = reporte_f[columnas_tabla].copy()
    tabla["Estado"] = tabla["Estado"].map(
        lambda estado: f"{ICONO_ESTADO[estado]} {estado}"
    )

    st.dataframe(
        tabla,
        use_container_width=True,
        hide_index=True,
        height=480,
        column_config={
            "PSI": st.column_config.NumberColumn("PSI", format="%.4f"),
            "Jensen_Shannon": st.column_config.NumberColumn(
                "Jensen-Shannon", format="%.4f"
            ),
            "KS": st.column_config.NumberColumn("KS", format="%.4f"),
            "Chi_Cuadrado": st.column_config.NumberColumn(
                "Chi²", format="%.3f"
            ),
            "Valor_p": st.column_config.NumberColumn(
                "Valor p", format="%.4f"
            ),
            "Significativo": st.column_config.CheckboxColumn(
                "Significativo"
            ),
        },
    )


# ------------------------------ TEMPORAL -----------------------------------

with tab_tiempo:
    if historial.empty or historial["Fecha_monitoreo"].nunique() < 2:
        st.info(
            "Aún no hay suficiente historial para mostrar una evolución. "
            "Cada ejecución de `mlops_pipeline.model_monitoring` agrega "
            "un registro; se necesitan al menos dos ejecuciones."
        )
        if not historial.empty:
            st.caption("Hay 1 ejecución registrada hasta ahora.")
    else:
        resumen_t = (
            historial.groupby("Fecha_monitoreo")["PSI"]
            .agg(PSI_maximo="max", PSI_promedio="mean")
            .reset_index()
        )
        resumen_t["Variacion"] = resumen_t["PSI_maximo"].diff().abs()

        linea = go.Figure()
        linea.add_trace(
            go.Scatter(
                x=resumen_t["Fecha_monitoreo"],
                y=resumen_t["PSI_maximo"],
                mode="lines+markers",
                name="PSI máximo",
                line=dict(color=ROJO, width=3),
            )
        )
        linea.add_trace(
            go.Scatter(
                x=resumen_t["Fecha_monitoreo"],
                y=resumen_t["PSI_promedio"],
                mode="lines+markers",
                name="PSI promedio",
                line=dict(color=TEAL, width=3),
            )
        )
        linea.add_hline(
            y=umbral_mod,
            line_dash="dash",
            line_color=AMBAR,
        )
        linea.add_hline(
            y=umbral_crit,
            line_dash="dash",
            line_color=ROJO,
        )
        linea.update_layout(title="Evolución del PSI en el tiempo")
        st.plotly_chart(
            estilo_plotly(linea, 380),
            use_container_width=True,
        )

        cambios = resumen_t[resumen_t["Variacion"] > 0.05]
        if not cambios.empty:
            st.warning(
                f"Se detectaron **{len(cambios)}** cambio(s) abrupto(s) "
                "del PSI máximo (variación mayor a 0.05 entre ejecuciones "
                "consecutivas)."
            )

        variables_h = sorted(historial["Variable"].dropna().unique())
        variables_default = [
            variable
            for variable in reporte.sort_values(
                "PSI", ascending=False
            )["Variable"].head(4)
            if variable in variables_h
        ]
        elegidas = st.multiselect(
            "Consultar evolución por variable",
            variables_h,
            default=variables_default,
        )

        if elegidas:
            sub = historial[
                historial["Variable"].isin(elegidas)
            ].sort_values("Fecha_monitoreo")
            fig = px.line(
                sub,
                x="Fecha_monitoreo",
                y="PSI",
                color="Variable",
                markers=True,
                title="Evolución del PSI por variable",
            )
            st.plotly_chart(
                estilo_plotly(fig, 380),
                use_container_width=True,
            )


# ---------------------------- RECOMENDACIONES ------------------------------

with tab_reco:
    st.subheader("Acciones sugeridas")

    def mostrar_recomendacion(titulo, texto, estado):
        st.markdown(
            f"""
            <div class="reco" style="--acento:{COLOR_ESTADO[estado]}">
                <b>{ICONO_ESTADO[estado]} {titulo}</b>{texto}
            </div>
            """,
            unsafe_allow_html=True,
        )

    criticas = reporte[
        reporte["Estado"] == "Crítico"
    ].sort_values("PSI", ascending=False)
    moderadas = reporte[
        reporte["Estado"] == "Moderado"
    ].sort_values("PSI", ascending=False)

    if not criticas.empty:
        lista = ", ".join(
            f"`{variable}`" for variable in criticas["Variable"].head(6)
        )
        mostrar_recomendacion(
            "Investigar variables con cambio crítico",
            f"{lista}. Verifica cambios en la fuente de datos, en la "
            "política de otorgamiento o en el perfil de los clientes, "
            "y evalúa un reentrenamiento del modelo.",
            "Crítico",
        )

    if not moderadas.empty:
        lista = ", ".join(
            f"`{variable}`"
            for variable in moderadas["Variable"].head(6)
        )
        mostrar_recomendacion(
            "Mantener seguimiento",
            f"{lista}. Revisa su comportamiento en las próximas "
            "ejecuciones del monitoreo.",
            "Moderado",
        )

    if criticas.empty and moderadas.empty:
        mostrar_recomendacion(
            "Sin acciones urgentes",
            "Las distribuciones se mantienen estables. "
            "Continúa con el monitoreo periódico.",
            "Estable",
        )

    significativas_estables = reporte[
        (reporte["Significativo"])
        & (reporte["Estado"] == "Estable")
    ]
    if not significativas_estables.empty:
        mostrar_recomendacion(
            "Diferencias significativas de baja magnitud",
            f"{len(significativas_estables)} variable(s) tienen un "
            "valor p bajo, pero un PSI estable. Contrasta la significancia "
            "estadística con la magnitud práctica del cambio.",
            "Moderado",
        )

    if (
        not historial.empty
        and historial["Fecha_monitoreo"].nunique() >= 2
    ):
        if resumen_t["Variacion"].max() > 0.05:
            mostrar_recomendacion(
                "Cambio abrupto en el tiempo",
                "El PSI máximo varió más de 0.05 entre ejecuciones "
                "consecutivas. Revisa qué cambió en los datos de entrada "
                "durante ese periodo.",
                "Crítico",
            )

    st.info(
        "Detectar data drift no implica necesariamente una pérdida de "
        "desempeño. Antes de reentrenar, contrasta estas alertas con "
        "resultados reales del modelo cuando estén disponibles."
    )
