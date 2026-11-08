"""Panel de monitoreo del modelo de riesgo crediticio (Streamlit).

Ejecutar desde la raíz del repositorio:
    streamlit run app.py

Requiere haber ejecutado antes ft_engineering.py, model_training_evaluation.py
y model_monitoring.py (genera los archivos dentro de outputs/).
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
COLS_MONITOREO_EXTRA = [
    "Incumplimiento_real",
    "Prediccion_incumplimiento",
    "Probabilidad_incumplimiento",
    "Fecha_monitoreo",
]

# Paleta
TINTA = "#16263B"
TEAL = "#0F766E"
VERDE = "#2E8B63"
AMBAR = "#D98E04"
ROJO = "#C8372D"
GRIS = "#8A94A3"
COLOR_HIST = "#5B7DB1"
COLOR_ACTUAL = "#E07A3F"


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

    /* Encabezado */
    .cabecera {{
        background: linear-gradient(120deg, {TINTA} 0%, #1F3A5C 60%, {TEAL} 130%);
        border-radius: 18px;
        padding: 26px 32px;
        margin-bottom: 22px;
        color: #fff;
    }}
    .cabecera h1 {{ color: #fff; font-size: 1.9rem; font-weight: 800; margin: 0 0 4px 0; padding: 0; }}
    .cabecera p  {{ color: rgba(255,255,255,.78); margin: 0; font-size: .98rem; }}
    .cabecera .meta {{ margin-top: 12px; font-size: .82rem; color: rgba(255,255,255,.65); }}

    /* Tarjetas KPI */
    .kpi {{
        border: 1px solid rgba(128,128,128,.25);
        border-left: 6px solid var(--acento, {TEAL});
        border-radius: 14px;
        padding: 14px 18px;
        height: 100%;
        background: rgba(128,128,128,.06);
    }}
    .kpi .titulo {{ font-size: .8rem; opacity: .7; margin-bottom: 4px; }}
    .kpi .valor  {{ font-size: 1.9rem; font-weight: 800; line-height: 1.1; }}
    .kpi .nota   {{ font-size: .78rem; opacity: .65; margin-top: 4px; }}

    /* Banner de estado */
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

    /* Etiquetas */
    .pill {{
        display: inline-block; padding: 3px 12px; border-radius: 999px;
        font-size: .78rem; font-weight: 700; color: #fff; background: var(--acento);
    }}

    /* Recomendaciones */
    .reco {{
        border-radius: 12px; padding: 12px 16px; margin-bottom: 10px;
        border-left: 5px solid var(--acento);
        background: color-mix(in srgb, var(--acento) 10%, transparent);
    }}
    .reco b {{ display: block; margin-bottom: 2px; }}

    /* Pestañas */
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
# 3. Utilidades
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def leer_csv(ruta: str, firma: float) -> pd.DataFrame:
    """Lee un CSV. `firma` (fecha de modificación) invalida la caché si cambia."""
    return pd.read_csv(ruta)


def cargar(ruta: Path, obligatorio: bool = True):
    if not ruta.exists():
        if obligatorio:
            st.error(
                f"No se encontró `{ruta.relative_to(CARPETA_PROYECTO)}`. "
                "Ejecuta primero ft_engineering.py, model_training_evaluation.py "
                "y model_monitoring.py."
            )
            st.stop()
        return None
    return leer_csv(str(ruta), ruta.stat().st_mtime)


def kpi(titulo, valor, nota="", acento=TEAL):
    st.markdown(
        f"""<div class="kpi" style="--acento:{acento}">
        <div class="titulo">{titulo}</div>
        <div class="valor">{valor}</div>
        <div class="nota">{nota}</div></div>""",
        unsafe_allow_html=True,
    )


def estado_psi(valor, umbral_mod, umbral_crit):
    if valor >= umbral_crit:
        return "Crítico"
    if valor >= umbral_mod:
        return "Moderado"
    return "Estable"


COLOR_ESTADO = {"Estable": VERDE, "Moderado": AMBAR, "Crítico": ROJO}
ICONO_ESTADO = {"Estable": "🟢", "Moderado": "🟡", "Crítico": "🔴"}


def es_binaria(serie: pd.Series) -> bool:
    return set(serie.dropna().unique()).issubset({0, 1})


def estilo_plotly(fig, alto=380):
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
historial = cargar(CARPETA_REPORTES / "historial_data_drift.csv", obligatorio=False)
evaluacion = cargar(CARPETA_REPORTES / "evaluacion_final.csv", obligatorio=False)
comparacion = cargar(CARPETA_REPORTES / "comparacion_modelos.csv", obligatorio=False)

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
# 5. Barra lateral: filtros y umbrales
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## ⚙️ Configuración")

    st.markdown("**Umbrales de alerta (PSI)**")
    umbral_mod = st.slider("Cambio moderado desde", 0.05, 0.30, 0.10, 0.01)
    umbral_crit = st.slider("Cambio crítico desde", 0.15, 0.60, 0.25, 0.01)
    if umbral_crit <= umbral_mod:
        st.warning("El umbral crítico debe ser mayor que el moderado.")
        umbral_crit = umbral_mod + 0.05

    alfa = st.select_slider(
        "Nivel de significancia (valor p)", options=[0.01, 0.05, 0.10], value=0.05
    )

    st.divider()
    st.markdown("**Filtros de variables**")
    tipos = sorted(reporte["Tipo"].unique())
    tipos_sel = st.multiselect("Tipo de variable", tipos, default=tipos)
    estados_sel = st.multiselect(
        "Estado",
        ["Estable", "Moderado", "Crítico"],
        default=["Estable", "Moderado", "Crítico"],
    )
    busqueda = st.text_input("Buscar variable", placeholder="p. ej. puntaje")

    st.divider()
    st.caption(f"Último monitoreo: **{fecha_ultima}**")
    if st.button("🔄 Recargar archivos", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# Recalcular estado con los umbrales elegidos
reporte = reporte.copy()
reporte["Estado"] = reporte["PSI"].apply(lambda v: estado_psi(v, umbral_mod, umbral_crit))
reporte["Significativo"] = reporte["Valor_p"] < alfa

reporte_f = reporte[
    reporte["Tipo"].isin(tipos_sel)
    & reporte["Estado"].isin(estados_sel)
    & reporte["Variable"].str.contains(busqueda, case=False, na=False)
]

n_total = len(reporte)
n_crit = int((reporte["Estado"] == "Crítico").sum())
n_mod = int((reporte["Estado"] == "Moderado").sum())
n_est = int((reporte["Estado"] == "Estable").sum())
n_sig = int(reporte["Significativo"].sum())
psi_max = float(reporte["PSI"].max())
var_max = reporte.loc[reporte["PSI"].idxmax(), "Variable"]

if n_crit > 0:
    nivel = "Crítico"
elif n_mod > 0:
    nivel = "Moderado"
else:
    nivel = "Estable"


# ---------------------------------------------------------------------------
# 6. Encabezado y estado general
# ---------------------------------------------------------------------------

st.markdown(
    f"""
    <div class="cabecera">
        <h1>🛡️ Monitoreo del modelo de riesgo crediticio</h1>
        <p>Estabilidad de las variables, calidad de las predicciones y alertas de data drift.</p>
        <div class="meta">Último monitoreo: {fecha_ultima} &nbsp;|&nbsp;
        {len(datos_monitoreo):,} registros monitoreados &nbsp;|&nbsp;
        {n_total} variables analizadas</div>
    </div>
    """,
    unsafe_allow_html=True,
)

mensajes_estado = {
    "Crítico": f"{n_crit} variable(s) superan el umbral crítico de PSI ({umbral_crit:.2f}). Revisa la pestaña Recomendaciones.",
    "Moderado": f"{n_mod} variable(s) muestran cambios moderados. Mantén seguimiento en la próxima ejecución.",
    "Estable": "Todas las variables se mantienen dentro de los umbrales definidos.",
}
st.markdown(
    f"""<div class="estado" style="--acento:{COLOR_ESTADO[nivel]}">
    <div class="punto"></div>
    <div class="texto"><b>Estado general: {nivel}</b>
    <span>{mensajes_estado[nivel]}</span></div></div>""",
    unsafe_allow_html=True,
)

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    kpi("Variables analizadas", n_total, "Todas las features del modelo")
with c2:
    kpi("Estables", n_est, f"PSI < {umbral_mod:.2f}", VERDE)
with c3:
    kpi("Cambio moderado", n_mod, f"PSI {umbral_mod:.2f} – {umbral_crit:.2f}", AMBAR)
with c4:
    kpi("Cambio crítico", n_crit, f"PSI ≥ {umbral_crit:.2f}", ROJO)
with c5:
    kpi("PSI máximo", f"{psi_max:.3f}", f"Variable: {var_max}", TINTA)

st.write("")

# ---------------------------------------------------------------------------
# 7. Pestañas
# ---------------------------------------------------------------------------

tab_resumen, tab_dist, tab_metricas, tab_tiempo, tab_modelo, tab_reco = st.tabs(
    [
        "📊 Resumen",
        "📈 Distribuciones",
        "🧮 Métricas por variable",
        "⏱️ Evolución temporal",
        "🎯 Desempeño del modelo",
        "💡 Recomendaciones",
    ]
)

# ------------------------------ RESUMEN ------------------------------------
with tab_resumen:
    col_a, col_b = st.columns([1, 2])

    with col_a:
        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=psi_max,
                number=dict(valueformat=".3f"),
                title=dict(text="PSI máximo observado"),
                gauge=dict(
                    axis=dict(range=[0, max(0.6, psi_max * 1.2)]),
                    bar=dict(color=TINTA, thickness=0.25),
                    steps=[
                        dict(range=[0, umbral_mod], color="rgba(46,139,99,.45)"),
                        dict(range=[umbral_mod, umbral_crit], color="rgba(217,142,4,.45)"),
                        dict(range=[umbral_crit, max(0.6, psi_max * 1.2)], color="rgba(200,55,45,.45)"),
                    ],
                ),
            )
        )
        st.plotly_chart(estilo_plotly(gauge, 320), use_container_width=True)

        dona = go.Figure(
            go.Pie(
                labels=["Estables", "Moderado", "Crítico"],
                values=[n_est, n_mod, n_crit],
                hole=0.62,
                marker=dict(colors=[VERDE, AMBAR, ROJO]),
                textinfo="value",
                sort=False,
            )
        )
        dona.update_layout(title="Variables por estado", showlegend=True)
        st.plotly_chart(estilo_plotly(dona, 300), use_container_width=True)

    with col_b:
        top = reporte.sort_values("PSI", ascending=True).tail(15)
        barras = go.Figure(
            go.Bar(
                x=top["PSI"],
                y=top["Variable"],
                orientation="h",
                marker_color=[COLOR_ESTADO[e] for e in top["Estado"]],
                text=top["PSI"].round(3),
                textposition="outside",
                hovertemplate="%{y}<br>PSI: %{x:.4f}<extra></extra>",
            )
        )
        barras.add_vline(x=umbral_mod, line_dash="dash", line_color=AMBAR,
                         annotation_text="Moderado", annotation_position="top")
        barras.add_vline(x=umbral_crit, line_dash="dash", line_color=ROJO,
                         annotation_text="Crítico", annotation_position="top")
        barras.update_layout(title="Top 15 variables con mayor PSI")
        st.plotly_chart(estilo_plotly(barras, 650), use_container_width=True)

    if n_sig > 0:
        st.info(
            f"**{n_sig}** variable(s) presentan diferencias estadísticamente "
            f"significativas (valor p < {alfa}). Con muestras grandes, una diferencia "
            "puede ser significativa pero pequeña: contrástala con el PSI antes de actuar."
        )

# --------------------------- DISTRIBUCIONES --------------------------------
with tab_dist:
    comunes = [
        c for c in datos_train.columns
        if c in datos_monitoreo.columns and c != OBJETIVO
    ]
    # Ordenar por PSI para destacar primero las más cambiadas
    orden = [v for v in reporte.sort_values("PSI", ascending=False)["Variable"] if v in comunes]

    cc1, cc2 = st.columns([2, 1])
    with cc1:
        variable = st.selectbox("Variable a inspeccionar (ordenadas por PSI)", orden)
    with cc2:
        bins = st.slider("Intervalos del histograma", 10, 60, 25)

    fila = reporte[reporte["Variable"] == variable].iloc[0]
    ref = datos_train[variable].dropna()
    act = datos_monitoreo[variable].dropna()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("PSI", f"{fila['PSI']:.4f}")
    m2.metric("Jensen-Shannon", f"{fila['Jensen_Shannon']:.4f}")
    if pd.notna(fila["KS"]):
        m3.metric("Estadístico KS", f"{fila['KS']:.4f}")
    else:
        m3.metric("Chi-cuadrado", f"{fila['Chi_Cuadrado']:.3f}")
    m4.metric("Valor p", f"{fila['Valor_p']:.4f}")

    st.markdown(
        f"Estado: <span class='pill' style='--acento:{COLOR_ESTADO[fila['Estado']]}'>"
        f"{fila['Estado']}</span>",
        unsafe_allow_html=True,
    )

    if es_binaria(ref):
        comp = pd.DataFrame(
            {
                "Valor": ["0", "1"] * 2,
                "Conjunto": ["Histórico"] * 2 + ["Actual"] * 2,
                "Porcentaje": list(
                    ref.value_counts(normalize=True).reindex([0, 1], fill_value=0) * 100
                ) + list(
                    act.value_counts(normalize=True).reindex([0, 1], fill_value=0) * 100
                ),
            }
        )
        fig = px.bar(
            comp, x="Valor", y="Porcentaje", color="Conjunto", barmode="group",
            color_discrete_map={"Histórico": COLOR_HIST, "Actual": COLOR_ACTUAL},
            text_auto=".1f", title=f"{variable}: distribución histórica vs. actual",
        )
        st.plotly_chart(estilo_plotly(fig, 420), use_container_width=True)
    else:
        lo = float(min(ref.min(), act.min()))
        hi = float(max(ref.max(), act.max()))
        recortar = st.checkbox("Recortar valores extremos (percentil 1–99)", value=True)
        if recortar:
            lo = float(np.nanpercentile(np.concatenate([ref, act]), 1))
            hi = float(np.nanpercentile(np.concatenate([ref, act]), 99))
        if lo == hi:
            st.info("La variable tiene un único valor en ambos conjuntos.")
        else:
            fig = go.Figure()
            for nombre, serie, color in [
                ("Histórico", ref, COLOR_HIST),
                ("Actual", act, COLOR_ACTUAL),
            ]:
                fig.add_trace(
                    go.Histogram(
                        x=serie, name=nombre, opacity=0.6, marker_color=color,
                        histnorm="percent", nbinsx=bins,
                        xbins=dict(start=lo, end=hi, size=(hi - lo) / bins),
                    )
                )
            fig.update_layout(
                barmode="overlay", xaxis_range=[lo, hi],
                title=f"{variable}: distribución histórica vs. actual",
                yaxis_title="% de registros",
            )
            st.plotly_chart(estilo_plotly(fig, 420), use_container_width=True)

            box = go.Figure()
            box.add_trace(go.Box(x=ref, name="Histórico", marker_color=COLOR_HIST, boxmean=True))
            box.add_trace(go.Box(x=act, name="Actual", marker_color=COLOR_ACTUAL, boxmean=True))
            box.update_layout(xaxis_range=[lo, hi], title="Boxplot comparativo", showlegend=False)
            st.plotly_chart(estilo_plotly(box, 260), use_container_width=True)

# ------------------------------ MÉTRICAS -----------------------------------
with tab_metricas:
    st.markdown(f"Mostrando **{len(reporte_f)}** de {n_total} variables según los filtros de la barra lateral.")

    tabla = reporte_f[
        ["Variable", "Tipo", "Estado", "PSI", "Jensen_Shannon", "KS", "Chi_Cuadrado", "Valor_p", "Significativo"]
    ].copy()
    tabla["Estado"] = tabla["Estado"].map(lambda e: f"{ICONO_ESTADO[e]} {e}")

    st.dataframe(
        tabla,
        use_container_width=True,
        hide_index=True,
        height=480,
        column_config={
            "PSI": st.column_config.ProgressColumn(
                "PSI", format="%.3f", min_value=0, max_value=max(0.5, psi_max)
            ),
            "Jensen_Shannon": st.column_config.NumberColumn("Jensen-Shannon", format="%.4f"),
            "KS": st.column_config.NumberColumn("KS", format="%.4f"),
            "Chi_Cuadrado": st.column_config.NumberColumn("Chi²", format="%.3f"),
            "Valor_p": st.column_config.NumberColumn("Valor p", format="%.4f"),
            "Significativo": st.column_config.CheckboxColumn("Signif."),
        },
    )

    d1, d2 = st.columns([1, 4])
    with d1:
        st.download_button(
            "⬇️ Descargar tabla (CSV)",
            reporte_f.to_csv(index=False).encode("utf-8"),
            file_name="metricas_drift_filtradas.csv",
            mime="text/csv",
            use_container_width=True,
        )

    if len(reporte_f) > 1:
        mapa = reporte_f.set_index("Variable")[["PSI", "Jensen_Shannon"]].sort_values("PSI", ascending=False)
        mapa_norm = mapa / mapa.max().replace(0, 1)
        heat = go.Figure(
            go.Heatmap(
                z=mapa_norm.T.values, x=mapa.index, y=["PSI", "Jensen-Shannon"],
                colorscale=[[0, "#E8F3EE"], [0.5, "#F2C15B"], [1, ROJO]],
                customdata=mapa.T.values,
                hovertemplate="%{x}<br>%{y}: %{customdata:.4f}<extra></extra>",
                showscale=False,
            )
        )
        heat.update_layout(title="Intensidad relativa del drift por variable")
        heat.update_xaxes(tickangle=-45)
        st.plotly_chart(estilo_plotly(heat, 320), use_container_width=True)

# ------------------------------ TEMPORAL -----------------------------------
with tab_tiempo:
    if historial.empty or historial["Fecha_monitoreo"].nunique() < 2:
        st.info(
            "Aún no hay suficiente historial para ver una evolución. "
            "Cada ejecución de `model_monitoring.py` agrega un punto; "
            "se necesitan al menos dos ejecuciones."
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
        linea.add_trace(go.Scatter(
            x=resumen_t["Fecha_monitoreo"], y=resumen_t["PSI_maximo"],
            mode="lines+markers", name="PSI máximo", line=dict(color=ROJO, width=3)))
        linea.add_trace(go.Scatter(
            x=resumen_t["Fecha_monitoreo"], y=resumen_t["PSI_promedio"],
            mode="lines+markers", name="PSI promedio", line=dict(color=TEAL, width=3)))
        linea.add_hline(y=umbral_mod, line_dash="dash", line_color=AMBAR)
        linea.add_hline(y=umbral_crit, line_dash="dash", line_color=ROJO)
        linea.update_layout(title="Evolución del PSI en el tiempo")
        st.plotly_chart(estilo_plotly(linea, 380), use_container_width=True)

        cambios = resumen_t[resumen_t["Variacion"] > 0.05]
        if not cambios.empty:
            st.warning(
                f"Se detectaron **{len(cambios)}** cambio(s) abrupto(s) del PSI máximo "
                "(variación mayor a 0.05 entre ejecuciones consecutivas)."
            )

        variables_h = sorted(historial["Variable"].unique())
        elegidas = st.multiselect(
            "Comparar variables",
            variables_h,
            default=list(reporte.sort_values("PSI", ascending=False)["Variable"].head(4)),
        )
        if elegidas:
            sub = historial[historial["Variable"].isin(elegidas)].sort_values("Fecha_monitoreo")
            fig = px.line(sub, x="Fecha_monitoreo", y="PSI", color="Variable", markers=True,
                          title="PSI por variable")
            st.plotly_chart(estilo_plotly(fig, 380), use_container_width=True)

        pivote = historial.pivot_table(
            index="Variable", columns="Fecha_monitoreo", values="PSI", aggfunc="last"
        )
        pivote = pivote.loc[pivote.max(axis=1).sort_values(ascending=False).head(20).index]
        mapa = px.imshow(
            pivote, aspect="auto", color_continuous_scale=["#E8F3EE", "#F2C15B", ROJO],
            title="Mapa de calor del PSI (top 20 variables)",
        )
        st.plotly_chart(estilo_plotly(mapa, 520), use_container_width=True)

# ------------------------------- MODELO ------------------------------------
with tab_modelo:
    st.caption(
        "Las métricas se calculan con los registros monitoreados. "
        "Incumplimiento = 1 (cliente que no pagó a tiempo)."
    )

    if evaluacion is not None and len(evaluacion):
        ev = evaluacion.iloc[0]
        cols = st.columns(5)
        for col, nombre in zip(cols, ["Recall", "Precision", "F1-Score", "ROC-AUC", "PR-AUC"]):
            if nombre in ev.index:
                col.metric(nombre, f"{ev[nombre]:.3f}")
        st.write("")

    if {"Incumplimiento_real", "Probabilidad_incumplimiento"}.issubset(datos_monitoreo.columns):
        y = datos_monitoreo["Incumplimiento_real"].astype(int).values
        p = datos_monitoreo["Probabilidad_incumplimiento"].values

        st.markdown("#### Simulador de umbral de decisión")
        umbral = st.slider(
            "Probabilidad mínima para marcar un crédito como riesgoso",
            0.05, 0.95, 0.50, 0.01,
        )
        pred = (p >= umbral).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        recall = tp / (tp + fn) if (tp + fn) else 0
        precision = tp / (tp + fp) if (tp + fp) else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            kpi("Recall", f"{recall:.1%}", "Incumplimientos detectados", TEAL)
        with k2:
            kpi("Precisión", f"{precision:.1%}", "Alertas que sí eran incumplimiento", COLOR_HIST)
        with k3:
            kpi("F1", f"{f1:.3f}", "Balance recall / precisión", TINTA)
        with k4:
            kpi("Alertas generadas", f"{int(pred.sum()):,}", f"de {len(pred):,} créditos", AMBAR)

        g1, g2 = st.columns(2)
        with g1:
            hist = go.Figure()
            hist.add_trace(go.Histogram(x=p[y == 0], name="Pagó a tiempo", marker_color=COLOR_HIST,
                                        opacity=0.65, nbinsx=40))
            hist.add_trace(go.Histogram(x=p[y == 1], name="Incumplió", marker_color=ROJO,
                                        opacity=0.75, nbinsx=40))
            hist.add_vline(x=umbral, line_dash="dash", line_color=TINTA,
                           annotation_text=f"Umbral {umbral:.2f}")
            hist.update_layout(barmode="overlay", title="Probabilidad predicha por clase real",
                               xaxis_title="Probabilidad de incumplimiento")
            hist.update_yaxes(type="log", title="Registros (escala log)")
            st.plotly_chart(estilo_plotly(hist, 400), use_container_width=True)
        with g2:
            matriz = np.array([[tn, fp], [fn, tp]])
            cm = go.Figure(
                go.Heatmap(
                    z=matriz, x=["Pred: Pagó", "Pred: Incumple"], y=["Real: Pagó", "Real: Incumplió"],
                    text=matriz, texttemplate="%{text:,}", textfont=dict(size=18),
                    colorscale=[[0, "#EEF3F9"], [1, TEAL]], showscale=False,
                )
            )
            cm.update_layout(title="Matriz de confusión")
            cm.update_yaxes(autorange="reversed")
            st.plotly_chart(estilo_plotly(cm, 400), use_container_width=True)

        # Curva recall / precisión según umbral
        grilla = np.linspace(0.05, 0.95, 46)
        rec_l, pre_l = [], []
        for t in grilla:
            pr = (p >= t).astype(int)
            tp_ = ((pr == 1) & (y == 1)).sum()
            fp_ = ((pr == 1) & (y == 0)).sum()
            fn_ = ((pr == 0) & (y == 1)).sum()
            rec_l.append(tp_ / (tp_ + fn_) if (tp_ + fn_) else 0)
            pre_l.append(tp_ / (tp_ + fp_) if (tp_ + fp_) else np.nan)
        curva = go.Figure()
        curva.add_trace(go.Scatter(x=grilla, y=rec_l, name="Recall", line=dict(color=TEAL, width=3)))
        curva.add_trace(go.Scatter(x=grilla, y=pre_l, name="Precisión", line=dict(color=COLOR_ACTUAL, width=3)))
        curva.add_vline(x=umbral, line_dash="dash", line_color=TINTA)
        curva.update_layout(title="Recall y precisión según el umbral", xaxis_title="Umbral")
        st.plotly_chart(estilo_plotly(curva, 340), use_container_width=True)

    if comparacion is not None:
        st.markdown("#### Comparación de modelos (validación cruzada)")
        cols_ok = [c for c in ["Modelo", "Recall_CV", "Precision_CV", "F1_CV", "PR_AUC_CV",
                               "ROC_AUC_CV", "Tiempo_entrenamiento"] if c in comparacion.columns]
        st.dataframe(
            comparacion[cols_ok], hide_index=True, use_container_width=True,
            column_config={c: st.column_config.NumberColumn(format="%.3f") for c in cols_ok if c != "Modelo"},
        )

    imagenes = [
        ("comparacion_metricas.png", "Comparación de métricas"),
        ("consistencia_modelos.png", "Consistencia del recall"),
        ("curva_roc.png", "Curva ROC del modelo seleccionado"),
    ]
    disponibles = [(CARPETA_REPORTES / f, t) for f, t in imagenes if (CARPETA_REPORTES / f).exists()]
    if disponibles:
        with st.expander("Ver gráficos generados en el entrenamiento"):
            for ruta, titulo in disponibles:
                st.image(str(ruta), caption=titulo)

# ---------------------------- RECOMENDACIONES ------------------------------
with tab_reco:
    st.markdown("#### Acciones sugeridas")

    def reco(titulo, texto, estado):
        st.markdown(
            f"""<div class="reco" style="--acento:{COLOR_ESTADO[estado]}">
            <b>{ICONO_ESTADO[estado]} {titulo}</b>{texto}</div>""",
            unsafe_allow_html=True,
        )

    criticas = reporte[reporte["Estado"] == "Crítico"].sort_values("PSI", ascending=False)
    moderadas = reporte[reporte["Estado"] == "Moderado"].sort_values("PSI", ascending=False)

    if not criticas.empty:
        lista = ", ".join(f"`{v}`" for v in criticas["Variable"].head(6))
        reco(
            "Investigar variables con cambio crítico",
            f"{lista}. Verifica cambios en la fuente de datos, en la política de otorgamiento "
            "o en el perfil de los clientes, y evalúa un reentrenamiento del modelo.",
            "Crítico",
        )
    if not moderadas.empty:
        lista = ", ".join(f"`{v}`" for v in moderadas["Variable"].head(6))
        reco(
            "Mantener seguimiento",
            f"{lista}. Revisa su comportamiento en las próximas ejecuciones del monitoreo.",
            "Moderado",
        )
    if criticas.empty and moderadas.empty:
        reco(
            "Sin acciones urgentes",
            "Las distribuciones se mantienen estables. Continúa con el monitoreo periódico.",
            "Estable",
        )

    sig_pequeno = reporte[(reporte["Significativo"]) & (reporte["Estado"] == "Estable")]
    if not sig_pequeno.empty:
        reco(
            "Diferencias significativas de baja magnitud",
            f"{len(sig_pequeno)} variable(s) tienen valor p bajo pero PSI estable. "
            "Probablemente reflejan el tamaño de la muestra; su impacto práctico es bajo.",
            "Moderado",
        )

    if (
        not historial.empty
        and historial["Fecha_monitoreo"].nunique() >= 2
        and resumen_t["Variacion"].max() > 0.05
    ):
        reco(
            "Cambio abrupto en el tiempo",
            "El PSI máximo varió más de 0.05 entre ejecuciones consecutivas. "
            "Revisa qué cambió en los datos de entrada en ese periodo.",
            "Crítico",
        )

    st.info(
        "Detectar data drift no implica necesariamente una pérdida de desempeño. "
        "Antes de reentrenar, contrasta estas alertas con las métricas del modelo "
        "calculadas sobre resultados reales."
    )

    reporte_txt = (
        f"Reporte de monitoreo ({fecha_ultima})\n"
        f"Estado general: {nivel}\n"
        f"Variables críticas: {', '.join(criticas['Variable']) or 'ninguna'}\n"
        f"Variables moderadas: {', '.join(moderadas['Variable']) or 'ninguna'}\n"
        f"PSI máximo: {psi_max:.4f} ({var_max})\n"
    )
    st.download_button(
        "⬇️ Descargar resumen de alertas", reporte_txt.encode("utf-8"),
        file_name="resumen_alertas.txt", mime="text/plain",
    )