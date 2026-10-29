# 1. Importar las librerías.
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 2. Configurar las rutas y el estilo de los gráficos.
carpeta_proyecto = Path(__file__).resolve().parent
carpeta_salida = carpeta_proyecto / "outputs"
carpeta_salida.mkdir(parents=True, exist_ok=True)
carpeta_graficos = carpeta_salida / "graficos_eda"
carpeta_graficos.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")

# 3. Cargar la base limpia creada por cargar_datos.py.
ruta_limpia = carpeta_salida / "BD_creditos_limpia.csv"
if not ruta_limpia.exists():
    raise FileNotFoundError(
        "No existe outputs/BD_creditos_limpia.csv. Ejecuta primero cargar_datos.py."
    )

datos = pd.read_csv(ruta_limpia, low_memory=False)
if "fecha_prestamo" in datos.columns:
    datos["fecha_prestamo"] = pd.to_datetime(datos["fecha_prestamo"], errors="coerce")

print("BASE LIMPIA CARGADA")
print(f"Filas: {datos.shape[0]} | Columnas: {datos.shape[1]}")
print("\nPrimeras cinco filas:")
print(datos.head().to_string())

# 4. Exploración inicial: tipos, dimensiones, faltantes y valores únicos.
exploracion = pd.DataFrame({
    "columna": datos.columns,
    "tipo_dato": [str(datos[columna].dtype) for columna in datos.columns],
    "nulos": [int(datos[columna].isna().sum()) for columna in datos.columns],
    "porcentaje_nulos": [round(float(datos[columna].isna().mean() * 100), 2) for columna in datos.columns],
    "valores_unicos": [int(datos[columna].nunique(dropna=True)) for columna in datos.columns],
})
exploracion.to_csv(carpeta_salida / "exploracion_inicial.csv", index=False, encoding="utf-8-sig")
print("\nTipos de datos:")
print(datos.dtypes.to_string())
print("\nFaltantes por columna:")
print(datos.isna().sum().sort_values(ascending=False).to_string())

# 5. Identificar variables numéricas y categóricas.
# fecha_prestamo se analiza aparte porque es una fecha, no una cantidad.
# La clasificación semántica no se deduce solo del dtype:
# - tipo_credito es nominal politómica aunque llegue como código numérico.
# - tipo_laboral es nominal dicotómica en los datos revisados.
# - tendencia_ingresos debería ser ordinal (Creciente/Estable/Decreciente);
#   cualquier otro valor debe inspeccionarse, no ordenarse automáticamente.
# - Pago_atiempo es el objetivo dicotómico y no un predictor.
if "fecha_prestamo" in datos.columns:
    columnas_fecha = ["fecha_prestamo"]
else:
    columnas_fecha = []

columnas_numericas = datos.select_dtypes(include=["number"]).columns.tolist()
columnas_categoricas = datos.select_dtypes(include=["object", "string", "category", "bool"]).columns.tolist()

# tipo_credito se guarda como código, pero representa categorías nominales.
if "tipo_credito" in columnas_numericas:
    columnas_numericas.remove("tipo_credito")
    columnas_categoricas.append("tipo_credito")

# La variable objetivo se mantiene numérica para resumir su distribución,
# pero se excluye de la lista de predictores cuando se calcula correlación.
print("\nVariables numéricas:")
print(columnas_numericas)
print("\nVariables categóricas:")
print(columnas_categoricas)

# 6. Estadísticas descriptivas para las variables numéricas.
# describe() incluye conteo, media, desviación, mínimo, cuartiles y máximo.
# Las medidas de asimetría y curtosis ayudan a reconocer distribuciones alejadas
# de la normalidad; no son por sí solas una razón para borrar o transformar datos.
if columnas_numericas:
    resumen_numerico = datos[columnas_numericas].describe().T
    resumen_numerico["moda"] = datos[columnas_numericas].mode(dropna=True).iloc[0]
    resumen_numerico["rango"] = resumen_numerico["max"] - resumen_numerico["min"]
    resumen_numerico["IQR"] = resumen_numerico["75%"] - resumen_numerico["25%"]
    resumen_numerico["varianza"] = datos[columnas_numericas].var()
    resumen_numerico["asimetria_skewness"] = datos[columnas_numericas].skew()
    resumen_numerico["curtosis_kurtosis"] = datos[columnas_numericas].kurtosis()
    resumen_numerico.to_csv(carpeta_salida / "estadisticas_numericas.csv", encoding="utf-8-sig")
    print("\nEstadísticas numéricas guardadas en estadisticas_numericas.csv")

# 7. Estadísticas y frecuencias para las variables categóricas.
# value_counts incluye los faltantes para que también queden documentados.
for columna in columnas_categoricas:
    frecuencias = datos[columna].value_counts(dropna=False).rename_axis("categoria").reset_index(name="frecuencia")
    frecuencias["porcentaje"] = (frecuencias["frecuencia"] / len(datos) * 100).round(2)
    nombre_archivo = f"frecuencias_{columna}.csv"
    frecuencias.to_csv(carpeta_salida / nombre_archivo, index=False, encoding="utf-8-sig")

# 8. Gráficos univariables de variables numéricas: histograma y boxplot.
# Se genera un par de gráficos por columna numérica para revisar distribución,
# asimetría, dispersión y posibles valores extremos.
# En variables monetarias, escalas y límites deben interpretarse con la definición
# y moneda confirmadas; no se infieren rangos válidos solo desde los gráficos.
for columna in columnas_numericas:
    valores = datos[columna].dropna()
    if len(valores) == 0:
        continue

    plt.figure(figsize=(8, 4))
    sns.histplot(valores, bins="auto", kde=True, color="C0")
    plt.title(f"Histograma de {columna}")
    plt.xlabel(columna)
    plt.ylabel("Frecuencia")
    plt.tight_layout()
    plt.savefig(carpeta_graficos / f"histograma_{columna}.png", dpi=130)
    plt.close()

    plt.figure(figsize=(8, 3))
    sns.boxplot(x=valores, color="C0")
    plt.title(f"Boxplot de {columna}")
    plt.xlabel(columna)
    plt.tight_layout()
    plt.savefig(carpeta_graficos / f"boxplot_{columna}.png", dpi=130)
    plt.close()

# 9. Gráficos univariables de variables categóricas: barras de frecuencia.
for columna in columnas_categoricas:
    # Para evitar gráficos ilegibles, se grafican las 30 categorías más frecuentes.
    frecuencias = datos[columna].astype("string").fillna("<FALTANTE>").value_counts().head(30)
    plt.figure(figsize=(9, max(4, len(frecuencias) * 0.3)))
    sns.barplot(x=frecuencias.values, y=frecuencias.index.astype(str), color="C0")
    plt.title(f"Frecuencia de {columna} (máximo 30 categorías)")
    plt.xlabel("Frecuencia")
    plt.ylabel(columna)
    plt.tight_layout()
    plt.savefig(carpeta_graficos / f"frecuencia_{columna}.png", dpi=130)
    plt.close()

# 10. Analizar la variable objetivo Pago_atiempo.
if "Pago_atiempo" not in datos.columns:
    raise ValueError("No se encontró la variable objetivo Pago_atiempo.")

conteo_objetivo = datos["Pago_atiempo"].value_counts(dropna=False).rename_axis("Pago_atiempo").reset_index(name="frecuencia")
conteo_objetivo["porcentaje"] = (conteo_objetivo["frecuencia"] / len(datos) * 100).round(2)
conteo_objetivo.to_csv(carpeta_salida / "distribucion_Pago_atiempo.csv", index=False, encoding="utf-8-sig")

plt.figure(figsize=(6, 4))
sns.countplot(data=datos, x="Pago_atiempo", color="C0")
plt.title("Distribución de la variable objetivo Pago_atiempo")
plt.xlabel("Pago a tiempo (0/1)")
plt.ylabel("Cantidad de registros")
plt.tight_layout()
plt.savefig(carpeta_graficos / "distribucion_Pago_atiempo.png", dpi=130)
plt.close()

# 11. Análisis bivariable: variables numéricas comparadas con Pago_atiempo.
# Se guardan medias y medianas por clase y se generan boxplots comparativos.
columnas_para_comparar = [columna for columna in columnas_numericas if columna != "Pago_atiempo"]
if columnas_para_comparar:
    resumen_por_objetivo = datos.groupby("Pago_atiempo")[columnas_para_comparar].agg(["mean", "median", "std"])
    resumen_por_objetivo.to_csv(carpeta_salida / "resumen_numericas_por_Pago_atiempo.csv", encoding="utf-8-sig")

    for columna in columnas_para_comparar:
        plt.figure(figsize=(7, 4))
        sns.boxplot(data=datos, x="Pago_atiempo", y=columna)
        plt.title(f"{columna} según Pago_atiempo")
        plt.xlabel("Pago a tiempo (0/1)")
        plt.ylabel(columna)
        plt.tight_layout()
        plt.savefig(carpeta_graficos / f"{columna}_por_Pago_atiempo.png", dpi=130)
        plt.close()

# 12. Análisis bivariable: variables categóricas comparadas con Pago_atiempo.
# Las tablas cruzadas permiten comparar porcentajes de pago a tiempo por grupo.
# Se crea una categoría explícita de faltantes para que los porcentajes incluyan
# también observaciones incompletas, sin presentarlas como una categoría real.
for columna in columnas_categoricas:
    if columna == "Pago_atiempo":
        continue
    categoria = datos[columna].astype("string").fillna("<FALTANTE>")
    tabla = pd.crosstab(categoria, datos["Pago_atiempo"], normalize="index") * 100
    tabla.index.name = columna
    tabla.to_csv(carpeta_salida / f"tabla_cruzada_{columna}_Pago_atiempo.csv", encoding="utf-8-sig")

    # Se dibuja el gráfico cuando la variable tiene una cantidad manejable de categorías.
    if datos[columna].nunique(dropna=True) <= 20:
        plt.figure(figsize=(9, 4))
        sns.countplot(data=datos, x=columna, hue="Pago_atiempo", palette=["C0", "C1"])
        plt.title(f"{columna} comparada con Pago_atiempo")
        plt.xlabel(columna)
        plt.ylabel("Cantidad de registros")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(carpeta_graficos / f"{columna}_por_Pago_atiempo.png", dpi=130)
        plt.close()

# 13. Preparar variables categóricas para el análisis de correlación.
# One-Hot Encoding convierte cada categoría nominal en una columna 0/1.
# drop_first=False conserva una columna por categoría para facilitar la lectura.
# Los valores faltantes se mantienen como una categoría explícita en este archivo
# de análisis, para no perder registros solo al calcular correlaciones.
datos_correlacion = datos.copy()
# tipo_credito es un código numérico, pero representa una categoría nominal;
# se convierte a texto para que One-Hot Encoding lo trate como categoría.
if "tipo_credito" in datos_correlacion.columns:
    datos_correlacion["tipo_credito"] = datos_correlacion["tipo_credito"].astype("string")

for columna in columnas_fecha:
    # La fecha no se convierte directamente a número porque el número resultante
    # sería difícil de interpretar. Se crean año, mes y día como variables simples.
    datos_correlacion[f"{columna}_anio"] = datos_correlacion[columna].dt.year
    datos_correlacion[f"{columna}_mes"] = datos_correlacion[columna].dt.month
    datos_correlacion[f"{columna}_dia"] = datos_correlacion[columna].dt.day
    datos_correlacion = datos_correlacion.drop(columns=[columna])

categoricas_para_one_hot = datos_correlacion.select_dtypes(include=["object", "string", "category", "bool"]).columns.tolist()
for columna in categoricas_para_one_hot:
    datos_correlacion[columna] = datos_correlacion[columna].astype("string").fillna("<FALTANTE>")

datos_codificados = pd.get_dummies(datos_correlacion, columns=categoricas_para_one_hot, dtype=int)
datos_codificados.to_csv(carpeta_salida / "datos_codificados_one_hot.csv", index=False, encoding="utf-8-sig")

# 14. Calcular la matriz de correlación DESPUÉS de la limpieza y codificación.
# Se usa correlación de Pearson para columnas numéricas. Una correlación alta
# indica asociación lineal, no demuestra causalidad ni basta por sí sola para
# eliminar una variable. La correlación de puntaje con Pago_atiempo (~0.926 en
# los datos revisados) exige confirmar cuándo y cómo se calcula puntaje: si usa
# información posterior al otorgamiento del crédito, podría haber data leakage.
matriz_correlacion = datos_codificados.corr(numeric_only=True)
matriz_correlacion.to_csv(carpeta_salida / "matriz_correlacion.csv", encoding="utf-8-sig")

plt.figure(figsize=(18, 14))
sns.heatmap(matriz_correlacion, cmap="coolwarm", center=0, annot=False)
plt.title("Matriz de correlación después de limpieza y One-Hot Encoding")
plt.tight_layout()
plt.savefig(carpeta_graficos / "matriz_correlacion.png", dpi=150)
plt.close()

# 15. Listar pares de variables con correlación alta para revisarlos.
# El umbral 0.90 es una alerta de posible redundancia, no una instrucción para borrar.
correlacion_objetivo = matriz_correlacion.abs()
pares_altos = []
columnas_matriz = correlacion_objetivo.columns.tolist()
for i in range(len(columnas_matriz)):
    for j in range(i + 1, len(columnas_matriz)):
        valor = correlacion_objetivo.iloc[i, j]
        if pd.notna(valor) and valor >= 0.90:
            pares_altos.append({
                "variable_1": columnas_matriz[i],
                "variable_2": columnas_matriz[j],
                "correlacion_pearson": matriz_correlacion.iloc[i, j],
                "correlacion_absoluta": valor,
                "incluye_variable_objetivo": columnas_matriz[i] == "Pago_atiempo" or columnas_matriz[j] == "Pago_atiempo",
            })

pd.DataFrame(pares_altos).to_csv(carpeta_salida / "pares_correlacion_mayor_090.csv", index=False, encoding="utf-8-sig")

# 16. Reporte de valores atípicos usando IQR y Z-score.
# Se reportan cantidades por variable, pero no se eliminan automáticamente.
# Cuando IQR=0 (por ejemplo, variables con muchos ceros o puntajes concentrados),
# la regla puede señalar muchos valores válidos; esos casos requieren criterio
# del negocio y revisión individual antes de cualquier transformación o descarte.
filas_outliers = []
for columna in columnas_numericas:
    serie = pd.to_numeric(datos[columna], errors="coerce").dropna()
    if serie.empty:
        continue
    q1 = serie.quantile(0.25)
    q3 = serie.quantile(0.75)
    iqr = q3 - q1
    limite_inferior = q1 - 1.5 * iqr
    limite_superior = q3 + 1.5 * iqr
    cantidad_iqr = int(((serie < limite_inferior) | (serie > limite_superior)).sum())
    desviacion = serie.std(ddof=0)
    if desviacion == 0 or pd.isna(desviacion):
        cantidad_z = 0
    else:
        z = (serie - serie.mean()) / desviacion
        cantidad_z = int((z.abs() > 3).sum())
    filas_outliers.append({
        "variable": columna,
        "Q1": q1,
        "Q3": q3,
        "IQR": iqr,
        "limite_inferior_IQR": limite_inferior,
        "limite_superior_IQR": limite_superior,
        "atipicos_IQR": cantidad_iqr,
        "atipicos_Z_mayor_3": cantidad_z,
        "nota": "Candidatos para revisión; no significa que sean errores.",
    })

pd.DataFrame(filas_outliers).to_csv(carpeta_salida / "reporte_outliers.csv", index=False, encoding="utf-8-sig")

# 17. Elaborar un resumen interpretativo basado en los datos de esta ejecución.
# El informe sintetiza hallazgos y decisiones pendientes; no reemplaza la revisión
# de las tablas/gráficos ni convierte asociaciones estadísticas en causalidad.
total_faltantes = int(datos.isna().sum().sum())
faltantes_prioritarios = datos.isna().sum().sort_values(ascending=False)
faltantes_prioritarios = faltantes_prioritarios[faltantes_prioritarios > 0].head(5)
porcentajes_objetivo = datos["Pago_atiempo"].value_counts(normalize=True, dropna=False) * 100
conteos_objetivo = datos["Pago_atiempo"].value_counts(dropna=False)
porcentaje_positivos = float(porcentajes_objetivo.get(1, 0))
porcentaje_negativos = float(porcentajes_objetivo.get(0, 0))
fecha_minima = datos["fecha_prestamo"].min() if "fecha_prestamo" in datos.columns else pd.NaT
fecha_maxima = datos["fecha_prestamo"].max() if "fecha_prestamo" in datos.columns else pd.NaT
variables_numericas_predictoras = [
    columna for columna in columnas_numericas if columna != "Pago_atiempo"
]
asimetria = datos[variables_numericas_predictoras].skew().dropna()
variables_mas_asimetricas = asimetria.reindex(asimetria.abs().sort_values(ascending=False).head(5).index)
correlaciones_objetivo = (
    matriz_correlacion["Pago_atiempo"].drop(labels=["Pago_atiempo"], errors="ignore")
    if "Pago_atiempo" in matriz_correlacion.columns
    else pd.Series(dtype=float)
)
correlaciones_objetivo = correlaciones_objetivo.dropna()
correlaciones_objetivo = correlaciones_objetivo.reindex(
    correlaciones_objetivo.abs().sort_values(ascending=False).head(5).index
)
filas_duplicadas = int(datos.duplicated().sum())

lineas_faltantes = [
    f"- {columna}: {int(cantidad):,} ({datos[columna].isna().mean() * 100:.2f} %)"
    for columna, cantidad in faltantes_prioritarios.items()
]
lineas_asimetria = [
    f"- {columna}: skewness = {valor:.2f}"
    for columna, valor in variables_mas_asimetricas.items()
]
lineas_correlacion = [
    f"- {columna}: r = {valor:.3f}"
    for columna, valor in correlaciones_objetivo.items()
]

lineas_tendencia = []
if "tendencia_ingresos" in datos.columns:
    categorias_tendencia_esperadas = {"Creciente", "Estable", "Decreciente"}
    tendencia_no_clasificada = datos["tendencia_ingresos"].dropna().astype("string").str.strip()
    tendencia_no_clasificada = tendencia_no_clasificada[
        ~tendencia_no_clasificada.isin(categorias_tendencia_esperadas)
    ]
    lineas_tendencia = [
        "",
        "Calidad de tendencia_ingresos:",
        (
            f"- Se encontraron {len(tendencia_no_clasificada):,} valores no clasificados "
            f"como Creciente/Estable/Decreciente en la definición esperada."
        ),
    ]
    if not tendencia_no_clasificada.empty:
        ejemplos_tendencia = ", ".join(
            repr(str(valor))
            for valor in tendencia_no_clasificada.value_counts().head(8).index
        )
        lineas_tendencia.append(f"- Ejemplos para investigar con la fuente: {ejemplos_tendencia}.")

texto_resumen = [
    "ANÁLISIS EXPLORATORIO DE DATOS (EDA)",
    "====================================",
    "",
    "1. Alcance y calidad general",
    f"- Registros analizados: {len(datos):,}; columnas: {datos.shape[1]}.",
    f"- Valores faltantes: {total_faltantes:,}; filas duplicadas completas: {filas_duplicadas:,}.",
    (
        f"- Periodo de fecha_prestamo: {fecha_minima} a {fecha_maxima}."
        if pd.notna(fecha_minima) and pd.notna(fecha_maxima)
        else "- No fue posible determinar el periodo de fecha_prestamo."
    ),
    "- Los conteos de faltantes describen el conjunto ya limpiado; revisar bitacora_limpieza.json para las reglas aplicadas.",
    "",
    "2. Esquema y naturaleza de las variables",
    "- Numéricas continuas candidatas: montos, saldos, salarios e indicadores de puntaje; confirmar unidades y definiciones.",
    "- Numéricas discretas candidatas: edad, plazo, cantidad de créditos y consultas.",
    "- tipo_credito es una variable nominal politómica aunque esté almacenada como número.",
    "- tipo_laboral es nominal (dicotómica en los datos revisados).",
    "- tendencia_ingresos se espera ordinal (Decreciente, Estable, Creciente); validar niveles y orden con el diccionario de datos.",
    "- Pago_atiempo es la variable objetivo dicotómica (0/1), no un predictor.",
    "",
    "3. Faltantes",
    *(lineas_faltantes if lineas_faltantes else ["- No se encontraron valores faltantes."]),
    "- Los faltantes de tendencia_ingresos y promedio_ingresos_datacredito son suficientemente frecuentes para estudiar su patrón.",
    "- En tablas cruzadas se representa el faltante como <FALTANTE>, no como una categoría real del negocio.",
    "",
    "4. Variable objetivo y análisis bivariable",
    f"- Pago_atiempo=1: {int(conteos_objetivo.get(1, 0)):,} registros ({porcentaje_positivos:.2f} %).",
    f"- Pago_atiempo=0: {int(conteos_objetivo.get(0, 0)):,} registros ({porcentaje_negativos:.2f} %).",
    (
        "- La distribución está desbalanceada; para la siguiente etapa evaluar métricas por clase "
        "(recall, precision, F1 y matriz de confusión), no solo accuracy."
        if porcentaje_negativos < 20
        else "- Comparar métricas por clase al evaluar modelos."
    ),
    "- Las tablas y gráficos por objetivo son asociaciones descriptivas; tamaños de grupo distintos pueden afectar la comparación.",
    "",
    "5. Distribuciones y valores extremos",
    *(
        ["- Variables con mayor asimetría absoluta (skewness):", *lineas_asimetria]
        if lineas_asimetria
        else ["- No hubo suficientes observaciones para calcular asimetría."]
    ),
    "- Las distribuciones asimétricas pueden justificar probar transformaciones (por ejemplo log1p para valores no negativos),",
    "  pero solo después de validar rangos, ceros, unidades y comportamiento en datos de entrenamiento.",
    "- El reporte IQR/Z-score identifica candidatos a revisión, no errores. Con IQR=0 y variables de muchos ceros",
    "  puede marcar valores legítimos; no eliminar observaciones automáticamente.",
    "- No se establecen topes monetarios: primero se deben confirmar moneda, unidades y reglas del proveedor/negocio.",
    "",
    "6. Relaciones entre variables y riesgo de fuga",
    *(
        ["- Correlaciones lineales más fuertes con Pago_atiempo:", *lineas_correlacion]
        if lineas_correlacion
        else ["- No se pudo calcular la correlación con Pago_atiempo."]
    ),
    (
        f"- puntaje tiene correlación de Pearson r={matriz_correlacion.loc['puntaje', 'Pago_atiempo']:.3f} "
        "con el objetivo. La asociación no demuestra causalidad."
        if {"puntaje", "Pago_atiempo"}.issubset(matriz_correlacion.columns)
        else "- Revisar las correlaciones con el objetivo antes de seleccionar predictores."
    ),
    "- Confirmar qué mide puntaje y cuándo está disponible. Si usa pagos posteriores al otorgamiento del crédito, excluirlo",
    "  o reconstruirlo sin información futura para evitar data leakage.",
    "- Correlaciones altas entre indicadores one-hot de categorías pueden deberse a categorías mutuamente excluyentes;",
    "  no implican por sí solas variables duplicadas ni justifican eliminarlas.",
    *lineas_tendencia,
    "",
    "7. Reglas y siguientes pasos",
    "- Mantener como controles las reglas documentadas en carga_datos.py: edad fuera del rango acordado, puntaje negativo",
    "  y puntaje_datacredito <150. Conservar >950 hasta contar con evidencia de que sea inválido.",
    "- Investigar categorías inesperadas de tendencia_ingresos y confirmar la semántica/timing de puntaje.",
    "- Registrar conversiones fallidas por separado de los nulos originales antes de fijar validaciones definitivas.",
    "- Evaluar variables derivadas solo con definiciones confirmadas; posibles candidatas: año/mes del préstamo,",
    "  cuota relativa al ingreso si unidades y moneda coinciden, y suma sectorial si no duplica otra medida.",
    "- Las transformaciones, imputaciones y codificaciones para modelado deben ajustarse dentro del conjunto de entrenamiento",
    "  para evitar fuga de información.",
    "",
    "Conclusión: estos resultados orientan controles y preguntas de negocio; no sustituyen la validación con el diccionario",
    "de datos ni autorizan eliminar variables o registros sin revisar su significado.",
]
(carpeta_salida / "resumen_eda.txt").write_text("\n".join(texto_resumen), encoding="utf-8")

# 18. Mostrar dónde quedaron los resultados.
print("\nEDA terminado.")
print(f"Resultados guardados en: {carpeta_salida}")
print("Archivos principales: estadisticas_numericas.csv, distribucion_Pago_atiempo.csv,")
print("matriz_correlacion.csv, pares_correlacion_mayor_090.csv, reporte_outliers.csv y graficos_eda/.")
