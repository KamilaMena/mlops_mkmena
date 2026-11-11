# Predicción del riesgo de incumplimiento crediticio

**Proyecto académico de Machine Learning | Machine Learning Operations (MLOps)**

Este proyecto desarrolla un flujo de Machine Learning orientado a identificar posibles incumplimientos crediticios. A partir de información de clientes, créditos e historial financiero, se exploran los datos, se entrenan y evalúan modelos de clasificación y se implementan herramientas para su monitoreo y despliegue.

El repositorio reúne las distintas etapas del proceso, desde la preparación de los datos hasta la disponibilidad del modelo mediante una API y una aplicación interactiva.

---

## 1. Contexto y objetivo del negocio

Una entidad financiera necesita identificar solicitudes de crédito con una mayor probabilidad de incumplimiento, con el propósito de contar con información que apoye sus procesos de evaluación.

El problema se aborda como una tarea de **clasificación binaria**, utilizando información relacionada con el crédito, el cliente y su historial crediticio.

La variable objetivo es `Pago_atiempo`:

| Valor | Interpretación |
|---|---|
| `1` | El cliente pagó a tiempo. |
| `0` | El cliente incumplió. |

Para el entrenamiento y el monitoreo se transforma esta etiqueta, de modo que `1` represente incumplimiento. Esta convención permite interpretar el *recall* como la proporción de incumplimientos reales que logra detectar el modelo.

> **Alcance:** se trata de un prototipo académico. Sus resultados no deben utilizarse directamente para tomar decisiones crediticias reales sin validación del negocio, evaluación de posibles sesgos y cumplimiento de los controles regulatorios correspondientes.

## 2. Datos y exploración

La fuente de información utilizada es `BD_creditos.xlsx`, con **10.763 registros y 23 columnas** (créditos otorgados entre noviembre de 2024 y abril de 2026). El cargador también contempla la posibilidad de utilizar `Base_de_datos.csv`, si se encuentra disponible.

**Ninguna fila se elimina**: los valores imposibles se convierten en nulos y se imputan después, dentro del pipeline, usando solo datos de entrenamiento. En la fuente original, los casos con `Pago_atiempo = 0` representan aproximadamente el 4,75 % de los registros (511), lo que evidencia un fuerte desbalance entre las clases.

### Principales hallazgos

El análisis exploratorio permitió identificar aspectos relevantes para la preparación y el modelado:

- **Valores faltantes:** se concentran en `tendencia_ingresos` y `promedio_ingresos_datacredito` (~27 %), y faltan con más frecuencia en clientes independientes. También hay faltantes en algunas variables de saldos (1 % a 5 %).
- **Valores imposibles:** 150 edades de 121 a 123 años, 98 salarios en 0 o superiores a 100 millones, 147 puntajes de Datacrédito menores a 150 (códigos de "sin puntaje") y 58 valores numéricos dentro de `tendencia_ingresos`, que debería tener solo las categorías `Decreciente`, `Estable` y `Creciente`.
- **Límites de validación:** los rangos usados (edad entre 18 y 100 años, salario entre 1 y 100 millones, puntaje mínimo de 150) son **supuestos razonables** basados en los datos, pendientes de confirmar con el negocio.
- **Distribuciones:** los montos presentan fuerte asimetría positiva y valores extremos reales, por lo que se transforman con logaritmo en lugar de eliminarse. `saldo_mora` y `saldo_mora_codeudor` son casi todo ceros, y los pocos valores positivos corresponden a clientes en mora, la señal más informativa.
- **Señales débiles pero coherentes:** más consultas, mora, plazos largos y puntaje de Datacrédito bajo se asocian con incumplir. Ninguna variable supera 0,12 de correlación con el objetivo, por lo que se espera un desempeño modesto.
- **Variables correlacionadas:** tres pares de variables superan una correlación de 0,70 y se resuelven excluyendo una de cada par (ver sección 3.1).
- **Posible fuga de información:** la variable `puntaje` presenta una correlación de Pearson alta con el objetivo (`r = 0,926`) y por sí sola separa perfectamente las clases (AUC = 1,0). Aunque esto no demuestra causalidad, su definición y disponibilidad temporal deben revisarse. Por precaución, se excluye del conjunto utilizado para entrenar los modelos.
- **Cambio en el tiempo:** el volumen de créditos cae de forma marcada entre 2025 y 2026, lo que justifica el monitoreo de *data drift*.

Las diferencias observadas entre grupos se interpretan de manera descriptiva y no como evidencia de relaciones causales.

Los análisis y las gráficas se generan desde `mlops_pipeline/eda_analisis.py`. Los resultados gráficos se almacenan en `outputs/graficos_eda/`.

## 3. Flujo de Machine Learning

El proyecto está organizado en módulos independientes que permiten ejecutar cada etapa del proceso desde la raíz del repositorio.

| Etapa | Módulo | Responsabilidad |
|---|---|---|
| Exploración | `eda_analisis.py` | Caracterización, análisis estadístico y visualización de los datos. |
| Ingeniería de características | `ft_engineering.py` | Limpieza, selección de variables, pipeline de transformación y partición estratificada. |
| Compatibilidad | `model_preprocessing.py` | Reexporta los nombres de `ft_engineering.py` para no romper modelos ya guardados. |
| Entrenamiento y evaluación | `model_training_evaluation.py` | Comparación, selección y evaluación de modelos. |
| Monitoreo | `model_monitoring.py` | Predicciones y cálculo de métricas de data drift. |
| Despliegue | `model_deploy.py` | Disponibilidad del modelo mediante una API. |

### 3.1. Ingeniería de características (`ft_engineering.py`)

Este módulo genera las variables del modelo a partir del dataset base y se puede importar sin efectos secundarios (no lee datos ni escribe archivos al importarse), por lo que también lo usan el entrenamiento, el monitoreo y la API. Su implementación se basa en componentes de `scikit-learn`, lo que permite integrar todo el proceso en un pipeline reproducible.

**Flujo de transformación**

1. `dividir_datos()`: separa predictoras y objetivo, y realiza una partición **estratificada** (75 % entrenamiento, 25 % evaluación, `random_state=42`) para conservar la proporción de incumplimientos en ambos conjuntos.
2. `CreditDataCleaner` (paso `limpieza`): aplica las reglas de validación del EDA y elimina las columnas que no entran al modelo.
3. `build_preprocessor()` (paso `preprocesador`): imputación, escalado y codificación diferenciados por tipo de variable.
4. `preparar_dataset_modelado()`: ajusta el pipeline **solo con entrenamiento** y devuelve los conjuntos transformados, listos para modelar.

**Limpieza de datos (`CreditDataCleaner`)**

- Agrupa los tipos de crédito poco frecuentes (6, 7 y 68) bajo la categoría `otros`.
- Convierte en faltantes las edades fuera de 18 a 100 años.
- Convierte en faltantes los salarios menores o iguales a cero y superiores a 100 millones.
- Convierte en faltantes los puntajes de Datacrédito inferiores a 150.
- Convierte en faltantes las categorías de `tendencia_ingresos` que no pertenecen a las opciones esperadas.
- Excluye las variables definidas como no utilizadas en el modelado.
- Valida que los datos de entrada sean un `DataFrame` y contengan las columnas requeridas.

**Transformación de variables (`build_preprocessor`)**

| Tipo de variable | Tratamiento | Motivo |
|---|---|---|
| Monetarias | `log1p`, imputación por mediana y estandarización. | Corregir la fuerte asimetría. |
| Numéricas restantes | Imputación por mediana y estandarización. | Los faltantes son frecuentes y no se pueden borrar filas. |
| Categóricas | Imputación con la categoría `Faltante` y codificación One-Hot. | La ausencia de `tendencia_ingresos` es informativa. |

La codificación utiliza `handle_unknown='ignore'`, lo que permite procesar categorías no observadas durante el entrenamiento sin generar errores. `build_preprocessing_pipeline()` integra la limpieza y la transformación en un único pipeline, de modo que las mismas reglas se aplican durante el entrenamiento, la evaluación y las predicciones sobre nuevos registros.

**Variables excluidas del modelo y por qué**

| Variable excluida | Motivo | Variable que se conserva |
|---|---|---|
| `puntaje` | Posible fuga de datos (r = 0,92 con el objetivo, AUC = 1,0 por sí sola). | — |
| `fecha_prestamo` | El comportamiento cambia con el tiempo; se usa para el análisis temporal y el monitoreo, no como predictor. | — |
| `cuota_pactada` | Redundante: correlación ≈ 0,76 con `capital_prestado`. | `capital_prestado` |
| `saldo_principal` | Redundante: correlación ≈ 0,73 con `saldo_total`. | `saldo_total` |
| `creditos_sectorFinanciero` | Redundante: correlación ≈ 0,79 con `cant_creditosvigentes`. | `cant_creditosvigentes` |

El criterio para las variables redundantes fue un **umbral de correlación de 0,70** (`UMBRAL_CORRELACION`) y, dentro de cada par, conservar la variable más general o de la que depende la otra (la cuota depende del capital; el saldo principal es parte del saldo total; el total de créditos vigentes incluye al sector financiero). Las razones de cada exclusión están documentadas en `RAZONES_EXCLUSION`, dentro de `ft_engineering.py`.

**Variables derivadas:** el EDA propone `tiene_mora`, `falta_ingresos`, `total_creditos_sectores` y `cuota_sobre_salario`. En esta versión **no se incorporaron** al modelo; quedan como mejora futura.

### 3.2. Preparación y entrenamiento

Se comparan cuatro algoritmos de clasificación:

- Regresión Logística.
- Árbol de Decisión.
- Random Forest.
- XGBoost.

La comparación se realiza mediante validación cruzada estratificada. En cada partición, los procesos de limpieza, imputación, transformación, escalado y codificación se ajustan únicamente con los datos de entrenamiento correspondientes, evitando utilizar información de los datos de validación durante el ajuste.

Como resultado se guarda el modelo seleccionado, que incluye el preprocesamiento necesario para recibir datos sin transformar.

### 3.3. Ejecución del flujo

Desde la raíz del proyecto, con el entorno virtual activado, ejecuta los módulos en el siguiente orden:

```bash
# 1. Análisis exploratorio
python -m mlops_pipeline.eda_analisis

# 2. Ingeniería de características (genera las particiones y los datasets transformados)
python -m mlops_pipeline.ft_engineering

# 3. Entrenamiento y evaluación
python -m mlops_pipeline.model_training_evaluation

# 4. Monitoreo y detección de drift
python -m mlops_pipeline.model_monitoring

# 5. Aplicación de monitoreo
streamlit run app.py
```

El análisis exploratorio también puede ejecutarse desde VS Code abriendo `mlops_pipeline/eda_analisis.py` y utilizando **Run All Cells** sobre las celdas `# %%`.

El módulo `cargar_datos.py` se importa desde los demás componentes y no requiere ejecución independiente.

### 3.4. Uso del modelo entrenado

El artefacto principal se encuentra en:

`outputs/models/mejor_modelo.joblib`

Este archivo contiene el proceso completo de limpieza, transformación y clasificación. Para realizar predicciones sobre nuevos registros, se debe proporcionar un `DataFrame` con las variables predictoras requeridas, sin incluir `Pago_atiempo`.

El modelo permite utilizar `predict()` y `predict_proba()`. No es necesario aplicar el preprocesamiento por separado.

## 4. Monitoreo y detección de data drift

El módulo `model_monitoring.py` compara las distribuciones del conjunto de entrenamiento y del conjunto de prueba para identificar posibles cambios en las características de los datos.

Se implementan las siguientes métricas:

| Métrica | Propósito |
|---|---|
| Kolmogorov-Smirnov (KS) | Identificar diferencias entre distribuciones numéricas. |
| Population Stability Index (PSI) | Cuantificar cambios en la distribución de las variables. |
| Jensen-Shannon | Medir la divergencia entre distribuciones. |
| Chi-cuadrado | Evaluar diferencias en variables categóricas. |

Los resultados se almacenan en reportes CSV y se utilizan como fuente para la aplicación de monitoreo.

### Dashboard interactivo

La aplicación desarrollada con **Streamlit** permite consultar los resultados del monitoreo mediante cinco pestañas:

- **Resumen:** estado general del monitoreo y variables con mayor PSI.
- **Distribuciones:** comparación visual (histograma y boxplot) entre los datos históricos y los actuales.
- **Métricas por variable:** tabla con PSI, Jensen-Shannon, KS, chi-cuadrado y valor p.
- **Evolución temporal:** seguimiento del PSI a lo largo de distintas ejecuciones y detección de cambios abruptos.
- **Recomendaciones:** mensajes y sugerencias según los niveles de drift identificados.

El panel incluye indicadores visuales de alerta (semáforo), umbrales de PSI configurables y filtros por tipo de variable, estado y nombre.

**Alcance actual del monitoreo:** la comparación utiliza un conjunto de prueba fijo como muestra actual, obtenido por una partición aleatoria de los mismos datos. No se reciben todavía datos nuevos desde producción ni se programan ejecuciones periódicas, por lo que es esperable que el drift salga bajo. El historial guarda cada ejecución manual; la pestaña temporal necesita al menos dos fechas de monitoreo distintas para mostrar una evolución. Los resultados permiten demostrar el funcionamiento de las métricas y del dashboard, pero no constituyen evidencia de estabilidad del modelo en un entorno productivo. Una mejora natural es comparar los meses antiguos contra los recientes usando `fecha_prestamo`, dado que el EDA mostró un cambio real en el tiempo.

## 5. Resultados de evaluación

En cada ejecución, `model_training_evaluation.py` compara los modelos con validación cruzada estratificada y selecciona uno con base en recall, F1, variabilidad del recall y tiempo de entrenamiento. También evalúa el modelo seleccionado en el conjunto de prueba y genera métricas, matriz de confusión, gráficos comparativos y una justificación.

Los valores dependen de la ejecución y se guardan en `outputs/reports/comparacion_modelos.csv`, `outputs/reports/evaluacion_final.csv` y `outputs/reports/justificacion_modelo.txt`. No se fijan aquí cifras de una ejecución anterior.

## 6. Despliegue mediante API y Docker

El modelo seleccionado se disponibiliza mediante una API desarrollada con **FastAPI**, ubicada en `mlops_pipeline/model_deploy.py`.

| Endpoint | Descripción |
|---|---|
| `GET /health` | Estado del servicio y del modelo cargado. |
| `POST /predict` | Predicción a partir de una lista de registros JSON, enviada directamente o dentro de la clave `records`. |
| `POST /predict/csv` | Predicción por lotes a partir de un archivo CSV; devuelve otro CSV con las predicciones. |

También dispone de una documentación interactiva para consultar y probar sus endpoints.

Detalles del comportamiento:

- La solicitud debe incluir las columnas predictoras requeridas por el pipeline. Los valores faltantes se procesan con las reglas de limpieza e imputación del modelo.
- Cada registro JSON debe ser un objeto; la solicitud debe ser una lista de objetos o un objeto con la clave `records` que contenga esa lista.
- No se debe enviar `Pago_atiempo`: es la variable que el modelo predice.
- La respuesta contiene las predicciones, su clasificación y la probabilidad estimada de incumplimiento.
- La API carga `outputs/models/mejor_modelo.joblib` al iniciar. Esa ruta está definida en el código y no se configura actualmente mediante una variable de entorno.

### Construcción y ejecución

Después de generar el modelo y contar con Docker instalado, construye la imagen desde la raíz del repositorio:

```powershell
docker build -t api-riesgo-crediticio:v2 .
```

Ejecuta el contenedor:

```powershell
docker run --rm -p 8000:8000 api-riesgo-crediticio:v2
```

Una vez iniciado, los servicios quedan disponibles en:

| Recurso | Dirección |
|---|---|
| API | http://localhost:8000 |
| Documentación interactiva | http://localhost:8000/docs |
| Estado del servicio | http://localhost:8000/health |

El `Dockerfile` copia el código de `mlops_pipeline/`, instala `requirements-api.txt` y copia el modelo entrenado. `requirements-api.txt` fija versiones de sus dependencias; `requirements.txt` no fija versiones, por lo que no se garantiza que ambos entornos utilicen las mismas. Si se reentrena el modelo o se cambian dependencias, verifica la compatibilidad del artefacto con las versiones de la imagen y vuelve a construirla. La construcción requiere Docker en ejecución y que exista `outputs/models/mejor_modelo.joblib`.

## 7. Pruebas y calidad del código

Actualmente hay **14 pruebas**, todas en `test/test_model_preprocessing.py`. Validan reglas de limpieza y columnas requeridas, el pipeline de transformación, la partición estratificada, la generación de datasets, la compatibilidad de los nombres reexportados y el guardado/carga del pipeline con `joblib`. No hay actualmente archivos de pruebas para la API, el entrenamiento, el monitoreo ni la aplicación Streamlit.

`pytest` y `pytest-cov` no están declarados en un archivo de dependencias de desarrollo. Instálalos en el entorno activo y ejecuta las pruebas desde la raíz del repositorio:

```bash
python -m pip install pytest pytest-cov
python -m pytest -v
```

El workflow de GitHub Actions ejecuta las pruebas y genera `coverage.xml` con este comando:

```bash
python -m pytest --cov=mlops_pipeline.model_preprocessing --cov-report=xml:coverage.xml
```

La cobertura configurada corresponde únicamente a `mlops_pipeline.model_preprocessing`, que reexporta componentes de `ft_engineering.py`; no mide la cobertura de `ft_engineering.py` ni la del proyecto completo.

### Integración con SonarCloud

El repositorio tiene un workflow de **GitHub Actions** en `.github/workflows/build.yml`. Está configurado para ejecutarse en *push* a las ramas `main` y `developer`, y al abrir o actualizar un *pull request*. El workflow instala las dependencias, ejecuta las pruebas, genera `coverage.xml` y lanza el análisis de SonarCloud. Para que el análisis remoto se autentique, debe estar configurado el secreto `SONAR_TOKEN` en GitHub.

La cobertura que recibe el análisis tiene el alcance limitado descrito arriba; no representa todos los módulos del proyecto.

## 8. Entorno y dependencias

El proyecto utiliza dos archivos de dependencias:

- `requirements.txt`: paquetes necesarios para el flujo de análisis, entrenamiento, monitoreo y visualización.
- `requirements-api.txt`: dependencias fijadas para ejecutar la API dentro de Docker. No incluye `xgboost`; si el modelo guardado requiere una dependencia adicional, esta debe agregarse a este archivo.

Entre las principales librerías utilizadas se encuentran:

`pandas`, `numpy`, `scikit-learn`, `xgboost`, `scipy`, `joblib`, `matplotlib`, `seaborn`, `streamlit` y `plotly`.

También se incluye `ipykernel` para facilitar la ejecución de celdas `# %%` desde VS Code.

### Instalación en Windows

Crea y activa el entorno virtual:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Actualiza pip e instala las dependencias:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Como alternativa, se puede ejecutar `set_up.bat` desde la raíz del proyecto para crear el entorno e instalar `requirements.txt`. Al terminar, activa el entorno en la terminal que vayas a utilizar.

## 9. Resultados y artefactos generados

Los principales resultados del flujo se organizan dentro de `outputs/`:

| Ubicación | Descripción |
|---|---|
| `outputs/graficos_eda/` | Visualizaciones del análisis exploratorio. |
| `outputs/datasets/` | Conjuntos de entrenamiento y evaluación, originales (`*_raw.csv`) y transformados. |
| `outputs/models/mejor_modelo.joblib` | Modelo seleccionado y entrenado. |
| `outputs/models/preprocesador.joblib` | Pipeline de limpieza y transformación ajustado. |
| `outputs/reports/comparacion_modelos.csv` | Resultados de la validación cruzada. |
| `outputs/reports/evaluacion_final.csv` | Métricas del modelo sobre el conjunto de evaluación. |
| `outputs/reports/justificacion_modelo.txt` | Justificación de la selección del modelo final. |
| `outputs/reports/curva_roc.png` | Curva ROC del modelo seleccionado. |
| `outputs/reports/reporte_data_drift.csv` | Métricas de drift de la ejecución más reciente. |
| `outputs/reports/historial_data_drift.csv` | Registro histórico de las ejecuciones de monitoreo. |
| `outputs/reports/datos_monitoreo.csv` | Registros evaluados, predicciones y probabilidades. |

`.gitignore` excluye `.venv/`, `__pycache__/`, `outputs/reports/`, `outputs/graficos_eda/` y `outputs/Contexto/`. No excluye actualmente `.coverage`, `.pytest_cache/` ni `coverage.xml`. `.dockerignore` excluye los datasets, reportes, gráficos y contexto para reducir el contenido de la imagen; el modelo entrenado no está excluido porque el `Dockerfile` lo copia.

## 10. Estructura del repositorio

```text
mlops_mkmena/
├── .github/
│   └── workflows/
│       └── build.yml
├── .gitignore
├── .dockerignore
├── Dockerfile
├── BD_creditos.xlsx
├── app.py
├── config.json
├── requirements.txt
├── requirements-api.txt
├── set_up.bat
├── README.md
├── sonar-project.properties
├── test/
│   └── test_model_preprocessing.py
├── mlops_pipeline/
│   ├── cargar_datos.py
│   ├── eda_analisis.py
│   ├── ft_engineering.py
│   ├── model_deploy.py
│   ├── model_monitoring.py
│   ├── model_preprocessing.py
│   └── model_training_evaluation.py
└── outputs/
    ├── datasets/
    ├── graficos_eda/
    ├── models/
    └── reports/
```

---

**Proyecto final Machine Learning | 2026 - 2**