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

La fuente de información utilizada es `BD_creditos.xlsx`, con **10.763 registros y 23 columnas**. El cargador también contempla la posibilidad de utilizar `Base_de_datos.csv`, si se encuentra disponible.

Después de aplicar las reglas de preparación del análisis exploratorio, se reportan 10.613 registros. En la fuente original, los casos con `Pago_atiempo = 0` representan aproximadamente el 4,75 % de los registros, lo que evidencia un desbalance entre las clases.

### Principales hallazgos

El análisis exploratorio permitió identificar aspectos relevantes para la preparación y el modelado:

- **Valores faltantes:** se concentran principalmente en `tendencia_ingresos` y `promedio_ingresos_datacredito`, con alrededor del 26 % de datos faltantes en el conjunto analizado. También se identificaron faltantes en algunas variables de saldos.
- **Distribuciones:** algunas variables monetarias presentan asimetría y valores extremos. No se establecen límites arbitrarios sin confirmar previamente las unidades y reglas de negocio.
- **Consistencia categórica:** se encontraron 58 valores de `tendencia_ingresos` que no corresponden a las categorías esperadas (`Decreciente`, `Estable` y `Creciente`). Estos valores se tratan como faltantes durante la preparación para el modelado.
- **Posible fuga de información:** la variable `puntaje` presenta una correlación de Pearson alta con el objetivo (`r = 0,926`). Aunque esto no demuestra causalidad, su definición y disponibilidad temporal deben revisarse. Por precaución, se excluye del conjunto utilizado para entrenar los modelos.

Las diferencias observadas entre grupos se interpretan de manera descriptiva y no como evidencia de relaciones causales.

Los análisis y las gráficas se generan desde `mlops_pipeline/eda_analisis.py`. Los resultados gráficos se almacenan en `outputs/graficos_eda/`.

## 3. Flujo de Machine Learning

El proyecto está organizado en módulos independientes que permiten ejecutar cada etapa del proceso desde la raíz del repositorio.

| Etapa | Módulo | Responsabilidad |
|---|---|---|
| Exploración | `eda_analisis.py` | Caracterización, análisis estadístico y visualización de los datos. |
| Ingeniería de características | `ft_engineering.py` | Preparación de variables y división estratificada de los datos. |
| Preprocesamiento | `model_preprocessing.py` | Limpieza, validación y transformación de variables. |
| Entrenamiento y evaluación | `model_training_evaluation.py` | Comparación, selección y evaluación de modelos. |
| Monitoreo | `model_monitoring.py` | Predicciones y cálculo de métricas de data drift. |
| Despliegue | `model_deploy.py` | Disponibilidad del modelo mediante una API. |

### 3.1. Limpieza y preprocesamiento de datos

El módulo `mlops_pipeline/model_preprocessing.py` centraliza las reglas de validación, limpieza y transformación de las variables utilizadas por los modelos. Su implementación se basa en componentes de `scikit-learn`, lo que permite integrar todo el proceso en un pipeline reproducible.

La preparación se divide en dos componentes principales:

**Limpieza de datos (`CreditDataCleaner`)**

Implementa reglas para corregir inconsistencias y valores inválidos antes del entrenamiento:

- Agrupa algunas categorías poco frecuentes de `tipo_credito` bajo la categoría `otros`.
- Identifica edades fuera del rango permitido (18 a 100 años).
- Trata como faltantes los salarios menores o iguales a cero y superiores a 100 millones.
- Identifica puntajes de Datacrédito inferiores a 150.
- Convierte en valores faltantes las categorías de `tendencia_ingresos` que no pertenecen a las opciones esperadas.
- Excluye las variables definidas como no utilizadas en el modelado.
- Valida que los datos de entrada sean un `DataFrame` y contengan las columnas requeridas.

**Transformación de variables (`build_preprocessor`)**

Utiliza un `ColumnTransformer` para aplicar tratamientos diferenciados según el tipo de variable:

| Tipo de variable | Tratamiento |
|---|---|
| Monetarias | Transformación logarítmica `log1p`, imputación por mediana y estandarización. |
| Numéricas restantes | Imputación por mediana y estandarización. |
| Categóricas | Imputación con la categoría `Faltante` y codificación One-Hot. |

La codificación utiliza `handle_unknown='ignore'`, lo que permite procesar categorías no observadas durante el entrenamiento sin generar errores.

Finalmente, `build_preprocessing_pipeline()` integra la limpieza y la transformación en un único pipeline. De esta manera, las mismas reglas se aplican durante el entrenamiento, la evaluación y las predicciones sobre nuevos registros.

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

# 2. Ingeniería de características
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

La aplicación desarrollada con **Streamlit** permite consultar los resultados del monitoreo mediante diferentes vistas:

- **Resumen general:** indicadores del estado del monitoreo y principales alertas.
- **Distribuciones:** comparación visual entre los datos de referencia y los datos evaluados.
- **Métricas de drift:** consulta de indicadores estadísticos por variable.
- **Evolución temporal:** seguimiento de los resultados registrados en distintas ejecuciones.
- **Desempeño del modelo:** visualización de métricas de clasificación y resultados de evaluación.
- **Recomendaciones:** mensajes y sugerencias según los niveles de drift identificados.

El panel incluye indicadores visuales de alerta, filtros y opciones para consultar y descargar reportes.

**Alcance actual del monitoreo:** la comparación utiliza un conjunto de prueba fijo como muestra actual. No se reciben todavía datos nuevos desde producción ni se programan ejecuciones periódicas. Por tanto, los resultados permiten demostrar el funcionamiento de las métricas y del dashboard, pero no constituyen evidencia de estabilidad del modelo en un entorno productivo.

## 5. Resultados de evaluación

En la última ejecución documentada, el modelo de Regresión Logística obtuvo los siguientes resultados sobre el conjunto de evaluación:

| Métrica | Resultado |
|---|---:|
| Recall | 0,625 |
| Precisión | 0,085 |
| F1-score | 0,149 |
| PR-AUC | 0,179 |

El modelo logra identificar una parte de los incumplimientos reales, pero su precisión es baja. Esto significa que una proporción importante de las alertas de posible incumplimiento corresponde a falsos positivos.

Por ello, la selección de un modelo no debe basarse únicamente en el recall: también es necesario considerar el umbral de clasificación y los costos asociados a los distintos tipos de error.

Los resultados se actualizan al volver a ejecutar el entrenamiento y la evaluación.

## 6. Despliegue mediante API y Docker

El modelo seleccionado se disponibiliza mediante una API desarrollada con **FastAPI**, ubicada en `mlops_pipeline/model_deploy.py`.

La API permite recibir información y generar predicciones, incluyendo procesamiento por lotes. También dispone de una documentación interactiva para consultar y probar sus endpoints.

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

La imagen incluye el código necesario para ejecutar el servicio, sus dependencias específicas y el modelo entrenado.

## 7. Pruebas y calidad del código

El proyecto incorpora pruebas unitarias con `pytest` para validar los procesos de limpieza, transformación y preparación de datos implementados en `mlops_pipeline/model_preprocessing.py`.

Para ejecutar las pruebas desde la raíz del repositorio:

```bash
python -m pytest -v
```

Para generar un informe de cobertura:

```bash
python -m pytest --cov=mlops_pipeline.model_preprocessing --cov-report=term-missing
```

En la ejecución local documentada se obtuvieron **8 pruebas satisfactorias y una cobertura del 100 % sobre `model_preprocessing.py`**. Este porcentaje corresponde exclusivamente al módulo evaluado y no a la totalidad del proyecto.

### Integración con SonarCloud

El repositorio integra SonarCloud mediante un flujo de **GitHub Actions**, configurado para ejecutar pruebas y generar el informe de cobertura antes del análisis estático.

La integración contempla la revisión de:

- Calidad y mantenibilidad del código.
- Duplicaciones y posibles problemas de implementación.
- Seguridad y posibles vulnerabilidades.
- Cobertura de pruebas.

El análisis se ejecuta automáticamente ante cambios en las ramas configuradas del repositorio.

La cobertura local está validada; sin embargo, la asociación de este informe con las medidas de cobertura del panel de SonarCloud continúa pendiente de revisión.

## 8. Entorno y dependencias

El proyecto utiliza Python y dos archivos de dependencias, separados según el propósito de ejecución:

- `requirements.txt`: paquetes necesarios para el flujo de análisis, entrenamiento, monitoreo y visualización.
- `requirements-api.txt`: dependencias específicas para ejecutar la API.

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

Como alternativa, se puede utilizar `set_up.bat` desde la raíz del proyecto para crear e instalar el entorno. Después de su ejecución, activa el entorno en la terminal actual.

## 9. Resultados y artefactos generados

Los principales resultados del flujo se organizan dentro de `outputs/`:

| Ubicación | Descripción |
|---|---|
| `outputs/graficos_eda/` | Visualizaciones del análisis exploratorio. |
| `outputs/datasets/` | Conjuntos de entrenamiento y evaluación, originales y transformados. |
| `outputs/models/mejor_modelo.joblib` | Modelo seleccionado y entrenado. |
| `outputs/models/preprocesador.joblib` | Pipeline de limpieza y transformación ajustado. |
| `outputs/reports/comparacion_modelos.csv` | Resultados de la validación cruzada. |
| `outputs/reports/evaluacion_final.csv` | Métricas del modelo sobre el conjunto de evaluación. |
| `outputs/reports/curva_roc.png` | Curva ROC del modelo seleccionado. |
| `outputs/reports/reporte_data_drift.csv` | Métricas de drift de la ejecución más reciente. |
| `outputs/reports/historial_data_drift.csv` | Registro histórico de las ejecuciones de monitoreo. |
| `outputs/reports/datos_monitoreo.csv` | Registros evaluados, predicciones y probabilidades. |

La carpeta `outputs/` está excluida del control de versiones mediante `.gitignore`. Por esta razón, los artefactos deben generarse ejecutando el flujo antes de iniciar la aplicación o construir la imagen Docker.

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
    ├── reports/
```

---

**Proyecto final Machine Learning | 2026 - 2**