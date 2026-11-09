# Predicción del riesgo de incumplimiento crediticio

Proyecto académico de Machine Learning para explorar una base de créditos,
entrenar modelos que identifiquen posibles incumplimientos y visualizar
métricas de monitoreo. Este README documenta el estado actual del código y
señala los entregables del proyecto que aún no están implementados.

## Caso de negocio

Una entidad que otorga créditos necesita identificar, antes de tomar decisiones,
solicitudes con mayor probabilidad de incumplimiento. El problema se plantea
como una clasificación binaria usando información del crédito, del cliente y
de su historial crediticio.

La variable objetivo es `Pago_atiempo`:

- `1`: pagó a tiempo.
- `0`: incumplió.

Durante el entrenamiento y el monitoreo se transforma la etiqueta para que
`1` represente incumplimiento. Así, el recall del modelo mide qué proporción de
los incumplimientos reales logra detectar. Este prototipo es académico y no
debe usarse para tomar decisiones crediticias reales sin validación del negocio,
revisión de sesgos y controles regulatorios.

## Datos

El cargador busca primero `Base_de_datos.csv` y, si no existe, usa
`BD_creditos.xlsx`. El archivo disponible en este repositorio de trabajo es
`BD_creditos.xlsx`, con 10.763 registros y 23 columnas. Si se usa otra fuente,
debe conservarse el esquema esperado por los scripts, incluida la columna
`Pago_atiempo`.

El EDA informa 10.613 registros en el conjunto analizado después de sus reglas
de preparación. En la fuente original, `Pago_atiempo=0` representa alrededor
del 4,75 % de los registros, por lo que las clases están desbalanceadas. Las
reglas de preparación y exclusión de variables están descritas en
`mlops_pipeline/ft_engineering.py` y `mlops_pipeline/eda_analisis.py`.

## Hallazgos principales del EDA

- Los faltantes se concentran en `tendencia_ingresos` y
  `promedio_ingresos_datacredito` (alrededor del 26 % en el conjunto analizado);
  también hay faltantes menores en algunas variables de saldos.
- Algunas variables monetarias presentan distribuciones muy asimétricas y
  valores extremos. No se fijan límites monetarios sin confirmar unidades y
  reglas con la fuente.
- Se encontraron 58 valores de `tendencia_ingresos` que no coinciden con las
  categorías esperadas (`Decreciente`, `Estable`, `Creciente`); se tratan como
  faltantes en la preparación para modelado.
- `puntaje` presenta una correlación de Pearson alta con el objetivo
  (`r = 0,926`). Esto no demuestra causalidad, pero su definición y momento de
  disponibilidad deben confirmarse para descartar fuga de información. Por esa
  razón se excluye del conjunto usado para modelar.
- Las diferencias entre grupos son descriptivas; por sí solas no prueban una
  relación causal.

Las tablas del EDA se muestran al ejecutar sus celdas y las gráficas se guardan
en `outputs/graficos_eda/`, relativo a la raíz del repositorio.

## Flujo del proyecto

Ejecuta los comandos desde la raíz del repositorio y en este orden:

1. `python -m mlops_pipeline.eda_analisis`: realiza el análisis exploratorio.
   En VS Code también puedes abrir `mlops_pipeline/eda_analisis.py` y usar
   **Run All Cells** para ejecutar las celdas `# %%`.
2. `python -m mlops_pipeline.ft_engineering`: divide los datos originales de forma estratificada y
   guarda las particiones crudas en `outputs/datasets/*_raw.csv`.
3. `python -m mlops_pipeline.model_training_evaluation`: compara Regresión Logística, Árbol de
   Decisión, Random Forest y XGBoost mediante validación cruzada estratificada.
   En cada pliegue ajusta la limpieza, imputación, transformaciones, escalado y
   codificación solo con los datos de entrenamiento del pliegue. Guarda el
   modelo completo y los datasets transformados para el monitoreo y el panel.
4. `python -m mlops_pipeline.model_monitoring`: genera predicciones para el conjunto de prueba y
   calcula métricas de drift frente al conjunto de entrenamiento.
5. `streamlit run app.py`: presenta reportes y evolución del drift en una
   aplicación Streamlit.

`mlops_pipeline/cargar_datos.py` no se ejecuta por separado: los scripts que
necesitan los datos lo importan y cargan el CSV o Excel disponible en la raíz
del repositorio. Los datos y resultados permanecen en la raíz (`BD_creditos.xlsx`
y `outputs/`); los módulos Python del flujo están en `mlops_pipeline/`.
El artefacto `outputs/models/mejor_modelo.joblib` contiene limpieza,
transformaciones y clasificador. Para puntuar datos nuevos, pásale un
`DataFrame` con las variables predictoras requeridas (sin `Pago_atiempo`) a
`predict` o `predict_proba`; no apliques el preprocesamiento por separado.
Las columnas excluidas durante el EDA, como `puntaje`, no son necesarias.

## Entorno y dependencias

Se requiere Python y los paquetes usados directamente por los scripts:
`pandas`, `numpy`, `openpyxl`, `matplotlib`, `seaborn`, `scikit-learn`,
`xgboost`, `scipy`, `joblib`, `streamlit`, `plotly` e `ipykernel` (para
ejecutar las celdas del archivo `.py` desde VS Code).

En Windows se puede crear y activar un entorno virtual e instalar los paquetes
con:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r .\requirements.txt
```

También se puede crear e instalar el entorno ejecutando `set_up.bat` desde la
raíz del proyecto. Al terminar, activa el entorno en la terminal actual con
`.\.venv\Scripts\Activate.ps1`. El archivo `requirements.txt` incluye
`ipykernel` para ejecutar las celdas `# %%` en VS Code.

## Artefactos generados

Los scripts crean sus resultados dentro de `outputs/`:

| Ruta | Contenido |
|---|---|
| `outputs/graficos_eda/` | Gráficas del análisis exploratorio. |
| `outputs/datasets/dataset_train.csv` | Datos transformados de entrenamiento. |
| `outputs/datasets/dataset_test.csv` | Datos transformados de evaluación. |
| `outputs/datasets/dataset_train_raw.csv` y `dataset_test_raw.csv` | Particiones originales que recibe el pipeline completo para entrenar, evaluar y predecir. |
| `outputs/models/mejor_modelo.joblib` | Modelo seleccionado y entrenado. |
| `outputs/models/preprocesador.joblib` | Pipeline de limpieza y transformación ajustado; el modelo completo también lo incluye. |
| `outputs/reports/comparacion_modelos.csv` | Métricas promedio de validación cruzada. |
| `outputs/reports/evaluacion_final.csv` | Métricas del modelo seleccionado sobre el conjunto de evaluación. |
| `outputs/reports/curva_roc.png` | Curva ROC del modelo seleccionado sobre el conjunto de evaluación. |
| `outputs/reports/reporte_data_drift.csv` | Métricas de drift de la ejecución más reciente. |
| `outputs/reports/historial_data_drift.csv` | Historial de ejecuciones de monitoreo. |
| `outputs/reports/datos_monitoreo.csv` | Registros de evaluación con predicciones y probabilidades. |

`outputs/` está excluida del control de versiones por `.gitignore`. Ejecuta el
flujo para regenerar estos archivos antes de iniciar la aplicación.

## API con Docker

La API FastAPI está en `mlops_pipeline/model_deploy.py`. Después de generar el
modelo con el flujo anterior y tener Docker instalado, construye y ejecuta la
imagen desde la raíz del repositorio:

```powershell
docker build -t riesgo-api .
docker run --rm -p 8000:8000 riesgo-api
```

La API queda disponible en `http://localhost:8000` y su documentación
interactiva en `http://localhost:8000/docs`.

## Evaluación y monitoreo actuales

En la última ejecución documentada del flujo, la Regresión Logística obtuvo
**recall 0,625**, **precisión 0,085**, **F1 0,149** y **PR-AUC 0,179** sobre el
conjunto de evaluación. La precisión baja implica que muchas alertas de
incumplimiento pueden ser falsos positivos; la selección por recall no elimina
la necesidad de revisar el umbral y el costo de errores junto con el negocio.
Los valores se actualizan al volver a ejecutar el entrenamiento.

El monitoreo implementado compara entrenamiento con un conjunto de prueba fijo.
Aunque calcula KS, PSI, Jensen-Shannon y chi-cuadrado, no recibe datos nuevos
de producción ni programa ejecuciones periódicas. Por tanto, sus resultados
son una demostración sobre los datos disponibles, no evidencia de estabilidad
en producción. El panel sí presenta alertas por umbral PSI y la evolución
registrada entre ejecuciones.

## Estructura actual

```text
.
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
    └── Contexto/
```