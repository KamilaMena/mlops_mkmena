from pathlib import Path
from io import StringIO
import sys

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException, UploadFile, File, Body
from fastapi.responses import StreamingResponse
from mlops_pipeline import model_preprocessing

# Supports loading models serialized before model_preprocessing moved into the package.
sys.modules.setdefault("model_preprocessing", model_preprocessing)

# 1. Carga del modelo entrenado
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "outputs" / "models" / "mejor_modelo.joblib"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"No se encontró el modelo entrenado en: {MODEL_PATH}"
    )

# Cargamos el pipeline completo:
# limpieza + preprocesamiento + clasificador.
modelo = joblib.load(MODEL_PATH)

app = FastAPI(
    title="API de Predicción de Riesgo Crediticio",
    description=(
        "API para predecir incumplimiento de pago "
        "utilizando el modelo de Machine Learning entrenado."
    ),
    version="1.0.0"
)

# 2. Función de predicción
def realizar_predicciones(datos: pd.DataFrame) -> pd.DataFrame:
    """
    Recibe datos originales, aplica el pipeline completo
    y devuelve las predicciones y probabilidades.
    """

    # Columnas que el modelo necesita para realizar predicciones
    COLUMNAS_REQUERIDAS = [
        "cant_creditosvigentes",
        "capital_prestado",
        "creditos_sectorCooperativo",
        "creditos_sectorReal",
        "edad_cliente",
        "huella_consulta",
        "plazo_meses",
        "promedio_ingresos_datacredito",
        "puntaje_datacredito",
        "salario_cliente",
        "saldo_mora",
        "saldo_mora_codeudor",
        "saldo_total",
        "tendencia_ingresos",
        "tipo_credito",
        "tipo_laboral",
        "total_otros_prestamos"
    ]

    # Evitar enviar la variable objetivo
    if "Pago_atiempo" in datos.columns:
        raise ValueError(
            "No debes enviar la columna 'Pago_atiempo'. "
            "Esta es la variable que el modelo debe predecir."
        )

    # Verificamos que estén todas las columnas necesarias
    columnas_faltantes = [
        columna for columna in COLUMNAS_REQUERIDAS
        if columna not in datos.columns
    ]

    if columnas_faltantes:
        raise ValueError(
            f"Faltan las siguientes columnas obligatorias: "
            f"{columnas_faltantes}"
        )

    if datos.empty:
        raise ValueError("No se recibieron registros para predecir.")


    # El pipeline realiza automáticamente la limpieza,
    # el preprocesamiento y la clasificación.
    predicciones = modelo.predict(datos)

    # Identificamos la posición de la clase 1,
    # que representa incumplimiento.
    clases = list(modelo.classes_)

    if 1 not in clases:
        raise ValueError(
            "El modelo no contiene la clase 1 (incumplimiento)."
        )

    indice_incumplimiento = clases.index(1)

    probabilidades = modelo.predict_proba(datos)[
        :, indice_incumplimiento
    ]

    # Copiamos los datos originales para conservar
    # la información recibida.
    resultados = datos.copy()

    resultados["prediccion_incumplimiento"] = [
        int(valor) for valor in predicciones
    ]

    resultados["clasificacion"] = [
        "Incumplimiento" if int(valor) == 1
        else "Pago a tiempo"
        for valor in predicciones
    ]

    resultados["probabilidad_incumplimiento"] = probabilidades

    return resultados

# 3. Verificación del estado de la API
@app.get("/health")
def verificar_estado():
    """
    Comprueba que la API esté funcionando
    y que el modelo se haya cargado.
    """

    return {
        "estado": "activo",
        "modelo_cargado": modelo is not None,
        "nombre_modelo": MODEL_PATH.name
    }

# 4. Predicción a partir de JSON
@app.post("/predict")
def predecir_json(payload: object = Body(...)):
    """
    Recibe uno o varios registros en formato JSON.

    Formatos admitidos:

    1. Un registro:
       {"records": [{"columna": valor, ...}]}

    2. Varios registros:
       {"records": [{...}, {...}]}

    3. Lista directa:
       [{...}, {...}]
    """

    try:
        # Admitimos tanto una lista directa como
        # un objeto que contenga la clave "records".
        if isinstance(payload, list):
            registros = payload

        elif isinstance(payload, dict):
            registros = payload.get("records")

        else:
            raise ValueError(
                "El JSON debe ser una lista o un objeto con la clave 'records'."
            )

        if not isinstance(registros, list) or not registros:
            raise ValueError(
                "Debes enviar al menos un registro."
            )

        if not all(isinstance(registro, dict) for registro in registros):
            raise ValueError(
                "Cada registro debe ser un objeto JSON."
            )

        datos = pd.DataFrame(registros)

        resultados = realizar_predicciones(datos)

        # Convertir los valores NaN a None para que FastAPI
        # pueda serializarlos correctamente como null en JSON.
        resultados = resultados.astype(object).where(
            resultados.notna(),
            None
        )

        return {
            "total_registros": len(resultados),
            "predicciones": resultados.to_dict(orient="records")
        }

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=f"No fue posible procesar los datos: {str(error)}"
        )

# 5. Predicción a partir de CSV
@app.post("/predict/csv")
async def predecir_csv(archivo: UploadFile = File(...)):
    """
    Recibe un archivo CSV y devuelve otro CSV
    con las predicciones agregadas.
    """

    if not archivo.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="El archivo debe tener formato CSV."
        )

    try:
        contenido = await archivo.read()

        datos = pd.read_csv(StringIO(contenido.decode("utf-8")))

        resultados = realizar_predicciones(datos)

        salida = StringIO()
        resultados.to_csv(salida, index=False)
        salida.seek(0)

        return StreamingResponse(
            iter([salida.getvalue()]),
            media_type="text/csv",
            headers={
                "Content-Disposition": (
                    "attachment; filename=predicciones_crediticias.csv"
                )
            }
        )

    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=f"No fue posible procesar el archivo: {str(error)}"
        )
