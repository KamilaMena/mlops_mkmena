import numpy as np
import pytest
from fastapi import HTTPException

from mlops_pipeline import model_deploy


class ModeloFalso:
    classes_ = np.array([0, 1])

    def predict(self, datos):
        return np.array([0, 1])[:len(datos)]

    def predict_proba(self, datos):
        return np.array([
            [0.90, 0.10],
            [0.20, 0.80]
        ])[:len(datos)]


def crear_registro_valido():
    """Crea un registro con todas las columnas requeridas por la API."""
    return {
        "cant_creditosvigentes": 5,
        "capital_prestado": 2000000,
        "creditos_sectorCooperativo": 1,
        "creditos_sectorReal": 0,
        "edad_cliente": 35,
        "huella_consulta": 3,
        "plazo_meses": 12,
        "promedio_ingresos_datacredito": 2500000,
        "puntaje_datacredito": 750,
        "salario_cliente": 3000000,
        "saldo_mora": 0,
        "saldo_mora_codeudor": 0,
        "saldo_total": 2000000,
        "tendencia_ingresos": "Creciente",
        "tipo_credito": "4",
        "tipo_laboral": "Empleado",
        "total_otros_prestamos": 200000
    }


def test_health_informa_estado_correctamente():
    resultado = model_deploy.verificar_estado()

    assert resultado["estado"] == "activo"
    assert resultado["modelo_cargado"] is True
    assert resultado["nombre_modelo"] == "mejor_modelo.joblib"


def test_predict_json_procesa_lote_y_rechaza_datos_incompletos(monkeypatch):
    # Sustituimos el modelo real por uno ficticio.
    monkeypatch.setattr(model_deploy, "modelo", ModeloFalso())

    # Enviamos dos registros válidos.
    registros = [
        crear_registro_valido(),
        crear_registro_valido()
    ]

    resultado = model_deploy.predecir_json(
        {"records": registros}
    )

    assert resultado["total_registros"] == 2
    assert len(resultado["predicciones"]) == 2

    assert resultado["predicciones"][0]["clasificacion"] == "Pago a tiempo"
    assert resultado["predicciones"][1]["clasificacion"] == "Incumplimiento"

    assert resultado["predicciones"][0]["probabilidad_incumplimiento"] == pytest.approx(0.10)
    assert resultado["predicciones"][1]["probabilidad_incumplimiento"] == pytest.approx(0.80)

    # Comprobamos que la API rechaza registros incompletos.
    with pytest.raises(HTTPException) as error:
        model_deploy.predecir_json(
            {"records": [{"tipo_credito": "4"}]}
        )

    assert error.value.status_code == 400
    assert "Faltan las siguientes columnas obligatorias" in error.value.detail
