import joblib
import numpy as np
import pandas as pd
import pytest

from mlops_pipeline import ft_engineering, model_preprocessing
from mlops_pipeline.ft_engineering import (
    EXCLUDED_COLUMNS,
    RAZONES_EXCLUSION,
    VARIABLE_OBJETIVO,
    CreditDataCleaner,
    build_preprocessor,
    build_preprocessing_pipeline,
    dividir_datos,
    generar_particiones,
    preparar_dataset_modelado,
)


def crear_datos_prueba():
    """Crea un conjunto pequeño de datos para las pruebas."""
    return pd.DataFrame({
        'tipo_credito': ['6', '1', '68'],
        'tipo_laboral': ['Empleado', 'Independiente', 'Empleado'],
        'tendencia_ingresos': [
            'Creciente', 'Estable', 'Invalida'
        ],
        'capital_prestado': [1000000, 2000000, 1500000],
        'salario_cliente': [2000000, 3000000, 2500000],
        'total_otros_prestamos': [100000, 200000, 150000],
        'saldo_mora': [0, 100000, 50000],
        'saldo_total': [1000000, 2000000, 1500000],
        'saldo_mora_codeudor': [0, 0, 10000],
        'promedio_ingresos_datacredito': [1800000, 2800000, 2300000],
        'edad_cliente': [25, 35, 120],
        'puntaje_datacredito': [700, 600, 100],
        'plazo_meses': [12, 24, 36],
        'cant_creditosvigentes': [1, 2, 3],
        'huella_consulta': [2, 3, 1],
        'creditos_sectorCooperativo': [0, 1, 0],
        'creditos_sectorReal': [1, 0, 2],
        'puntaje': [700, 600, 500],
        'fecha_prestamo': ['2025-01-01', '2025-02-01', '2025-03-01'],
        'cuota_pactada': [100000, 200000, 150000],
        'saldo_principal': [900000, 1800000, 1350000],
        'creditos_sectorFinanciero': [1, 2, 1],
    })


def crear_base_con_objetivo(repeticiones=20):
    """Base más grande (con objetivo) para probar la partición."""
    base = pd.concat(
        [crear_datos_prueba()] * repeticiones, ignore_index=True
    )
    base[VARIABLE_OBJETIVO] = [0 if i % 5 == 0 else 1 for i in range(len(base))]
    return base


def test_fit_devuelve_el_limpiador():
    limpiador = CreditDataCleaner()
    resultado = limpiador.fit(crear_datos_prueba())

    assert resultado is limpiador


def test_limpiador_corrige_valores_invalidos():
    datos = crear_datos_prueba()

    resultado = CreditDataCleaner().fit_transform(datos)

    assert resultado.loc[0, 'tipo_credito'] == 'otros'
    assert resultado.loc[2, 'tipo_credito'] == 'otros'
    assert pd.isna(resultado.loc[2, 'edad_cliente'])
    assert pd.isna(resultado.loc[2, 'puntaje_datacredito'])
    assert pd.isna(resultado.loc[2, 'tendencia_ingresos'])


def test_limpiador_elimina_columnas_excluidas():
    resultado = CreditDataCleaner().fit_transform(
        crear_datos_prueba()
    )

    columnas_excluidas = [
        'puntaje',
        'fecha_prestamo',
        'cuota_pactada',
        'saldo_principal',
        'creditos_sectorFinanciero',
    ]

    for columna in columnas_excluidas:
        assert columna not in resultado.columns


def test_toda_exclusion_tiene_una_razon_documentada():
    assert set(EXCLUDED_COLUMNS) == set(RAZONES_EXCLUSION)
    assert all(RAZONES_EXCLUSION.values())


def test_limpiador_rechaza_entradas_invalidas():
    limpiador = CreditDataCleaner()

    with pytest.raises(TypeError):
        limpiador.fit([1, 2, 3])


def test_limpiador_detecta_columnas_faltantes():
    datos = crear_datos_prueba().drop(columns=['edad_cliente'])

    with pytest.raises(ValueError, match='Faltan columnas requeridas'):
        CreditDataCleaner().fit(datos)


def test_limpiador_detecta_salarios_invalidos():
    datos = crear_datos_prueba()
    datos.loc[0, 'salario_cliente'] = -1000
    datos.loc[1, 'salario_cliente'] = 200000000

    resultado = CreditDataCleaner().fit_transform(datos)

    assert pd.isna(resultado.loc[0, 'salario_cliente'])
    assert pd.isna(resultado.loc[1, 'salario_cliente'])


def test_build_preprocessor_devuelve_transformador():
    preprocesador = build_preprocessor()

    assert preprocesador is not None
    assert hasattr(preprocesador, 'fit_transform')


def test_pipeline_completo_transforma_datos():
    datos = crear_datos_prueba()

    pipeline = build_preprocessing_pipeline()
    resultado = pipeline.fit_transform(datos)

    assert isinstance(resultado, np.ndarray)
    assert resultado.shape[0] == len(datos)
    assert resultado.shape[1] > 0
    assert np.isfinite(resultado).all()


def test_dividir_datos_es_estratificada_y_sin_objetivo_en_x():
    base = crear_base_con_objetivo()

    X_train, X_test, y_train, y_test = dividir_datos(base)

    assert VARIABLE_OBJETIVO not in X_train.columns
    assert len(X_train) + len(X_test) == len(base)
    assert len(X_test) == round(len(base) * 0.25)
    assert y_train.mean() == pytest.approx(y_test.mean(), abs=0.05)


def test_preparar_dataset_modelado_devuelve_datos_listos():
    base = crear_base_con_objetivo()
    X_train, X_test, _, _ = dividir_datos(base)

    X_train_t, X_test_t, pipeline = preparar_dataset_modelado(X_train, X_test)

    assert list(X_train_t.columns) == list(X_test_t.columns)
    assert len(X_train_t) == len(X_train)
    assert len(X_test_t) == len(X_test)
    assert np.isfinite(X_train_t.to_numpy()).all()
    assert np.isfinite(X_test_t.to_numpy()).all()
    assert 'limpieza' in pipeline.named_steps


def test_generar_particiones_guarda_los_archivos(tmp_path):
    base = crear_base_con_objetivo()

    generar_particiones(base, carpeta_salida=tmp_path)

    esperados = [
        'dataset_train_raw.csv',
        'dataset_test_raw.csv',
        'dataset_test_api.csv',
        'dataset_train.csv',
        'dataset_test.csv',
    ]
    for nombre in esperados:
        assert (tmp_path / nombre).exists()

    assert VARIABLE_OBJETIVO in pd.read_csv(tmp_path / 'dataset_train.csv').columns
    assert VARIABLE_OBJETIVO not in pd.read_csv(tmp_path / 'dataset_test_api.csv').columns


def test_model_preprocessing_reexporta_las_mismas_clases():
    # Protege la carga de modelos .joblib guardados con la ruta anterior.
    assert model_preprocessing.CreditDataCleaner is ft_engineering.CreditDataCleaner
    assert (
        model_preprocessing.build_preprocessing_pipeline
        is ft_engineering.build_preprocessing_pipeline
    )
    assert model_preprocessing.EXCLUDED_COLUMNS == ft_engineering.EXCLUDED_COLUMNS


def test_pipeline_se_guarda_y_se_carga_con_joblib(tmp_path):
    datos = crear_datos_prueba()
    pipeline = build_preprocessing_pipeline().fit(datos)
    ruta = tmp_path / 'pipeline.joblib'

    joblib.dump(pipeline, ruta)
    recargado = joblib.load(ruta)

    np.testing.assert_allclose(
        pipeline.transform(datos), recargado.transform(datos)
    )