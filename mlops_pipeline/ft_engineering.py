"""Ingeniería de características del proyecto de riesgo crediticio.

Este módulo es la primera pieza del flujo de modelado. Se puede importar sin
efectos secundarios (no lee datos ni escribe archivos al importarse), de modo
que lo usan tanto el entrenamiento como la API y el monitoreo.

Flujo de transformación
-----------------------
1. ``dividir_datos``: separa X / y y hace una partición estratificada
   (75 % entrenamiento, 25 % evaluación) para conservar la proporción de
   incumplimientos (~4,75 %) en ambos conjuntos.
2. ``CreditDataCleaner`` (paso ``limpieza``): convierte en nulo los valores
   imposibles detectados en el EDA, agrupa tipos de crédito raros y elimina
   las columnas que no entran al modelo.
3. ``build_preprocessor`` (paso ``preprocesador``): imputación, escalado y
   codificación diferenciados por tipo de variable.
4. ``preparar_dataset_modelado``: ajusta el pipeline SOLO con entrenamiento y
   devuelve los conjuntos transformados, listos para modelar.

Se ejecuta con: ``python -m mlops_pipeline.ft_engineering``
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


# ---------------------------------------------------------------------------
# 1. Configuración
# ---------------------------------------------------------------------------

CARPETA_PROYECTO = Path(__file__).resolve().parent.parent
CARPETA_DATASETS = CARPETA_PROYECTO / 'outputs' / 'datasets'

VARIABLE_OBJETIVO = 'Pago_atiempo'
TEST_SIZE = 0.25
RANDOM_STATE = 42

# Umbral de |correlación| a partir del cual se considera que dos variables
# numéricas son redundantes (ver sección 6.1 del EDA).
UMBRAL_CORRELACION = 0.70

# Columnas que NO entran al modelo y la razón de cada decisión.
RAZONES_EXCLUSION = {
    'puntaje': 'Posible fuga de datos (correlación 0,92 con el objetivo, '
               'AUC = 1,0 por sí sola). Ver sección 3 del EDA.',
    'fecha_prestamo': 'El comportamiento cambia con el tiempo; se conserva '
                      'para el análisis temporal y el monitoreo, no como predictor.',
    'cuota_pactada': 'Redundante con capital_prestado (r ~ 0,76 >= 0,70). '
                     'Se conserva capital_prestado.',
    'saldo_principal': 'Redundante con saldo_total (r ~ 0,73 >= 0,70). '
                       'Se conserva saldo_total.',
    'creditos_sectorFinanciero': 'Redundante con cant_creditosvigentes '
                                 '(r ~ 0,79 >= 0,70). Se conserva el total.',
}

EXCLUDED_COLUMNS = list(RAZONES_EXCLUSION)

CATEGORICAL_COLUMNS = [
    'tipo_credito',
    'tipo_laboral',
    'tendencia_ingresos',
]

MONETARY_COLUMNS = [
    'capital_prestado',
    'salario_cliente',
    'total_otros_prestamos',
    'saldo_mora',
    'saldo_total',
    'saldo_mora_codeudor',
    'promedio_ingresos_datacredito',
]

TREND_CATEGORIES = ['Decreciente', 'Estable', 'Creciente']
RARE_CREDIT_TYPES = {'6', '7', '68'}


# ---------------------------------------------------------------------------
# 2. Limpieza (reglas de validación del EDA)
# ---------------------------------------------------------------------------

class CreditDataCleaner(BaseEstimator, TransformerMixin):
    """Aplica las reglas de validación del EDA sin eliminar filas.

    Los valores imposibles pasan a nulo y se imputan después, dentro del
    pipeline, usando solo estadísticas del conjunto de entrenamiento.
    """

    def fit(self, X, y=None):
        self._validate_input(X)
        return self

    def transform(self, X):
        self._validate_input(X)
        datos = X.copy()

        datos['tipo_credito'] = datos['tipo_credito'].astype('string')
        datos['tipo_credito'] = datos['tipo_credito'].replace(
            {tipo: 'otros' for tipo in RARE_CREDIT_TYPES}
        )
        datos['tipo_credito'] = datos['tipo_credito'].astype(object).where(
            datos['tipo_credito'].notna(),
            np.nan
        )

        datos.loc[
            (datos['edad_cliente'] < 18) | (datos['edad_cliente'] > 100),
            'edad_cliente'
        ] = np.nan
        datos.loc[
            (datos['salario_cliente'] <= 0)
            | (datos['salario_cliente'] > 100_000_000),
            'salario_cliente'
        ] = np.nan
        datos.loc[
            datos['puntaje_datacredito'] < 150,
            'puntaje_datacredito'
        ] = np.nan
        datos.loc[
            datos['tendencia_ingresos'].notna()
            & ~datos['tendencia_ingresos'].isin(TREND_CATEGORIES),
            'tendencia_ingresos'
        ] = np.nan

        return datos.drop(columns=EXCLUDED_COLUMNS, errors='ignore')

    @staticmethod
    def _validate_input(X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError('Los datos de entrada deben ser un pandas DataFrame.')

        required = set(CATEGORICAL_COLUMNS + MONETARY_COLUMNS + [
            'edad_cliente',
            'puntaje_datacredito',
            'plazo_meses',
            'cant_creditosvigentes',
            'huella_consulta',
            'creditos_sectorCooperativo',
            'creditos_sectorReal',
        ])
        missing = sorted(required.difference(X.columns))
        if missing:
            raise ValueError(
                f'Faltan columnas requeridas para preparar los datos: {missing}'
            )


# ---------------------------------------------------------------------------
# 3. Transformación (imputación, escalado, codificación)
# ---------------------------------------------------------------------------

def build_preprocessor():
    """ColumnTransformer con un tratamiento distinto por tipo de variable.

    - Monetarias: log1p (asimetría fuerte) + mediana + estandarización.
    - Numéricas restantes: mediana + estandarización.
    - Categóricas: categoría 'Faltante' (la ausencia es informativa) + One-Hot.
    """
    numeric_columns = [
        'plazo_meses',
        'edad_cliente',
        'puntaje_datacredito',
        'cant_creditosvigentes',
        'huella_consulta',
        'saldo_mora',
        'saldo_total',
        'saldo_mora_codeudor',
        'creditos_sectorCooperativo',
        'creditos_sectorReal',
    ]
    monetary_columns = [
        'capital_prestado',
        'salario_cliente',
        'total_otros_prestamos',
        'saldo_mora',
        'saldo_total',
        'saldo_mora_codeudor',
        'promedio_ingresos_datacredito',
    ]
    remaining_numeric = [
        column for column in numeric_columns
        if column not in monetary_columns
    ]

    monetary_pipeline = Pipeline(steps=[
        ('log1p', FunctionTransformer(
            np.log1p,
            feature_names_out='one-to-one'
        )),
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])
    numeric_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])
    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Faltante')),
        ('onehot', OneHotEncoder(
            handle_unknown='ignore',
            sparse_output=False
        )),
    ])

    return ColumnTransformer(
        transformers=[
            ('monetary', monetary_pipeline, monetary_columns),
            ('numeric', numeric_pipeline, remaining_numeric),
            ('categorical', categorical_pipeline, CATEGORICAL_COLUMNS),
        ],
        remainder='drop',
        verbose_feature_names_out=False,
    )


def build_preprocessing_pipeline():
    """Pipeline completo: limpieza + preprocesamiento (sin modelo)."""
    return Pipeline(steps=[
        ('limpieza', CreditDataCleaner()),
        ('preprocesador', build_preprocessor()),
    ])


# ---------------------------------------------------------------------------
# 4. Partición y dataset listo para modelar
# ---------------------------------------------------------------------------

def dividir_datos(db, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    """Partición estratificada entrenamiento / evaluación."""
    X = db.drop(columns=[VARIABLE_OBJETIVO])
    y = db[VARIABLE_OBJETIVO]
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )


def preparar_dataset_modelado(X_train, X_test):
    """Ajusta el pipeline con entrenamiento y transforma ambos conjuntos.

    Devuelve (X_train_transformado, X_test_transformado, pipeline_ajustado).
    El ajuste usa solo X_train para no filtrar información de evaluación.
    """
    pipeline = build_preprocessing_pipeline()
    pipeline.fit(X_train)

    limpieza = pipeline.named_steps['limpieza']
    preprocesador = pipeline.named_steps['preprocesador']
    columnas = preprocesador.get_feature_names_out()

    X_train_t = pd.DataFrame(
        preprocesador.transform(limpieza.transform(X_train)),
        columns=columnas,
        index=X_train.index
    )
    X_test_t = pd.DataFrame(
        preprocesador.transform(limpieza.transform(X_test)),
        columns=columnas,
        index=X_test.index
    )
    return X_train_t, X_test_t, pipeline


def generar_particiones(db, carpeta_salida=CARPETA_DATASETS):
    """Genera y guarda todos los conjuntos que usan las etapas siguientes."""
    X_train, X_test, y_train, y_test = dividir_datos(db)

    carpeta_salida = Path(carpeta_salida)
    carpeta_salida.mkdir(parents=True, exist_ok=True)

    # Particiones originales: el entrenamiento las pasa por el pipeline completo.
    X_train.assign(**{VARIABLE_OBJETIVO: y_train}).to_csv(
        carpeta_salida / 'dataset_train_raw.csv', index=False
    )
    X_test.assign(**{VARIABLE_OBJETIVO: y_test}).to_csv(
        carpeta_salida / 'dataset_test_raw.csv', index=False
    )
    # Entrada de ejemplo para probar la API (sin variable objetivo).
    X_test.to_csv(carpeta_salida / 'dataset_test_api.csv', index=False)

    # Datasets transformados, listos para modelar y para el monitoreo.
    X_train_t, X_test_t, _ = preparar_dataset_modelado(X_train, X_test)
    X_train_t.assign(**{VARIABLE_OBJETIVO: y_train}).to_csv(
        carpeta_salida / 'dataset_train.csv', index=False
    )
    X_test_t.assign(**{VARIABLE_OBJETIVO: y_test}).to_csv(
        carpeta_salida / 'dataset_test.csv', index=False
    )

    return X_train, X_test, y_train, y_test, X_train_t, X_test_t


def main():  # pragma: no cover
    from mlops_pipeline.cargar_datos import datos

    X_train, X_test, y_train, y_test, X_train_t, X_test_t = generar_particiones(datos)

    print('\nDimensiones originales:')
    print('Entrenamiento:', X_train.shape, '| Evaluación:', X_test.shape)
    print('\nDimensiones transformadas (listas para modelar):')
    print('Entrenamiento:', X_train_t.shape, '| Evaluación:', X_test_t.shape)
    print('\nDistribución de la variable objetivo:')
    print('Entrenamiento:', y_train.value_counts().to_dict())
    print('Evaluación:', y_test.value_counts().to_dict())
    print('\nVariables excluidas del modelo:')
    for columna, razon in RAZONES_EXCLUSION.items():
        print(f'- {columna}: {razon}')
    print('\nArchivos guardados en outputs/datasets/')


if __name__ == '__main__':
    main()