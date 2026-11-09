import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


EXCLUDED_COLUMNS = [
    'puntaje',
    'fecha_prestamo',
    'cuota_pactada',
    'saldo_principal',
    'creditos_sectorFinanciero',
]

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


class CreditDataCleaner(BaseEstimator, TransformerMixin):
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


def build_preprocessor():
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
    return Pipeline(steps=[
        ('limpieza', CreditDataCleaner()),
        ('preprocesador', build_preprocessor()),
    ])
