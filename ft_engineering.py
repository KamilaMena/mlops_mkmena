# 1. Importar las librerías necesarias

import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from cargar_datos import datos as db

# 2. Definir las rutas y configuraciones del proyecto

CARPETA_PROYECTO = Path(__file__).resolve().parent
CARPETA_OUTPUTS = CARPETA_PROYECTO / 'outputs' / 'datasets'

VARIABLE_OBJETIVO = 'Pago_atiempo'

TEST_SIZE = 0.25
RANDOM_STATE = 42


# 3. Preparar los datos

# Crear una copia para no modificar los datos originales
datos = db.copy()

# Convertir tipo_credito en variable categórica
datos['tipo_credito'] = datos['tipo_credito'].astype('string')

# Convertir fecha_prestamo a formato fecha
datos['fecha_prestamo'] = pd.to_datetime(
    datos['fecha_prestamo'],
    errors='coerce'
)

# Aplicar las reglas de limpieza definidas en el EDA

# Edades negativas o superiores a 100: convertir en nulos
datos.loc[
    (datos['edad_cliente'] < 0) |
    (datos['edad_cliente'] > 100),
    'edad_cliente'
] = np.nan

# Convertir en nulos los puntajes datacrédito menores a 150
datos.loc[
    datos['puntaje_datacredito'] < 150,
    'puntaje_datacredito'
] = np.nan

# Conservar únicamente las categorías válidas de tendencia
categorias_tendencia = [
    'Decreciente',
    'Estable',
    'Creciente'
]

datos.loc[
    datos['tendencia_ingresos'].notna() &
    ~datos['tendencia_ingresos'].isin(categorias_tendencia),
    'tendencia_ingresos'
] = np.nan


# 4. Seleccionar las variables para el modelado

# Eliminar variables excluidas durante el EDA
columnas_eliminar = [
    'puntaje',
    'fecha_prestamo',
    'cuota_pactada',
    'saldo_principal',
    'creditos_sectorFinanciero'
]

datos = datos.drop(
    columns=columnas_eliminar
)

# Separar las features y la variable objetivo
X = datos.drop(columns=[VARIABLE_OBJETIVO])
y = datos[VARIABLE_OBJETIVO]


# 5. Clasificar las variables según su naturaleza

# Variables categóricas nominales
variables_nominales = [
    'tipo_credito',
    'tipo_laboral'
]

# Variables categóricas ordinales
variables_ordinales = [
    'tendencia_ingresos'
]

# Variables numéricas
variables_numericas = X.select_dtypes(
    include=['number']
).columns.tolist()

print('Variables numéricas:', variables_numericas)
print('Variables nominales:', variables_nominales)
print('Variables ordinales:', variables_ordinales)

# 6. Dividir los datos en entrenamiento y evaluación

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

print('\nDimensiones de los conjuntos:')
print('X_train:', X_train.shape)
print('X_test:', X_test.shape)
print('y_train:', y_train.shape)
print('y_test:', y_test.shape)

# 7. Crear los pipelines de transformación

# 7.1 Pipeline para variables numéricas
pipeline_numerico = Pipeline(steps=[
    ('imputador', SimpleImputer(strategy='median'))
])


# 7.2 Pipeline para variables categóricas nominales
pipeline_nominal = Pipeline(steps=[
    ('imputador', SimpleImputer(strategy='most_frequent')),
    ('codificador', OneHotEncoder(
        handle_unknown='ignore',
        sparse_output=False
    ))
])


# 7.3 Pipeline para variables categóricas ordinales
pipeline_ordinal = Pipeline(steps=[
    ('imputador', SimpleImputer(strategy='most_frequent')),
    ('codificador', OrdinalEncoder(
        categories=[categorias_tendencia],
        handle_unknown='use_encoded_value',
        unknown_value=-1
    ))
])


# 8. Integrar los pipelines en un ColumnTransformer

preprocesador = ColumnTransformer(
    transformers=[
        ('numeric', pipeline_numerico, variables_numericas),
        ('categoric', pipeline_nominal, variables_nominales),
        ('categoric_ordinal', pipeline_ordinal, variables_ordinales)
    ],
    remainder='drop',
    verbose_feature_names_out=False
)


# 9. Ajustar y aplicar las transformaciones

# Aprender los parámetros de transformación con entrenamiento
X_train_transformado = preprocesador.fit_transform(X_train)

# Aplicar las mismas transformaciones a evaluación
X_test_transformado = preprocesador.transform(X_test)


# 10. Recuperar los nombres de las características generadas

nombres_features = preprocesador.get_feature_names_out()

# Convertir los resultados en DataFrames
X_train_transformado = pd.DataFrame(
    X_train_transformado,
    columns=nombres_features,
    index=X_train.index
)

X_test_transformado = pd.DataFrame(
    X_test_transformado,
    columns=nombres_features,
    index=X_test.index
)


# 11. Incorporar la variable objetivo

dataset_train = X_train_transformado.copy()
dataset_train[VARIABLE_OBJETIVO] = y_train

dataset_test = X_test_transformado.copy()
dataset_test[VARIABLE_OBJETIVO] = y_test


# 12. Guardar los conjuntos de datos resultantes

CARPETA_OUTPUTS.mkdir(parents=True, exist_ok=True)

dataset_train.to_csv(
    CARPETA_OUTPUTS / 'dataset_train.csv',
    index=False
)

dataset_test.to_csv(
    CARPETA_OUTPUTS / 'dataset_test.csv',
    index=False
)


# 13. Mostrar los resultados

print('\n Feature Engineering completado:')

print('\nDimensiones finales:')
print('Dataset de entrenamiento:', dataset_train.shape)
print('Dataset de evaluación:', dataset_test.shape)

print('\nDistribución de la variable objetivo:')
print('\nEntrenamiento:')
print(dataset_train[VARIABLE_OBJETIVO].value_counts())

print('\nEvaluación:')
print(dataset_test[VARIABLE_OBJETIVO].value_counts())

print('\nValores nulos restantes:')
print('Entrenamiento:', dataset_train.isnull().sum().sum())
print('Evaluación:', dataset_test.isnull().sum().sum())

print('\nCaracterísticas generadas:')
print(nombres_features.tolist())

print('\nDatasets guardados correctamente.')
