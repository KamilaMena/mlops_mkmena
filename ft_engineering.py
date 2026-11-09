# 1. Importar las librerías necesarias

import pandas as pd

from pathlib import Path

from sklearn.model_selection import train_test_split
from cargar_datos import datos as db

# 2. Definir las rutas y configuraciones del proyecto

CARPETA_PROYECTO = Path(__file__).resolve().parent
CARPETA_OUTPUTS = CARPETA_PROYECTO / 'outputs' / 'datasets'

VARIABLE_OBJETIVO = 'Pago_atiempo'

TEST_SIZE = 0.25
RANDOM_STATE = 42


X = db.drop(columns=[VARIABLE_OBJETIVO])
y = db[VARIABLE_OBJETIVO]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print('\nDimensiones de los conjuntos:')
print('X_train:', X_train.shape)
print('X_test:', X_test.shape)
print('y_train:', y_train.shape)
print('y_test:', y_test.shape)

# Guardar particiones originales; el pipeline completo las transforma durante
# la validación cruzada y el entrenamiento del modelo.

CARPETA_OUTPUTS.mkdir(parents=True, exist_ok=True)
X_train.assign(**{VARIABLE_OBJETIVO: y_train}).to_csv(
    CARPETA_OUTPUTS / 'dataset_train_raw.csv',
    index=False
)

X_test.assign(**{VARIABLE_OBJETIVO: y_test}).to_csv(
    CARPETA_OUTPUTS / 'dataset_test_raw.csv',
    index=False
)

print('\nParticiones originales listas para el pipeline completo:')
print('Entrenamiento:', X_train.shape)
print('Evaluación:', X_test.shape)

print('\nDistribución de la variable objetivo:')
print('\nEntrenamiento:')
print(y_train.value_counts())

print('\nEvaluación:')
print(y_test.value_counts())
print('\nParticiones guardadas en outputs/datasets/*_raw.csv')
