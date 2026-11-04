from pathlib import Path
import pandas as pd

carpeta_proyecto = Path(__file__).resolve().parent
archivo_csv = carpeta_proyecto / "Base_de_datos.csv"
archivo_excel = carpeta_proyecto / "BD_creditos.xlsx"

if archivo_csv.exists():
    datos = pd.read_csv(archivo_csv, low_memory=False)
    print(f"Datos cargados desde: {archivo_csv}")
elif archivo_excel.exists():
    datos = pd.read_excel(archivo_excel)
    print(f"Datos cargados desde: {archivo_excel}")
else:
    raise FileNotFoundError(
        "No se encontró Base_de_datos.csv ni BD_creditos.xlsx en la raíz del proyecto."
    )

print(f"\nDimensiones: {datos.shape[0]:,} filas y {datos.shape[1]} columnas")

print("\nVista previa (primeras 10 filas, datos originales):")
print(datos.head(10).to_string(index=False))

print("\nTipos de datos originales:")
print(datos.dtypes.to_string())

print("\nValores faltantes por columna:")
print(datos.isna().sum().sort_values(ascending=False).to_string())