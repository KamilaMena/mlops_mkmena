# 1. Importar las librerías necesarias.
from pathlib import Path
import json
import pandas as pd
import numpy as np

# 2. Definir las rutas del proyecto.
# La guía propone Base_de_datos.csv. Para que también funcione con el archivo
# recibido en esta etapa, se acepta BD_creditos.xlsx.
carpeta_proyecto = Path(__file__).resolve().parent
carpeta_salida = carpeta_proyecto / "outputs"
carpeta_salida.mkdir(parents=True, exist_ok=True)

archivo_csv = carpeta_proyecto / "Base_de_datos.csv"
archivo_excel = carpeta_proyecto / "BD_creditos.xlsx"

# 3. Cargar la base original, sin modificar el archivo fuente.
# El archivo fuente se conserva intacto; todas las reglas de calidad se aplican
# a una copia en memoria y la salida se guarda en outputs/.
if archivo_csv.exists():
    datos = pd.read_csv(archivo_csv, low_memory=False)
    print(f"Base cargada desde: {archivo_csv}")
elif archivo_excel.exists():
    datos = pd.read_excel(archivo_excel)
    print(f"Base cargada desde: {archivo_excel}")
else:
    raise FileNotFoundError(
        "No se encontró Base_de_datos.csv ni BD_creditos.xlsx en la raíz del proyecto."
    )

# 4. Limpiar espacios accidentales en los nombres de las columnas.
datos.columns = datos.columns.astype(str).str.strip()

# 5. Revisar el tamaño de la base antes de limpiar.
filas_iniciales = len(datos)
print("\nTamaño original de la base:")
print(f"Filas: {datos.shape[0]}")
print(f"Columnas: {datos.shape[1]}")

# 6. Unificar representaciones textuales comunes de datos faltantes.
# No se convierte el número cero en faltante, porque puede ser un dato válido.
# La limpieza de texto evita tratar espacios o marcadores como categorías reales.
marcadores_nulos = ["", "NA", "N/A", "NaN", "nan", "NULL", "null", "None", "-", "--", "sin dato", "no informado"]
for columna in datos.select_dtypes(include=["object", "string"]).columns:
    datos[columna] = datos[columna].astype("string").str.strip()
    datos[columna] = datos[columna].replace(marcadores_nulos, np.nan)

# 7. Convertir la fecha a tipo fecha.
if "fecha_prestamo" in datos.columns:
    datos["fecha_prestamo"] = pd.to_datetime(datos["fecha_prestamo"], errors="coerce")

# 8. Convertir a numéricas las columnas que representan cantidades o puntajes.
columnas_numericas = [
    "capital_prestado", "plazo_meses", "edad_cliente", "salario_cliente",
    "total_otros_prestamos", "cuota_pactada", "puntaje", "puntaje_datacredito",
    "cant_creditosvigentes", "huella_consulta", "saldo_mora", "saldo_total",
    "saldo_principal", "saldo_mora_codeudor", "creditos_sectorFinanciero",
    "creditos_sectorCooperativo", "creditos_sectorReal",
    "promedio_ingresos_datacredito", "Pago_atiempo"
]
for columna in columnas_numericas:
    if columna in datos.columns:
        datos[columna] = pd.to_numeric(datos[columna], errors="coerce")

# 9. Aplicar la regla acordada para edades fuera del rango razonable del estudio.
# En el archivo revisado, las edades superiores a 100 son 121, 122 y 123 años.
# La regla se conserva como criterio del proyecto y debe revisarse si cambia la
# población objetivo; no representa un límite universal para cualquier negocio.
filas_edad_eliminadas = 0
edades_eliminadas_por_valor = {}
if "edad_cliente" in datos.columns:
    edad_invalida = (datos["edad_cliente"] < 0) | (datos["edad_cliente"] > 100)
    filas_edad_eliminadas = int(edad_invalida.sum())
    edades_eliminadas_por_valor = {
        str(edad): int(cantidad)
        for edad, cantidad in datos.loc[edad_invalida, "edad_cliente"].value_counts().sort_index().items()
    }
    datos = datos.loc[~edad_invalida].copy()
print(f"\nFilas eliminadas por edad fuera de 0 a 100: {filas_edad_eliminadas}")

# 10. Marcar como faltantes los puntajes que acordamos tratar como sospechosos.
# No se elimina toda la fila: solo se reemplaza el valor dudoso por NaN. Para
# puntaje_datacredito se usa el umbral acordado (<150); los valores >950 se
# conservan porque no se ha documentado una regla de negocio que los invalide.
# Ninguna variable monetaria se limita sin conocer primero moneda y definición.
valores_puntaje_negativos = 0
if "puntaje" in datos.columns:
    mascara = datos["puntaje"] < 0
    valores_puntaje_negativos = int(mascara.sum())
    datos.loc[mascara, "puntaje"] = np.nan
print(f"Valores negativos de puntaje convertidos a faltantes: {valores_puntaje_negativos}")

valores_datacredito_bajos = 0
if "puntaje_datacredito" in datos.columns:
    mascara = datos["puntaje_datacredito"] < 150
    valores_datacredito_bajos = int(mascara.sum())
    datos.loc[mascara, "puntaje_datacredito"] = np.nan
print(f"Puntajes Datacrédito inferiores a 150 convertidos a faltantes: {valores_datacredito_bajos}")

# 11. Revisar la variable objetivo sin imputarla ni modificar sus valores.
# Pago_atiempo debe ser 0/1 según la definición acordada para el proyecto.
if "Pago_atiempo" in datos.columns:
    valores_objetivo = sorted(datos["Pago_atiempo"].dropna().unique().tolist())
    print(f"Valores encontrados en Pago_atiempo: {valores_objetivo}")
    if not set(valores_objetivo).issubset({0, 1}):
        print("ADVERTENCIA: Pago_atiempo contiene valores diferentes de 0 y 1. Revisarlos.")

# 12. Mostrar el resumen de faltantes después de la limpieza.
print("\nValores faltantes por columna después de la limpieza:")
print(datos.isna().sum().sort_values(ascending=False).to_string())

# 13. Guardar la base limpia como CSV.
# Se guarda la fecha como texto legible y se conserva la base original intacta.
ruta_limpia = carpeta_salida / "BD_creditos_limpia.csv"
datos.to_csv(ruta_limpia, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d %H:%M:%S")

# 14. Guardar una bitácora simple con los cambios aplicados.
bitacora = {
    "filas_originales": int(filas_iniciales),
    "filas_finales": int(len(datos)),
    "filas_eliminadas_por_edad": int(filas_edad_eliminadas),
    "edades_eliminadas_por_valor": edades_eliminadas_por_valor,
    "valores_puntaje_negativos_convertidos_a_faltantes": int(valores_puntaje_negativos),
    "valores_datacredito_menores_150_convertidos_a_faltantes": int(valores_datacredito_bajos),
    "nulos_por_columna_despues_de_limpieza": {str(k): int(v) for k, v in datos.isna().sum().items()},
}
ruta_bitacora = carpeta_salida / "bitacora_limpieza.json"
ruta_bitacora.write_text(json.dumps(bitacora, ensure_ascii=False, indent=4), encoding="utf-8")

# 15. Confirmar dónde quedaron los archivos.
print("\nLimpieza inicial terminada.")
print(f"Base limpia: {ruta_limpia}")
