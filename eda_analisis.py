"""eda_analisis

>**Proyecto final de Machine Learning 2026 | Comprensión de los datos y análisis exploratorio (EDA): riesgo de incumplimiento en créditos**

La base tiene 10.763 créditos (de nov-2024 a abr-2026) y 23 columnas: datos del crédito (tipo, monto, plazo, cuota), del cliente (edad, tipo laboral, salario), su historial en Datacrédito (puntaje, saldos, consultas, créditos por sector) y la variable objetivo **Pago_atiempo** (1 = pagó a tiempo, 0 = incumplió).

### **1. Preprocesamiento**

#### **Librerías**
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from cargar_datos import datos as db

sns.set_theme(style='whitegrid')
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 200)

CARPETA_PROYECTO = Path(__file__).resolve().parent
CARPETA_GRAFICOS = CARPETA_PROYECTO / 'outputs' / 'graficos_eda'
CARPETA_GRAFICOS.mkdir(parents=True, exist_ok=True)


def guardar_grafica(nombre):
    """Guarda la figura actual y libera sus recursos."""
    ruta = CARPETA_GRAFICOS / nombre
    plt.savefig(ruta, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'Gráfica guardada: {ruta}')


"""#### **1.1 Carga y revisión inicial**

Se reutilizan los datos originales cargados por `cargar_datos.py`; este script
no vuelve a leer el Excel ni modifica el DataFrame original.
"""

print('Filas y columnas:', db.shape)
print(db.head())

resumen = pd.DataFrame({'tipo': db.dtypes.astype(str), 'valores únicos': db.nunique()})
print(resumen)

"""#### **1.2 Caracterización de las variables**

Clasificamos cada variable por su naturaleza:

| Variable | Naturaleza | Comentario |
|---|---|---|
| `tipo_credito` | Nominal politómica | Llega como número (4, 6, 7, 9, 10, 68) pero es una categoría, no una cantidad |
| `tipo_laboral` | Nominal dicotómica | Empleado / Independiente |
| `tendencia_ingresos` | Ordinal | Decreciente < Estable < Creciente |
| `Pago_atiempo` | Dicotómica (objetivo) | 1 = paga a tiempo, 0 = incumple |
| `fecha_prestamo` | Fecha | Llega como texto |
| `capital_prestado`, `salario_cliente`, `total_otros_prestamos`, `cuota_pactada`, `saldo_*`, `promedio_ingresos_datacredito` | Numéricas continuas (montos) | |
| `plazo_meses`, `edad_cliente`, `cant_creditosvigentes`, `huella_consulta`, `creditos_sector*` | Numéricas discretas (conteos) | |
| `puntaje`, `puntaje_datacredito` | Numéricas (puntajes) | `puntaje` es sospechosa, se revisa en la sección 3 |

#### **1.3 Valores nulos**

Primero miramos cuántos nulos hay y luego buscamos "falsos nulos" (texto como *NULL*, *N/A* o *-*) que pandas no cuenta como nulos.
"""

nulos = pd.DataFrame({'nulos': db.isnull().sum(), '% nulos': (db.isnull().mean() * 100).round(2)})
print(nulos[nulos['nulos'] > 0])

marcadores = ['', 'NA', 'N/A', 'NULL', 'null', 'None', '-', '--', 'sin dato', 'no informado']

print('Falsos nulos (texto) por columna de texto:')
for col in db.select_dtypes(exclude='number').columns:
    print(col, ':', db[col].astype(str).str.strip().isin(marcadores).sum())

"""No hay falsos nulos: los faltantes ya vienen como nulos reales. Se concentran en `promedio_ingresos_datacredito` y `tendencia_ingresos` (~27%) y en los saldos (1% a 5%). Revisamos si los faltantes siguen algún patrón o son aleatorios."""

print('Faltan promedio de ingresos y tendencia a la vez:',
      (db['promedio_ingresos_datacredito'].isnull() & db['tendencia_ingresos'].isnull()).sum())
print()
print('% sin promedio de ingresos según tipo laboral:')
print((db.groupby('tipo_laboral')['promedio_ingresos_datacredito'].apply(lambda s: s.isnull().mean()) * 100).round(1))

"""Los dos faltan casi siempre juntos (2.930 casos) y faltan mucho más en independientes (41%) que en empleados (19%). Esto parece falta de historial de ingresos reportado y no azar.

#### **1.4 Revisión de inconsistencias**

Revisamos cada variable buscando valores imposibles o raros.

**Edad (edad_cliente).**
"""

print(db['edad_cliente'].describe(), '\n')
print('Edades más altas:')
print(db['edad_cliente'].value_counts().sort_index().tail(5))
print('\nEdades mayores a 100:', (db['edad_cliente'] > 100).sum())

"""Hay **150 clientes con 121, 122 o 123 años** (144 con exactamente 122). Es imposible, parece un valor de relleno o un error de digitación. Las edades normales llegan hasta ~69.

**Salario (salario_cliente).**
"""

# Resumen estadístico básico
salarios = db['salario_cliente']

print("Resumen estadístico")
print(f"Cantidad total de registros: {len(salarios)}")
print(f"Media (Promedio):           {salarios.mean():,.2f}")
print(f"Mediana:                    {salarios.median():,.2f}")
print(f"Desviación Estándar:        {salarios.std():,.2f}")
print(f"Mínimo:                     {salarios.min():,.2f}")
print(f"Máximo:                     {salarios.max():,.2f}")

# Percentiles clave para identificar el rango común
percentiles = salarios.quantile([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])

print("Percentiles de salario_cliente")
for p, val in percentiles.items():
    print(f"Percentil {int(p*100)}%: {val:,.2f}")

# Rango intercuartílico (IQR)
q25 = percentiles[0.25]
q75 = percentiles[0.75]
iqr = q75 - q25

print(f"\nRango Intercuartílico (IQR): {iqr:,.2f} ")
print(f"Rango Central 50% de los datos: {q25:,.2f} a {q75:,.2f} ")
print(f"Rango Típico (Percentil 10 a 90): {percentiles[0.10]:,.2f} a {percentiles[0.90]:,.2f} ")

# Visualización del rango común
salario_filtrado = salarios[salarios.between(salarios.quantile(0.01), salarios.quantile(0.95))]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histograma
sns.histplot(salario_filtrado, kde=True, ax=axes[0], color='skyblue')
axes[0].set_title('Distribución del Salario (Percentil 1 a 95)')
axes[0].set_xlabel('Salario (unidad por confirmar)')
axes[0].set_ylabel('Frecuencia')

# Boxplot
sns.boxplot(x=salario_filtrado, ax=axes[1], color='lightgreen')
axes[1].set_title('Boxplot del Salario (Percentil 1 a 95)')
axes[1].set_xlabel('Salario (unidad por confirmar)')

plt.tight_layout()
guardar_grafica('salario_distribucion.png')

# 1. Registros en cero
zeros_count = (salarios == 0).sum()
print(f"Registros con salario igual a 0: {zeros_count} ({zeros_count/len(db):.2%})")

# 2. Valores altos para inspección; la unidad monetaria no está documentada.
extremos = db[db['salario_cliente'] > 1e9][['salario_cliente', 'promedio_ingresos_datacredito', 'capital_prestado', 'edad_cliente', 'tipo_laboral']]
print(f"\nRegistros con salario > 1.000 millones (unidad por confirmar): {len(extremos)}")
print("\nMuestra de registros atípicos superiores:")
print(extremos.head(10))

"""La variable presenta valores cero y algunos valores muy altos. Los percentiles
describen la distribución, pero no permiten decidir por sí solos si un registro
es inválido. Sin diccionario de datos ni unidad confirmada, se conservan para
investigación y no se fija un límite monetario en esta etapa.

**Puntaje de Datacrédito (puntaje_datacredito).**
"""

print(db['puntaje_datacredito'].describe().round(1), '\n')
print('Valores menores a 150:')
print(db[db['puntaje_datacredito'] < 150]['puntaje_datacredito'].value_counts())
print('\nValores mayores a 950:', (db['puntaje_datacredito'] > 950).sum(), '| máximo:', db['puntaje_datacredito'].max())

"""El puntaje observado va de -7 a 999. Los valores menores a 150 se investigan
y se marcan como faltantes según el criterio acordado; los valores mayores a 950
se conservan porque no hay una regla confirmada que los invalide.

**Tendencia de ingresos (tendencia_ingresos).**
"""

print(db['tendencia_ingresos'].value_counts(dropna=False).head(8))

categorias = ['Creciente', 'Estable', 'Decreciente']
basura = db[db['tendencia_ingresos'].notna() & ~db['tendencia_ingresos'].isin(categorias)]
print('\nValores que no son una tendencia:', len(basura))
print('Ejemplos:', basura['tendencia_ingresos'].head(10).tolist())

"""Esta columna debería tener solo 3 categorías (Creciente, Estable, Decreciente), pero hay 58 registros con números (8315, 0, 158042, -28589...).

**Tipo de crédito, montos y fechas.**
"""

print(db['tipo_credito'].value_counts(), '\n')
print('Créditos de tipos con menos de 25 casos:', db['tipo_credito'].isin([6, 7, 68]).sum(), '\n')
print('Mayores valores de total_otros_prestamos:', db['total_otros_prestamos'].nlargest(5).tolist())
print('Registros con total_otros_prestamos > 1.000 millones:', (db['total_otros_prestamos'] > 1_000_000_000).sum(), '\n')
fechas = pd.to_datetime(db['fecha_prestamo'])
print('Rango de fechas:', fechas.min(), '->', fechas.max())

"""- `tipo_credito`: predominan los tipos 4 y 9; otros tipos tienen pocos casos y sus tasas deben interpretarse con cautela.
- `total_otros_prestamos`: aparecen valores numéricos muy altos. Sin unidad confirmada, se reportan para revisión y no se tratan como errores automáticamente.
- Las fechas son coherentes (ninguna fecha imposible).

#### **1.5 Duplicados**
"""

print('Filas duplicadas:', db.duplicated().sum())

"""No hay filas duplicadas, no hay nada que eliminar.

### **2. Limpieza**

Como base general, los valores imposibles se convierten en nulo y no se elimina la fila ya que, si se borra la fila, se pierde la información de las otras 22 columnas, y además en producción el modelo tiene que poder puntuar cualquier solicitud que llegue.

Guardamos en `registro` cuántos valores afecta cada regla.
"""

datos = db.copy()
registro = []

def invalidar(columna, mascara, regla):
    registro.append((regla, int(mascara.sum())))
    datos.loc[mascara, columna] = np.nan

"""Pasamos la fecha a tipo fecha y `tipo_credito` a categoría (es un código, no una cantidad)."""

datos['fecha_prestamo'] = pd.to_datetime(datos['fecha_prestamo'])
datos['tipo_credito'] = datos['tipo_credito'].astype(str)
print(datos.dtypes[['fecha_prestamo', 'tipo_credito']])

"""Edades negativas o superiores a 100 años pasan a nulo según el criterio
acordado para este proyecto.
"""

invalidar('edad_cliente', (datos['edad_cliente'] < 0) | (datos['edad_cliente'] > 100), 'edad fuera de 0-100')
print(datos['edad_cliente'].describe().round(1))

"""No se modifica `salario_cliente`: no se cuenta con una unidad monetaria ni
con límites de negocio confirmados para decidir qué valores son inválidos.

Los ceros y extremos se reportan y se investigan, pero no se convierten
automáticamente en faltantes.
"""

"""Puntajes de Datacrédito menores a 150 pasan a nulo; los mayores a 950 se
conservan porque no se ha establecido que sean inválidos.
"""

invalidar('puntaje_datacredito', datos['puntaje_datacredito'] < 150, 'puntaje_datacredito menor a 150')
print(datos['puntaje_datacredito'].describe().round(1))

"""Los valores de `tendencia_ingresos` que no son Creciente, Estable o Decreciente pasan a nulo."""

invalidar('tendencia_ingresos', datos['tendencia_ingresos'].notna() & ~datos['tendencia_ingresos'].isin(categorias), 'tendencia_ingresos no es una categoría')
print(datos['tendencia_ingresos'].value_counts(dropna=False))

"""Resumen de lo que se convirtió en nulo. Ninguna fila se eliminó (seguimos con 10.763) y la base original `db` queda intacta."""

print('Filas:', len(datos))
print(pd.DataFrame(registro, columns=['Regla', 'Valores convertidos a nulo']))

"""### **3. Selección de variables**

Antes de analizar, revisamos si alguna variable no debería usarse para predecir. Ninguna es un identificador ni una constante, pero puntaje casi explica por sí sola la variable objetivo.
"""

print('Correlación de puntaje con Pago_atiempo:', round(db['puntaje'].corr(db['Pago_atiempo']), 3), '\n')

"""La correlación alta de `puntaje` con el objetivo es una señal de alerta, no
demuestra por sí sola fuga de información. Se debe confirmar cuándo se calcula
el puntaje y si estaría disponible al aprobar el crédito. Hasta validarlo, se
excluye preventivamente del conjunto de modelado.

La fecha se conserva durante el EDA, pero se excluye del conjunto final de
predictores en esta etapa; si se necesitan patrones temporales, se deben crear
variables derivadas y validarlas con una partición temporal.
"""

datos = datos.drop(columns='puntaje')
print('Columnas:', datos.shape[1])

"""### **4. Análisis univariable**"""

numericas = ['capital_prestado', 'plazo_meses', 'edad_cliente', 'salario_cliente', 'total_otros_prestamos',
             'cuota_pactada', 'puntaje_datacredito', 'cant_creditosvigentes', 'huella_consulta', 'saldo_mora',
             'saldo_total', 'saldo_principal', 'saldo_mora_codeudor', 'creditos_sectorFinanciero',
             'creditos_sectorCooperativo', 'creditos_sectorReal', 'promedio_ingresos_datacredito']
categoricas = ['tipo_credito', 'tipo_laboral', 'tendencia_ingresos']

"""#### **4.1 Variables numéricas**

Con `describe()` y otras medidas de tendencia central y dispersión (moda, rango, IQR, varianza, asimetría y curtosis).
"""

print(datos[numericas].describe().T.round(1))

estadisticos = pd.DataFrame({
    'moda': datos[numericas].mode().iloc[0],
    'rango': datos[numericas].max() - datos[numericas].min(),
    'IQR': datos[numericas].quantile(0.75) - datos[numericas].quantile(0.25),
    'varianza': datos[numericas].var(),
    'asimetría': datos[numericas].skew(),
    'curtosis': datos[numericas].kurtosis(),
    '% ceros': (datos[numericas] == 0).mean() * 100
})
print(estadisticos.round(1))

"""Lo que se ve:
- Montos, saldos y otras variables presentan asimetría positiva: muchos valores bajos y pocos muy altos. No son gaussianas.
- `saldo_mora` (98% de ceros) y `saldo_mora_codeudor` (94%) son casi constantes en 0: solo unos pocos clientes tienen mora.
- `creditos_sectorCooperativo` (82% ceros) y `creditos_sectorReal` (41%) también tienen muchos ceros.
- `edad_cliente` es la única con asimetría cercana a 0 (0,3) y `puntaje_datacredito` es parecida a una campana con cola hacia los valores bajos (-0,7).
"""

fig, axes = plt.subplots(4, 5, figsize=(20, 14))
for ax, col in zip(axes.ravel(), numericas):
    serie = datos[col].dropna()
    sns.histplot(serie.clip(upper=serie.quantile(0.99)), bins=30, ax=ax, color='steelblue')
    ax.set_title(col, fontsize=10)
    ax.set_xlabel('')
for ax in axes.ravel()[len(numericas):]:
    ax.axis('off')
plt.suptitle('Histogramas (recortados al percentil 99 para poder ver la forma)', fontsize=14)
plt.tight_layout()
guardar_grafica('histogramas_numericas.png')

"""Tipo de distribución aproximado:
- **Asimétrica a la derecha (tipo log-normal):** `capital_prestado`, `cuota_pactada`, `salario_cliente`, `total_otros_prestamos`, `saldos`, `promedio_ingresos_datacredito`.
- **Conteos con cola a la derecha:** `plazo_meses`, `cant_creditosvigentes`, `huella_consulta`, `creditos_sector*`.
- **Casi simétrica:** `edad_cliente` (entre 19 y 69) y `puntaje_datacredito`.
- **Concentrada en cero:** `saldo_mora` y `saldo_mora_codeudor`.
"""

fig, axes = plt.subplots(4, 5, figsize=(20, 12))
for ax, col in zip(axes.ravel(), numericas):
    sns.boxplot(x=datos[col], ax=ax, color='lightblue')
    ax.set_title(col, fontsize=10)
    ax.set_xlabel('')
for ax in axes.ravel()[len(numericas):]:
    ax.axis('off')
plt.tight_layout()
guardar_grafica('boxplots_numericas.png')

"""**Valores atípicos.** Contamos cuántos marca la regla del IQR (1,5 veces el rango intercuartil) y la del Z-score (mayor a 3)."""

def contar_atipicos(serie):
    s = serie.dropna()
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    por_iqr = ((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum()
    por_z = (((s - s.mean()) / s.std()).abs() > 3).sum()
    return pd.Series({'atípicos IQR': por_iqr, 'atípicos Z>3': por_z})

print(datos[numericas].apply(contar_atipicos).T)

"""Las dos reglas marcan muchos valores en varias variables, pero **no se eliminan**:
- En variables con muchos ceros (`saldo_mora`, `saldo_mora_codeudor`) el IQR es 0, entonces cualquier valor distinto de 0 queda como "atípico". Son justo los clientes en mora, que es la señal más importante.
- Sin unidades y reglas de negocio documentadas no se puede afirmar que un valor alto sea dinero real o inválido. No se aplican transformaciones ni topes en este EDA.

#### **4.2 Variables categóricas**
"""

for col in categoricas:
    frec = datos[col].fillna('Faltante').value_counts()
    print(pd.DataFrame({'n': frec, '%': (frec / len(datos) * 100).round(1)}), '\n')

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for ax, col in zip(axes, categoricas):
    sns.countplot(x=datos[col].fillna('Faltante'), ax=ax, color='steelblue')
    ax.set_title(col)
    ax.set_xlabel('')
plt.tight_layout()
guardar_grafica('frecuencias_categoricas.png')

"""- `tipo_credito`: dominada por los tipos 4 (72%) y 9 (27%). El resto casi no tiene casos.
- `tipo_laboral`: 63% empleados y 37% independientes.
- `tendencia_ingresos`: 49% Creciente, 12% Decreciente, 11% Estable y 28% faltante (2.990, incluyendo los 58 que antes eran basura).

### **5. Análisis bivariable (contra la variable objetivo)**

#### **5.1 Variable objetivo**
"""

print(datos['Pago_atiempo'].value_counts())
print((datos['Pago_atiempo'].value_counts(normalize=True) * 100).round(2))

sns.countplot(x='Pago_atiempo', data=datos, color='steelblue')
plt.title('Distribución de Pago_atiempo')
guardar_grafica('distribucion_pago_atiempo.png')

"""Solo el **4,75%** de los créditos (511) incumple. La clase está muy desbalanceada: un modelo que siempre diga "paga a tiempo" tendría 95% de acierto sin servir para nada. Por eso no conviene evaluar con *accuracy* sino con recall, precisión y F1 de la clase de incumplimiento.

#### **5.2 Variables numéricas según el objetivo**
"""

print(datos.groupby('Pago_atiempo')[numericas].median().T.round(1))

columnas = ['puntaje_datacredito', 'huella_consulta', 'plazo_meses', 'edad_cliente',
            'cant_creditosvigentes', 'saldo_total', 'capital_prestado', 'promedio_ingresos_datacredito']
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
for ax, col in zip(axes.ravel(), columnas):
    visible = datos[datos[col] <= datos[col].quantile(0.99)]
    sns.boxplot(data=visible, x='Pago_atiempo', y=col, ax=ax)
    ax.set_title(col)
plt.tight_layout()
guardar_grafica('variables_numericas_por_pago.png')

"""Las medianas de ambos grupos son muy parecidas en casi todas las variables. Las diferencias más claras: quienes incumplen tienen **menor puntaje de Datacrédito** (mediana 769 vs 792), **más consultas** (5 vs 4), **menor edad** (37 vs 42 años) y piden **montos algo más altos** (mediana 2,27 vs 1,91 millones). Se necesita mirar por rangos para ver más.

#### **5.3 Tasa de pago a tiempo por grupos**
"""

def tasa_pago(columna, grupos=None):
    agrupador = datos[columna].fillna('Faltante') if grupos is None else pd.cut(datos[columna], grupos)
    return datos.groupby(agrupador, observed=True)['Pago_atiempo'].agg(clientes='count', tasa_pago='mean').round(3)

for col in ['tipo_laboral', 'tendencia_ingresos', 'tipo_credito']:
    print(f'\nTasa de pago por {col}:')
    print(tasa_pago(col))

for col, grupos in [
    ('puntaje_datacredito', [0, 700, 750, 800, 850, 1000]),
    ('huella_consulta', [-1, 2, 5, 10, 50]),
    ('plazo_meses', [0, 6, 12, 24, 100]),
    ('edad_cliente', [17, 30, 40, 50, 60, 101]),
]:
    print(f'\nTasa de pago por rangos de {col}:')
    print(tasa_pago(col, grupos))

print('Con mora (saldo_mora > 0) vs sin mora:')
print(datos.groupby(datos['saldo_mora'].fillna(0) > 0)['Pago_atiempo'].agg(clientes='count', tasa_pago='mean').round(3))

"""Cada grupo se compara con el promedio general (95,3%):
- **Puntaje de Datacrédito:** sube con el puntaje (85% si es 700 o menos, 97% si es más de 850).
- **Consultas:** baja con las consultas (97% con 2 o menos, 90% con más de 10).
- **Mora:** los 55 clientes con mora pagan a tiempo solo 64%.
- **Plazo:** los créditos a más de 24 meses pagan a tiempo 87% (el resto 93% a 96%).
- **Edad:** los menores de 30 pagan 93% y los demás 95% a 96%.
- **Tendencia de ingresos:** Creciente 96,1%, Decreciente 93,7%, la diferencia es pequeña pero en el sentido esperado.
- **Tipo laboral:** casi no separa (95,7% empleados vs 94,5% independientes).
- **Tipo de crédito:** el tipo 6 paga solo 57%, pero son 21 casos, no hay evidencia suficiente.

Las señales son **débiles pero coherentes**. Hay que tener presente que los grupos tienen tamaños muy distintos.

### **6. Análisis multivariable**

#### **6.1 Correlación entre variables**
"""

correlacion = datos[numericas + ['Pago_atiempo']].corr()

plt.figure(figsize=(14, 11))
sns.heatmap(correlacion, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5, annot_kws={'size': 7})
plt.title('Matriz de correlación')
guardar_grafica('matriz_correlacion.png')

print('Correlación con la variable objetivo:')
print(correlacion['Pago_atiempo'].drop('Pago_atiempo').sort_values(key=abs, ascending=False).round(3).head(8))

pares = correlacion.drop(columns='Pago_atiempo').drop(index='Pago_atiempo').abs()
pares = pares.where(np.triu(np.ones(pares.shape, dtype=bool), 1)).stack().sort_values(ascending=False)
print('\nPares de variables más correlacionados:')
print(pares.head(5).round(2))

"""- **Con el objetivo:** ninguna variable pasa de 0,12 de correlación (la mayor es `puntaje_datacredito`, 0,12). Es un problema difícil y se espera un desempeño modesto del modelo.
- **Entre variables:** los pares más altos son `cant_creditosvigentes` con `creditos_sectorFinanciero` (0,79), `capital_prestado` con `cuota_pactada` (0,76, normal: la cuota depende del capital) y `saldo_total` con `saldo_principal` (0,73).

### **8. Transformación de variables y reducción de redundancia**
Como preparación exploratoria, se excluyen algunas columnas y se codifican las categorías. Las decisiones deben validarse durante el entrenamiento.
"""

# 8.1 Exclusión preventiva y reducción de redundancia
# puntaje se excluye hasta confirmar su momento de disponibilidad. La fecha
# cruda no se usa como predictor; se podrán evaluar variables temporales después.
cols_a_eliminar = [
    'puntaje',
    'fecha_prestamo',
    'cuota_pactada',
    'saldo_principal',
    'creditos_sectorFinanciero',
]

# Creamos una copia del dataset procesado
datos_modelo = datos.drop(columns=[col for col in cols_a_eliminar if col in datos.columns]).copy()

print(f"Shape original: {datos.shape}")
print(f"Shape tras eliminar variables altamente correlacionadas: {datos_modelo.shape}")
print("\nVariables eliminadas:")
for c in cols_a_eliminar:
    print(f" - {c}")

# 8.2 Codificación exploratoria (sin imputación)

# 1. Corrección y depuración de valores anómalos en variables categóricas
datos_modelo['tendencia_ingresos'] = datos_modelo['tendencia_ingresos'].apply(
    lambda x: x if str(x) in ['Creciente', 'Estable', 'Decreciente'] else np.nan
)

# 2. Separación de variables categóricas y numéricas
var_categoricas = ['tipo_credito', 'tipo_laboral', 'tendencia_ingresos']
var_numericas = [c for c in datos_modelo.columns if c not in var_categoricas + ['fecha_prestamo', 'Pago_atiempo']]

# 3. Codificación de Variables Categóricas (One-Hot Encoding)
datos_encoded = pd.get_dummies(datos_modelo, columns=var_categoricas, drop_first=True, dummy_na=True)

print("Base de datos final procesada y codificada:")
print(f"Dimensiones finales: {datos_encoded.shape}")
print("\nValores faltantes que todavía requieren tratamiento en entrenamiento:")
print(datos_encoded.isna().sum().sort_values(ascending=False).head(10))
print(datos_encoded.head())

"""### **9. Conclusiones y Resumen del Proceso**

1. **Gestión de calidad de datos y anomalías:**
   - Se marcaron como faltantes las edades fuera del rango acordado, los puntajes Datacrédito menores a 150 y las categorías no reconocidas de `tendencia_ingresos`. No se impusieron límites monetarios sin unidades o reglas confirmadas.

2. **Revisión de posible fuga de información:**
   - `puntaje` presenta una correlación alta con `Pago_atiempo`; se excluye preventivamente hasta confirmar su definición y disponibilidad al otorgar el crédito. La correlación por sí sola no demuestra fuga ni causalidad.

3. **Eliminación de multicolinealidad (> 0.70):**
   - Se retiraron algunas variables por redundancia observada (`cuota_pactada`, `saldo_principal` y `creditos_sectorFinanciero`). Esta selección debe revisarse con conocimiento del negocio y durante el entrenamiento.

4. **Transformación y codificación (*Encoding*):**
   - Se depuraron incongruencias en `tendencia_ingresos` y se aplicó One-Hot Encoding como exploración. La imputación, selección y codificación finales deben ajustarse únicamente con datos de entrenamiento; este resultado no es aún un pipeline listo para producción.

"""