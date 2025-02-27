# Manejo de datos con Pandas

## ¿Qué es Pandas y por qué necesitamos aprenderlo?

Pandas es una biblioteca de Python diseñada para el análisis y manipulación de datos estructurados. Si NumPy es bueno con arrays numéricos, Pandas es excelente con datos tabulares (como hojas de cálculo o tablas SQL).

En el mundo de la IA y el aprendizaje automático, Pandas es crucial porque:

- La mayoría de los datasets vienen en formatos tabulares (CSV, Excel, etc.)
- Necesitamos limpiar, transformar y preparar los datos antes de entrenar modelos
- Pandas facilita el análisis exploratorio para entender qué contienen los datos
- Integra perfectamente con NumPy, Matplotlib y scikit-learn

## Instalación

Si no lo has instalado:

```bash
pip install pandas
```

## Importación

La convención es importar Pandas como `pd`:

```python
import pandas as pd
import numpy as np  # A menudo usamos NumPy junto con Pandas
```

## Estructuras de datos principales

Pandas tiene dos estructuras fundamentales:

### 1. Series

Una **Series** es como un array unidimensional pero con etiquetas (índices):

```python
# Crear una Series
s = pd.Series([10, 20, 30, 40])
print(s)
# 0    10
# 1    20
# 2    30
# 3    40
# dtype: int64

# Series con índices personalizados
s = pd.Series([10, 20, 30, 40], index=['a', 'b', 'c', 'd'])
print(s)
# a    10
# b    20
# c    30
# d    40
# dtype: int64

# Series desde un diccionario
diccionario = {'a': 10, 'b': 20, 'c': 30, 'd': 40}
s = pd.Series(diccionario)
print(s)
# a    10
# b    20
# c    30
# d    40
# dtype: int64
```

### 2. DataFrame

Un **DataFrame** es como una tabla o una hoja de cálculo:

```python
# Crear un DataFrame desde un diccionario
datos = {
    'Nombre': ['Ana', 'Juan', 'María', 'Pedro'],
    'Edad': [25, 30, 22, 35],
    'Ciudad': ['Madrid', 'Barcelona', 'Valencia', 'Sevilla']
}
df = pd.DataFrame(datos)
print(df)
#   Nombre  Edad     Ciudad
# 0    Ana    25     Madrid
# 1   Juan    30  Barcelona
# 2  María    22   Valencia
# 3  Pedro    35    Sevilla

# DataFrame desde un array NumPy
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
df = pd.DataFrame(arr, columns=['A', 'B', 'C'])
print(df)
#    A  B  C
# 0  1  2  3
# 1  4  5  6
# 2  7  8  9
```

## Lectura y escritura de datos

Una de las grandes ventajas de Pandas es la facilidad para leer y escribir datos en diferentes formatos:

```python
# Leer un archivo CSV
df = pd.read_csv('datos.csv')

# Leer un archivo Excel
df = pd.read_excel('datos.xlsx', sheet_name='Hoja1')

# Guardar a CSV
df.to_csv('resultado.csv', index=False)

# Guardar a Excel
df.to_excel('resultado.xlsx', index=False)
```

## Exploración de datos

Cuando trabajamos con un nuevo dataset, lo primero es explorarlo:

```python
# Crear un DataFrame de ejemplo (ventas de productos)
datos = {
    'Producto': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'],
    'Categoria': ['Electrónica', 'Ropa', 'Hogar', 'Electrónica', 
                 'Ropa', 'Hogar', 'Electrónica', 'Ropa'],
    'Precio': [100, 50, 30, 200, 80, 40, 150, 60],
    'Ventas': [150, 200, 100, 50, 300, 120, 80, 240],
    'Fecha': pd.date_range(start='2023-01-01', periods=8, freq='M')
}

df = pd.DataFrame(datos)

# Primeras filas
print(df.head())

# Últimas filas
print(df.tail())

# Información general
print(df.info())

# Estadísticas descriptivas
print(df.describe())

# Dimensiones (filas, columnas)
print(f"Dimensiones: {df.shape}")

# Nombres de columnas
print(f"Columnas: {df.columns}")

# Tipos de datos
print(f"Tipos de datos:\n{df.dtypes}")

# Valores únicos en una columna
print(f"Categorías únicas: {df['Categoria'].unique()}")
print(f"Número de categorías: {df['Categoria'].nunique()}")

# Conteo de valores
print(df['Categoria'].value_counts())
```

## Selección y filtrado de datos

### Selección de columnas

```python
# Una columna (devuelve una Series)
precios = df['Precio']
print(precios)

# Varias columnas (devuelve un DataFrame)
seleccion = df[['Producto', 'Precio', 'Ventas']]
print(seleccion)
```

### Selección de filas

```python
# Por posición usando iloc (integer location)
primera_fila = df.iloc[0]  # Primera fila
print(primera_fila)

tres_filas = df.iloc[0:3]  # Primeras tres filas
print(tres_filas)

# Por etiqueta usando loc
fila_por_indice = df.loc[2]  # Fila con índice 2
print(fila_por_indice)
```

### Filtrado con condiciones

```python
# Productos con precio > 100
caros = df[df['Precio'] > 100]
print(caros)

# Productos de Electrónica
electronica = df[df['Categoria'] == 'Electrónica']
print(electronica)

# Combinación de condiciones (AND)
filtro_and = df[(df['Precio'] > 50) & (df['Ventas'] > 100)]
print(filtro_and)

# Combinación de condiciones (OR)
filtro_or = df[(df['Categoria'] == 'Electrónica') | (df['Precio'] > 70)]
print(filtro_or)
```

## Manejo de valores faltantes

Los datos del mundo real suelen tener valores faltantes:

```python
# Crear un DataFrame con algunos valores NaN
datos = {
    'A': [1, 2, np.nan, 4, 5],
    'B': [np.nan, 2, 3, 4, 5],
    'C': [1, 2, 3, np.nan, 5]
}
df_nan = pd.DataFrame(datos)
print(df_nan)

# Detectar valores NaN
print(df_nan.isna())  # Devuelve True donde hay NaN
print(df_nan.isna().sum())  # Cuenta NaN por columna

# Eliminar filas con NaN
df_sin_nan = df_nan.dropna()
print(df_sin_nan)

# Eliminar columnas con NaN
df_columnas_completas = df_nan.dropna(axis=1)
print(df_columnas_completas)

# Rellenar NaN con un valor
df_rellenado = df_nan.fillna(0)
print(df_rellenado)

# Rellenar NaN con la media de la columna
df_media = df_nan.copy()
df_media['A'] = df_media['A'].fillna(df_media['A'].mean())
df_media['B'] = df_media['B'].fillna(df_media['B'].mean())
df_media['C'] = df_media['C'].fillna(df_media['C'].mean())
print(df_media)
```

## Operaciones con datos

### Añadir y eliminar columnas

```python
# Añadir una columna calculada
df['Ingresos'] = df['Precio'] * df['Ventas']
print(df)

# Eliminar columnas
df_reducido = df.drop('Ingresos', axis=1)
print(df_reducido)
```

### Operaciones por grupos

Una de las operaciones más potentes es `groupby`:

```python
# Agrupar por categoría y calcular media de precio y ventas
por_categoria = df.groupby('Categoria')[['Precio', 'Ventas']].mean()
print(por_categoria)

# Agrupar y contar
conteo = df.groupby('Categoria').size()
print(conteo)

# Múltiples operaciones
resumen = df.groupby('Categoria').agg({
    'Precio': ['min', 'max', 'mean'],
    'Ventas': ['sum', 'mean']
})
print(resumen)
```

### Ordenar datos

```python
# Ordenar por precio (ascendente)
df_ordenado = df.sort_values('Precio')
print(df_ordenado)

# Ordenar por múltiples columnas (precio descendente, ventas ascendente)
df_ordenado_multi = df.sort_values(['Categoria', 'Precio'], 
                                  ascending=[True, False])
print(df_ordenado_multi)
```

## Transformación de datos

### Aplicar funciones a datos

```python
# Aplicar una función a una columna
df['Precio_Descuento'] = df['Precio'].apply(lambda x: x * 0.9)
print(df)

# Aplicar una función a filas
def rentabilidad(row):
    if row['Ventas'] > 200:
        return 'Alta'
    elif row['Ventas'] > 100:
        return 'Media'
    else:
        return 'Baja'

df['Rentabilidad'] = df.apply(rentabilidad, axis=1)
print(df)
```

### Pivoteo de tablas

```python
# Crear una tabla pivot
pivot = df.pivot_table(
    values='Ventas',
    index='Categoria',
    columns='Rentabilidad',
    aggfunc='sum'
)
print(pivot)
```

## Unión de DataFrames

### Concatenación

```python
# Crear dos DataFrames
df1 = pd.DataFrame({
    'A': ['A0', 'A1', 'A2'],
    'B': ['B0', 'B1', 'B2']
})

df2 = pd.DataFrame({
    'A': ['A3', 'A4', 'A5'],
    'B': ['B3', 'B4', 'B5']
})

# Concatenar verticalmente
resultado_v = pd.concat([df1, df2])
print(resultado_v)

# Concatenar horizontalmente
resultado_h = pd.concat([df1, df2], axis=1)
print(resultado_h)
```

### Merge (similar a JOIN en SQL)

```python
# Crear DataFrames para combinar
clientes = pd.DataFrame({
    'cliente_id': [1, 2, 3, 4, 5],
    'nombre': ['Ana', 'Juan', 'María', 'Pedro', 'Luis']
})

compras = pd.DataFrame({
    'compra_id': [101, 102, 103, 104, 105],
    'cliente_id': [1, 2, 3, 1, 2],
    'producto': ['A', 'B', 'C', 'D', 'E'],
    'monto': [100, 200, 300, 150, 250]
})

# Inner join (sólo coincidencias)
inner = pd.merge(clientes, compras, on='cliente_id')
print(inner)

# Left join (todos los clientes, incluso sin compras)
left = pd.merge(clientes, compras, on='cliente_id', how='left')
print(left)

# Right join (todas las compras, incluso de clientes no registrados)
right = pd.merge(clientes, compras, on='cliente_id', how='right')
print(right)

# Outer join (todos los registros de ambas tablas)
outer = pd.merge(clientes, compras, on='cliente_id', how='outer')
print(outer)
```

## Caso práctico: Análisis de datos de ventas

Veamos un ejemplo completo de análisis de datos:

```python
# Crear un DataFrame más completo de ventas
np.random.seed(42)  # Para reproducibilidad

# Generar datos
n = 1000  # número de registros
productos = ['Laptop', 'Smartphone', 'Tablet', 'Auriculares', 'Monitor']
categorias = ['Electrónica', 'Electrónica', 'Electrónica', 'Accesorios', 'Electrónica']
tiendas = ['Norte', 'Sur', 'Este', 'Oeste', 'Centro']

datos_ventas = {
    'fecha': pd.date_range(start='2023-01-01', periods=n),
    'producto': np.random.choice(productos, n),
    'tienda': np.random.choice(tiendas, n),
    'unidades': np.random.randint(1, 10, n),
    'precio_unitario': np.random.randint(50, 500, n)
}

# Crear DataFrame
df_ventas = pd.DataFrame(datos_ventas)

# Agregar columna de ingresos
df_ventas['ingresos'] = df_ventas['unidades'] * df_ventas['precio_unitario']

# Exploración inicial
print("Dimensiones del dataset:", df_ventas.shape)
print("\nPrimeros registros:")
print(df_ventas.head())
print("\nEstadísticas descriptivas:")
print(df_ventas.describe())

# Añadir columnas de tiempo para análisis
df_ventas['año'] = df_ventas['fecha'].dt.year
df_ventas['mes'] = df_ventas['fecha'].dt.month
df_ventas['dia_semana'] = df_ventas['fecha'].dt.day_name()

# Análisis por producto
ventas_por_producto = df_ventas.groupby('producto').agg({
    'unidades': 'sum',
    'ingresos': 'sum'
}).sort_values('ingresos', ascending=False)

print("\nVentas por producto:")
print(ventas_por_producto)

# Análisis por tienda y mes
ventas_mensuales = df_ventas.groupby(['tienda', 'mes']).agg({
    'ingresos': 'sum'
}).reset_index()

print("\nIngresos mensuales por tienda (primeros 10 registros):")
print(ventas_mensuales.head(10))

# Productos más vendidos por tienda
top_productos = df_ventas.groupby(['tienda', 'producto']).agg({
    'unidades': 'sum'
}).reset_index()

# Para cada tienda, encontrar el producto más vendido
mejor_producto = top_productos.loc[top_productos.groupby('tienda')['unidades'].idxmax()]
print("\nProducto más vendido por tienda:")
print(mejor_producto)

# Análisis de tendencia temporal
tendencia = df_ventas.groupby('mes').agg({
    'ingresos': 'sum',
    'unidades': 'sum'
})

print("\nTendencia mensual:")
print(tendencia)

# Identificar días de la semana con mayores ventas
ventas_por_dia = df_ventas.groupby('dia_semana').agg({
    'ingresos': 'sum'
}).reset_index()

print("\nVentas por día de la semana:")
print(ventas_por_dia)

# Análisis de correlación
print("\nCorrelación entre unidades vendidas y precio unitario:")
print(df_ventas[['unidades', 'precio_unitario']].corr())
```

## Ejercicios prácticos

### Ejercicio 1: Lectura y exploración de datos

1. Descarga un dataset de ejemplo (por ejemplo, el [Titanic dataset](https://www.kaggle.com/c/titanic/data))
2. Carga el dataset con Pandas
3. Explora las primeras filas, información y estadísticas descriptivas
4. Identifica cuántos valores nulos hay en cada columna

### Ejercicio 2: Limpieza y transformación

Partiendo del dataset anterior:
1. Elimina columnas que consideres irrelevantes
2. Reemplaza los valores nulos con estrategias adecuadas
3. Crea nuevas características (por ejemplo, extraer título del nombre)
4. Codifica variables categóricas (por ejemplo, género a 0/1)

### Ejercicio 3: Análisis y visualización

1. Calcula la tasa de supervivencia por género, clase y grupo de edad
2. Encuentra correlaciones entre diferentes variables
3. Crea al menos una tabla pivot para analizar datos

## Recursos adicionales

- [Documentación oficial de Pandas](https://pandas.pydata.org/docs/)
- [Pandas Cheat Sheet](https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf)
- [10 Minutes to Pandas](https://pandas.pydata.org/docs/user_guide/10min.html) - Tutorial oficial
- [Kaggle Learn Pandas](https://www.kaggle.com/learn/pandas) - Curso interactivo gratuito

En el siguiente tema, exploraremos cómo visualizar datos utilizando Matplotlib y Seaborn, herramientas esenciales para entender y comunicar patrones en nuestros datos.
