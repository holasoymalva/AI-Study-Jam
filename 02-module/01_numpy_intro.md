# Introducción a NumPy

## ¿Qué es NumPy y por qué es importante?

NumPy (Numerical Python) es la biblioteca fundamental para la computación científica en Python. Si quieres trabajar con IA, NumPy será tu mejor amigo por estas razones:

- **Rapidez**: Las operaciones con NumPy son mucho más rápidas que con listas de Python
- **Arrays multidimensionales**: Permite trabajar fácilmente con matrices y tensores
- **Operaciones vectorizadas**: Puedes realizar operaciones en arrays completos sin bucles
- **Base para otras bibliotecas**: Pandas, SciPy, scikit-learn y TensorFlow utilizan NumPy

## Instalación

Si aún no has instalado NumPy:

```bash
pip install numpy
```

## Importación

NumPy se importa convencionalmente como `np`:

```python
import numpy as np
```

## Arrays de NumPy: Lo básico

Un array de NumPy es similar a una lista de Python, pero con superpoderes. La clase principal es `ndarray`.

### Creando arrays

```python
# Desde una lista
lista = [1, 2, 3, 4, 5]
array = np.array(lista)
print(array)  # [1 2 3 4 5]

# Array de ceros
ceros = np.zeros(5)
print(ceros)  # [0. 0. 0. 0. 0.]

# Array de unos
unos = np.ones(5)
print(unos)  # [1. 1. 1. 1. 1.]

# Rango de números
rango = np.arange(0, 10, 2)  # Inicio, fin (exclusivo), paso
print(rango)  # [0 2 4 6 8]

# Números espaciados uniformemente
lineal = np.linspace(0, 1, 5)  # Inicio, fin (inclusivo), cantidad
print(lineal)  # [0.   0.25 0.5  0.75 1.  ]

# Array de valores aleatorios
aleatorio = np.random.random(5)
print(aleatorio)  # [0.42 0.71 0.28 ...]
```

### Arrays multidimensionales (matrices)

```python
# Matriz 2x3
matriz = np.array([
    [1, 2, 3],
    [4, 5, 6]
])
print(matriz)
# [[1 2 3]
#  [4 5 6]]

# Matriz de ceros 3x3
matriz_ceros = np.zeros((3, 3))
print(matriz_ceros)
# [[0. 0. 0.]
#  [0. 0. 0.]
#  [0. 0. 0.]]

# Matriz identidad 3x3
matriz_identidad = np.eye(3)
print(matriz_identidad)
# [[1. 0. 0.]
#  [0. 1. 0.]
#  [0. 0. 1.]]
```

## Atributos de los arrays

```python
# Crear un array
arr = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

# Forma (dimensiones)
print(f"Forma: {arr.shape}")  # (2, 3)

# Dimensiones
print(f"Dimensiones: {arr.ndim}")  # 2

# Tamaño (total de elementos)
print(f"Tamaño: {arr.size}")  # 6

# Tipo de datos
print(f"Tipo de datos: {arr.dtype}")  # int64
```

## Indexación y rebanado (slicing)

```python
# Crear un array
arr = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12]
])

# Acceder a un elemento
print(arr[0, 0])  # 1
print(arr[2, 3])  # 12

# Obtener una fila
print(arr[1])  # [5 6 7 8]

# Obtener una columna
print(arr[:, 2])  # [3 7 11]

# Rebanado (slicing)
print(arr[0:2, 1:3])
# [[2 3]
#  [6 7]]

# Indexación booleana
print(arr > 6)
# [[False False False False]
#  [False False  True  True]
#  [ True  True  True  True]]

# Filtrado con condición
print(arr[arr > 6])  # [7 8 9 10 11 12]
```

## Operaciones básicas

### Aritméticas

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Suma
print(a + b)  # [5 7 9]

# Resta
print(b - a)  # [3 3 3]

# Multiplicación (elemento por elemento)
print(a * b)  # [4 10 18]

# División
print(b / a)  # [4. 2.5 2.]

# Potencia
print(a ** 2)  # [1 4 9]

# Operaciones con escalares
print(a + 10)  # [11 12 13]
print(a * 2)   # [2 4 6]

# Funciones matemáticas
print(np.sqrt(a))  # [1. 1.41421356 1.73205081]
print(np.exp(a))   # [ 2.71828183  7.3890561  20.08553692]
print(np.log(a))   # [0. 0.69314718 1.09861229]
```

### Agregación

```python
arr = np.array([1, 2, 3, 4, 5])

# Suma de todos los elementos
print(np.sum(arr))  # 15

# Media (promedio)
print(np.mean(arr))  # 3.0

# Desviación estándar
print(np.std(arr))  # 1.4142135623730951

# Mínimo y máximo
print(np.min(arr))  # 1
print(np.max(arr))  # 5

# Argumento del mínimo y máximo (índices)
print(np.argmin(arr))  # 0
print(np.argmax(arr))  # 4
```

### Con matrices

```python
a = np.array([
    [1, 2],
    [3, 4]
])
b = np.array([
    [5, 6],
    [7, 8]
])

# Suma por eje
print(np.sum(a, axis=0))  # [4 6] (suma por columnas)
print(np.sum(a, axis=1))  # [3 7] (suma por filas)

# Producto matricial
print(np.dot(a, b))
# [[19 22]
#  [43 50]]

# Otra forma de producto matricial
print(a @ b)
# [[19 22]
#  [43 50]]

# Transpuesta
print(a.T)
# [[1 3]
#  [2 4]]
```

## Reorganización de arrays

```python
# Cambiar forma
arr = np.arange(12)
print(arr)  # [ 0  1  2  3  4  5  6  7  8  9 10 11]

# Cambiar a 3x4
matriz = arr.reshape(3, 4)
print(matriz)
# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]]

# Aplanar
plano = matriz.flatten()
print(plano)  # [ 0  1  2  3  4  5  6  7  8  9 10 11]
```

## Broadcasting

NumPy puede trabajar con arrays de diferentes formas mediante "broadcasting":

```python
# Array 3x3
a = np.ones((3, 3))
# Array 1D
b = np.array([1, 2, 3])

# NumPy "transmite" b para que coincida con a
c = a + b
print(c)
# [[2. 3. 4.]
#  [2. 3. 4.]
#  [2. 3. 4.]]
```

## NumPy para IA: Aplicación práctica

Veamos un ejemplo práctico: normalización de datos, común en preparación para modelos de IA.

```python
# Datos de ejemplo: medidas de altura (cm) y peso (kg)
altura = np.array([165, 170, 180, 175, 160, 190])
peso = np.array([60, 65, 75, 70, 55, 85])

# Visualizamos los datos originales
print("Datos originales:")
print(f"Altura: {altura}, Media: {np.mean(altura):.1f}, Desv. Est: {np.std(altura):.1f}")
print(f"Peso: {peso}, Media: {np.mean(peso):.1f}, Desv. Est: {np.std(peso):.1f}")

# Normalización Z-score (centrar en 0, escalar a desviación estándar 1)
altura_norm = (altura - np.mean(altura)) / np.std(altura)
peso_norm = (peso - np.mean(peso)) / np.std(peso)

print("\nDatos normalizados:")
print(f"Altura: {altura_norm}")
print(f"Media altura normalizada: {np.mean(altura_norm):.10f}")  # Muy cercana a 0
print(f"Desv. Est. altura normalizada: {np.std(altura_norm):.10f}")  # Exactamente 1
```

## Ejercicios prácticos

### Ejercicio 1: Operaciones básicas

Crea dos arrays NumPy y realiza:
1. Suma, resta, multiplicación y división elemento a elemento
2. Calcula la media, mediana y desviación estándar de cada array
3. Encuentra el valor máximo y mínimo, y sus posiciones

### Ejercicio 2: Manipulación de matrices

1. Crea una matriz 3x3 con valores aleatorios entre 0 y 1
2. Extrae la primera fila y la última columna
3. Reemplaza todos los valores mayores a 0.5 por 1 y los demás por 0
4. Calcula la transpuesta de la matriz resultante

### Ejercicio 3: Aplicación - Distancia Euclidiana

La distancia euclidiana es fundamental en muchos algoritmos de IA como K-nearest neighbors.

```python
# Implementa una función que calcule la distancia euclidiana entre dos puntos
def distancia_euclidiana(punto1, punto2):
    # Sugerencia: usa np.sqrt() y operaciones vectorizadas
    return np.sqrt(np.sum((punto1 - punto2) ** 2))

# Prueba con algunos puntos
p1 = np.array([1, 2, 3])
p2 = np.array([4, 5, 6])
print(f"Distancia entre {p1} y {p2}: {distancia_euclidiana(p1, p2):.4f}")
```

## Recursos adicionales

- [Documentación oficial de NumPy](https://numpy.org/doc/stable/)
- [NumPy Cheat Sheet](https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Numpy_Python_Cheat_Sheet.pdf)
- [Tutorial de NumPy en Kaggle](https://www.kaggle.com/code/python10pm/numpy-tutorial-with-examples)

En el siguiente tema, aprenderemos sobre Pandas, que nos permitirá trabajar con datos estructurados en formato de tabla.
