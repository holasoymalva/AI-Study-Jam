# Fundamentos de Python - Guía practica

Aprenderás los conceptos fundamentales de Python en solo 1 hora! 🚀

## Requisitos Previos
- Python 3.x instalado
- Un editor de código (VS Code recomendado)
- ¡Ganas de aprender!

## 1. Variables y Tipos de Datos

```python
# Variables y asignación
nombre = "Juan"           # String (cadena de texto)
edad = 25                 # Integer (número entero)
altura = 1.75            # Float (número decimal)
es_estudiante = True     # Boolean (booleano)
nulo = None             # None (nulo)

# Tipos de datos compuestos
lista = [1, 2, 3, "cuatro"]        # List (lista)
tupla = (1, 2, 3)                  # Tuple (tupla)
conjunto = {1, 2, 3}               # Set (conjunto)
diccionario = {                    # Dictionary (diccionario)
    "nombre": "Ana",
    "edad": 30
}

# Strings y formato
nombre = "Python"
version = 3
mensaje = f"Bienvenido a {nombre} {version}!"  # f-strings
print(mensaje)

# Conversión de tipos
numero_str = "42"
numero_int = int(numero_str)       # String a Integer
texto = str(123)                   # Integer a String
decimal = float("3.14")           # String a Float
```

## 2. Operadores

```python
# Operadores aritméticos
suma = 5 + 3          # 8
resta = 10 - 4        # 6
multiplicacion = 3 * 4 # 12
division = 15 / 3     # 5.0 (división flotante)
division_entera = 17 // 3  # 5 (división entera)
modulo = 17 % 5       # 2 (resto)
potencia = 2 ** 3     # 8 (exponente)

# Operadores de comparación
igual = 5 == 5        # True
diferente = 5 != 3    # True
mayor = 5 > 3         # True
menor = 5 < 3         # False
mayor_igual = 5 >= 5  # True
menor_igual = 5 <= 6  # True

# Operadores lógicos
and_logico = True and False  # False
or_logico = True or False   # True
not_logico = not True      # False
```

## 3. Control de Flujo

```python
# if, elif, else
edad = 18

if edad < 13:
    print("Eres un niño")
elif edad < 20:
    print("Eres un adolescente")
else:
    print("Eres un adulto")

# Match (Python 3.10+)
dia = "Lunes"
match dia:
    case "Lunes":
        print("Inicio de semana")
    case "Viernes":
        print("¡Fin de semana!")
    case _:
        print("Otro día")

# Operador ternario
edad = 20
mensaje = "Mayor de edad" if edad >= 18 else "Menor de edad"
```

## 4. Bucles

```python
# For loop
for i in range(5):
    print(i)  # 0, 1, 2, 3, 4

# For con lista
frutas = ["manzana", "pera", "uva"]
for fruta in frutas:
    print(fruta)

# For con enumerate
for indice, fruta in enumerate(frutas):
    print(f"{indice}: {fruta}")

# While loop
contador = 0
while contador < 3:
    print(contador)  # 0, 1, 2
    contador += 1

# Break y Continue
for i in range(5):
    if i == 2:
        continue  # Salta la iteración
    if i == 4:
        break    # Sale del bucle
    print(i)
```

## 5. Funciones

```python
# Función básica
def saludar(nombre):
    return f"Hola {nombre}!"

# Función con parámetros por defecto
def crear_perfil(nombre, edad=25, ciudad="Madrid"):
    return {
        "nombre": nombre,
        "edad": edad,
        "ciudad": ciudad
    }

# Función con args y kwargs
def info(*args, **kwargs):
    print(f"Args: {args}")
    print(f"Kwargs: {kwargs}")

info(1, 2, 3, nombre="Juan", edad=30)

# Lambda (funciones anónimas)
cuadrado = lambda x: x ** 2

# Decoradores básicos
def mi_decorador(func):
    def wrapper():
        print("Antes de la función")
        func()
        print("Después de la función")
    return wrapper

@mi_decorador
def saludar():
    print("¡Hola!")
```

## 6. Listas, Tuplas y Sets

```python
# Listas
numeros = [1, 2, 3, 4, 5]
numeros.append(6)        # Añadir al final
numeros.pop()           # Eliminar del final
numeros.insert(0, 0)    # Insertar en posición
primer = numeros[0]     # Acceder por índice
sublista = numeros[1:4] # Slice

# List comprehension
cuadrados = [x**2 for x in range(5)]

# Tuplas (inmutables)
coordenadas = (10, 20)
x, y = coordenadas      # Desempaquetado

# Sets
conjunto = {1, 2, 3, 3} # Elementos únicos
conjunto.add(4)         # Añadir elemento
conjunto.remove(1)      # Eliminar elemento
```

## 7. Diccionarios

```python
# Creación y acceso
persona = {
    "nombre": "Carlos",
    "edad": 30,
    "ciudad": "Barcelona"
}

# Acceder y modificar
nombre = persona["nombre"]
persona["edad"] = 31

# Métodos útiles
claves = persona.keys()
valores = persona.values()
items = persona.items()

# Dictionary comprehension
cuadrados = {x: x**2 for x in range(5)}

# get con valor por defecto
edad = persona.get("edad", 0)  # Si no existe, retorna 0
```

## Ejercicios Prácticos

1. Crear una función que calcule el promedio de una lista
```python
def promedio(numeros):
    return sum(numeros) / len(numeros)

# Uso
notas = [15, 18, 12, 20, 14]
print(promedio(notas))  # 15.8
```

2. Crear una clase simple
```python
class Estudiante:
    def __init__(self, nombre, edad):
        self.nombre = nombre
        self.edad = edad
    
    def presentarse(self):
        return f"Hola, soy {self.nombre} y tengo {self.edad} años"

# Uso
estudiante = Estudiante("Ana", 20)
print(estudiante.presentarse())
```

## Recursos Adicionales
- [Documentación oficial de Python](https://docs.python.org/es/3/)
- [Real Python](https://realpython.com/)
- [Python para principiantes](https://www.python.org/about/gettingstarted/)
- [Automate the Boring Stuff with Python](https://automatetheboringstuff.com/)

¡Felicitaciones! 🎉 Has completado el tutorial de fundamentos de Python. Ahora tienes las bases necesarias para comenzar a programar en Python.
