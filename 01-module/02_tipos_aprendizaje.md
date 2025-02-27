# Tipos de Aprendizaje Automático

El Aprendizaje Automático (Machine Learning) se divide en varios enfoques diferentes, cada uno adecuado para resolver distintos tipos de problemas. Vamos a explorarlos:

## 1. Aprendizaje Supervisado

Es como aprender con un profesor que te dice las respuestas correctas.

### ¿Cómo funciona?

1. El algoritmo recibe **datos etiquetados** (ejemplos con respuestas correctas)
2. Aprende a asociar características con etiquetas
3. Puede predecir etiquetas para nuevos datos no vistos

### Tipos de problemas:

#### Clasificación

Asignar datos a categorías discretas.

**Ejemplos:**
- Identificar si un email es spam o no
- Reconocer dígitos manuscritos (0-9)
- Diagnosticar enfermedades a partir de síntomas

```python
# Ejemplo sencillo de clasificación con scikit-learn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Datos ficticios: características de frutas [peso, textura, dulzura]
X = np.array([
    [150, 0, 8],  # manzana
    [170, 0, 9],  # manzana
    [140, 0, 7],  # manzana
    [130, 1, 10], # naranja
    [150, 1, 9],  # naranja
    [160, 1, 8]   # naranja
])

# Etiquetas: 0 para manzana, 1 para naranja
y = np.array([0, 0, 0, 1, 1, 1])

# Dividir en datos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Crear y entrenar el modelo
clf = RandomForestClassifier(n_estimators=10)
clf.fit(X_train, y_train)

# Predecir con datos de prueba
y_pred = clf.predict(X_test)

# Calcular precisión
print(f"Precisión: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# Predecir una nueva fruta [peso=145, textura=0, dulzura=8]
nueva_fruta = np.array([[145, 0, 8]])
prediccion = clf.predict(nueva_fruta)
print(f"Predicción: {'Manzana' if prediccion[0] == 0 else 'Naranja'}")
```

#### Regresión

Predecir valores numéricos continuos.

**Ejemplos:**
- Predecir el precio de una casa
- Estimar ventas futuras
- Calcular la temperatura esperada

```python
# Ejemplo sencillo de regresión con scikit-learn
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

# Datos ficticios: metros cuadrados de casas
X = np.array([[50], [65], [80], [100], [120], [140], [160], [180]])
# Precios correspondientes (en miles de $)
y = np.array([100, 150, 200, 250, 300, 350, 400, 450])

# Crear y entrenar el modelo
modelo = LinearRegression()
modelo.fit(X, y)

# Predecir para un nuevo tamaño de casa: 110m²
precio_predicho = modelo.predict([[110]])
print(f"Precio predicho para 110m²: ${precio_predicho[0]:.2f} mil")

# Visualizar la regresión
plt.figure(figsize=(8, 6))
plt.scatter(X, y, color='blue', label='Datos reales')
plt.plot(X, modelo.predict(X), color='red', label='Línea de regresión')
plt.scatter([[110]], precio_predicho, color='green', s=100, label='Predicción')
plt.xlabel('Metros cuadrados')
plt.ylabel('Precio (miles $)')
plt.title('Regresión Lineal: Precio vs Tamaño')
plt.legend()
plt.grid(True)
plt.show()

# Coeficientes del modelo
print(f"Pendiente: {modelo.coef_[0]:.2f}")
print(f"Intercepto: {modelo.intercept_:.2f}")
```

### Algoritmos comunes de Aprendizaje Supervisado:

- **Regresión Lineal/Logística**: Modelos simples basados en ecuaciones lineales
- **Árboles de Decisión**: Estructura similar a un diagrama de flujo para toma de decisiones
- **Random Forest**: Combinación de múltiples árboles de decisión
- **SVM (Support Vector Machines)**: Encuentra hiperplanos óptimos para separar clases
- **KNN (K-Nearest Neighbors)**: Clasifica basándose en los k ejemplos más cercanos
- **Redes Neuronales**: Modelos inspirados en el cerebro humano

## 2. Aprendizaje No Supervisado

Es como explorar sin un mapa, buscando patrones por ti mismo.

### ¿Cómo funciona?

1. El algoritmo recibe **datos sin etiquetar**
2. Busca estructuras, patrones o relaciones interesantes
3. Organiza los datos según similitudes o diferencias

### Tipos de problemas:

#### Clustering (Agrupamiento)

Agrupar datos similares sin conocer categorías previas.

**Ejemplos:**
- Segmentación de clientes según comportamiento de compra
- Agrupación de documentos por temas
- Detección de comunidades en redes sociales

```python
# Ejemplo de clustering con K-means
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

# Datos ficticios: puntos en 2D
X = np.array([
    [1, 2], [1.5, 1.8], [2, 2.5], 
    [8, 8], [8.5, 8], [8, 9],
    [1, 8], [1.5, 8.5], [2, 8]
])

# Crear y entrenar el modelo
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# Obtener las etiquetas de cluster y centros
labels = kmeans.labels_
centers = kmeans.cluster_centers_

# Visualizar los clusters
plt.figure(figsize=(8, 6))
colors = ['red', 'green', 'blue']
for i in range(len(X)):
    plt.scatter(X[i, 0], X[i, 1], color=colors[labels[i]], s=50)
plt.scatter(centers[:, 0], centers[:, 1], color='black', marker='X', s=200, label='Centroides')
plt.title('Clustering K-means')
plt.legend()
plt.grid(True)
plt.show()

# Predecir el cluster para un nuevo punto
nuevo_punto = np.array([[5, 5]])
cluster = kmeans.predict(nuevo_punto)
print(f"El nuevo punto pertenece al cluster {cluster[0]}")
```

#### Reducción de Dimensionalidad

Reducir el número de variables manteniendo la información importante.

**Ejemplos:**
- Comprimir imágenes o datos
- Visualizar datos de alta dimensionalidad en 2D o 3D
- Eliminar ruido y características redundantes

```python
# Ejemplo de PCA (Análisis de Componentes Principales)
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

# Datos ficticios multidimensionales (5 dimensiones)
X = np.random.rand(100, 5)

# Aplicar PCA para reducir a 2 dimensiones
pca = PCA(n_components=2)
X_reducido = pca.fit_transform(X)

# Ver cuánta varianza explica cada componente
varianza_explicada = pca.explained_variance_ratio_
print(f"Varianza explicada por componente: {varianza_explicada}")
print(f"Varianza total explicada: {sum(varianza_explicada) * 100:.2f}%")

# Visualizar los datos reducidos
plt.figure(figsize=(8, 6))
plt.scatter(X_reducido[:, 0], X_reducido[:, 1], alpha=0.8)
plt.title('Datos reducidos con PCA')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.grid(True)
plt.show()
```

#### Detección de Anomalías

Identificar datos inusuales o atípicos.

**Ejemplos:**
- Detección de fraudes en transacciones bancarias
- Identificación de fallas en maquinaria
- Monitoreo de seguridad en redes

### Algoritmos comunes de Aprendizaje No Supervisado:

- **K-means**: Agrupa datos en k clusters
- **DBSCAN**: Clustering basado en densidad
- **PCA (Análisis de Componentes Principales)**: Reduce dimensionalidad
- **t-SNE**: Visualización de datos de alta dimensionalidad
- **Isolation Forest**: Detección de anomalías
- **Autoencoders**: Redes neuronales para representaciones comprimidas

## 3. Aprendizaje por Refuerzo

Es como aprender a través de ensayo y error, recibiendo recompensas o castigos.

### ¿Cómo funciona?

1. Un **agente** interactúa con un **entorno**
2. Realiza **acciones** y recibe **recompensas** o **penalizaciones**
3. Aprende a maximizar la recompensa total a largo plazo

### Conceptos clave:

- **Estado**: Situación actual del entorno
- **Acción**: Lo que el agente puede hacer
- **Recompensa**: Feedback numérico sobre la acción
- **Política**: Estrategia que sigue el agente

### Aplicaciones:

- **Videojuegos**: DeepMind con Atari, AlphaGo
- **Robótica**: Robots que aprenden a caminar o manipular objetos
- **Vehículos autónomos**: Aprender a conducir de manera segura
- **Recomendaciones**: Sistemas que mejoran con feedback del usuario

```python
# Ejemplo conceptual simplificado de Q-learning
import numpy as np
import matplotlib.pyplot as plt

# Entorno simplificado: grid 4x4 con obstáculos (-1), meta (10) y camino libre (0)
entorno = np.array([
    [0, 0, 0, 0],
    [0, -1, 0, -1],
    [0, 0, 0, -1],
    [-1, 0, 0, 10]
])

# Visualizar el entorno
plt.figure(figsize=(6, 6))
plt.imshow(entorno, cmap='coolwarm')
plt.title('Entorno de Aprendizaje por Refuerzo')
plt.colorbar(label='Recompensa')
for i in range(4):
    for j in range(4):
        plt.text(j, i, f"{entorno[i, j]}", ha='center', va='center')
plt.show()

# En un sistema real, implementaríamos Q-learning:
# 1. Inicializar tabla Q con ceros
# 2. Para cada episodio:
#    a. Elegir un estado inicial
#    b. Mientras no se llegue al estado terminal:
#       i. Elegir una acción (exploración vs explotación)
#       ii. Realizar la acción y observar recompensa y nuevo estado
#       iii. Actualizar valor Q según la ecuación:
#            Q(s,a) = Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
#    c. Repetir hasta converger

# Este es solo un ejemplo conceptual. El aprendizaje por refuerzo
# requiere más código para una implementación completa.
```

## 4. Otros tipos de aprendizaje

### Aprendizaje Semi-supervisado

Combina datos etiquetados (pocos) y no etiquetados (muchos).

**Aplicaciones:**
- Clasificación de páginas web
- Análisis de imágenes médicas
- Reconocimiento de voz

### Aprendizaje por Transferencia (Transfer Learning)

Aprovechar conocimiento de un modelo pre-entrenado para una nueva tarea.

**Aplicaciones:**
- Procesamiento de lenguaje natural (BERT, GPT)
- Visión por computadora (modelos pre-entrenados con ImageNet)
- Reconocimiento de voz

```python
# Ejemplo conceptual de Transfer Learning con un modelo pre-entrenado
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

# Cargar modelo pre-entrenado (sin la capa de clasificación)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# "Congelar" las capas del modelo base para que no se actualicen durante el entrenamiento
for layer in base_model.layers:
    layer.trainable = False

# Añadir nuevas capas para nuestra tarea específica
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
predictions = Dense(10, activation='softmax')(x)  # 10 clases para nuestro problema

# Crear el modelo final
model = Model(inputs=base_model.input, outputs=predictions)

# Compilar el modelo
model.compile(optimizer=Adam(learning_rate=0.0001), 
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Ahora podríamos entrenar este modelo con nuestros datos específicos
# model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_val, y_val))
```

## ¿Cómo elegir el enfoque adecuado?

La elección depende del problema y los datos disponibles:

| Si tienes... | Y quieres... | Considera usar... |
|--------------|--------------|-------------------|
| Datos etiquetados | Predecir categorías | Clasificación supervisada |
| Datos etiquetados | Predecir valores numéricos | Regresión supervisada |
| Datos sin etiquetar | Descubrir grupos | Clustering |
| Datos sin etiquetar | Reducir complejidad | Reducción de dimensionalidad |
| Un entorno interactivo | Aprender comportamientos óptimos | Aprendizaje por refuerzo |
| Pocos datos etiquetados | Aprovechar datos sin etiquetar | Aprendizaje semi-supervisado |
| Pocos datos para tu problema | Aprovechar conocimiento previo | Transfer learning |

## Para practicar

Para entender mejor estos conceptos, puedes:

1. Explorar datasets públicos en [Kaggle](https://www.kaggle.com/datasets)
2. Probar ejemplos en Scikit-learn: [Tutoriales](https://scikit-learn.org/stable/tutorial/index.html)
3. Experimentar con [Google Colab](https://colab.research.google.com/) (notebooks gratuitos)

## Recursos adicionales

- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html) - Guía detallada de algoritmos
- [Machine Learning Mastery](https://machinelearningmastery.com/) - Tutoriales prácticos
- [TensorFlow Playground](https://playground.tensorflow.org/) - Experimentación visual con redes neuronales

---

En el siguiente tema, configuraremos nuestro entorno de trabajo para comenzar a implementar estos algoritmos.
