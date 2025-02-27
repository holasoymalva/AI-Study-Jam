# Visualización de datos con Matplotlib y Seaborn

## ¿Por qué visualizar datos?

La visualización de datos es fundamental en el proceso de análisis y en proyectos de IA por varias razones:

- **Entender los datos**: Una imagen vale más que mil números
- **Identificar patrones**: Ver tendencias, agrupaciones y valores atípicos
- **Comunicar resultados**: Transmitir hallazgos de forma clara y efectiva
- **Detectar problemas**: Identificar errores, sesgos o valores extraños
- **Generar ideas**: Inspirar nuevas hipótesis y enfoques

## Bibliotecas de visualización en Python

Vamos a trabajar con dos bibliotecas principales:

1. **Matplotlib**: La biblioteca fundamental - potente pero a veces compleja
2. **Seaborn**: Construida sobre Matplotlib - más sencilla y con gráficos estadísticos

## Instalación

```bash
pip install matplotlib seaborn
```

## Importación

```python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Configuración para mejorar la apariencia
plt.style.use('seaborn-v0_8-whitegrid')
sns.set(style="whitegrid")

# Para visualizar en notebooks de Jupyter
%matplotlib inline  
```

## Visualización con Matplotlib

### Gráficos básicos

#### Gráfico de líneas

```python
# Datos de ejemplo
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Crear la figura y los ejes
plt.figure(figsize=(10, 6))

# Crear el gráfico
plt.plot(x, y, color='blue', linestyle='-', linewidth=2, label='sen(x)')

# Personalización
plt.title('Función Seno', fontsize=16)
plt.xlabel('x', fontsize=14)
plt.ylabel('sen(x)', fontsize=14)
plt.grid(True)
plt.legend(fontsize=12)

# Mostrar el gráfico
plt.show()
```

#### Gráfico de dispersión (scatter plot)

```python
# Datos aleatorios
n = 50
x = np.random.rand(n)
y = x + np.random.normal(0, 0.2, n)

plt.figure(figsize=(8, 6))
plt.scatter(x, y, color='red', alpha=0.7, s=100)
plt.title('Gráfico de Dispersión', fontsize=16)
plt.xlabel('Variable X', fontsize=14)
plt.ylabel('Variable Y', fontsize=14)
plt.grid(True)
plt.show()
```

#### Múltiples gráficos

```python
# Datos
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y1, color='blue', label='sen(x)')
plt.plot(x, y2, color='red', label='cos(x)')
plt.title('Funciones Trigonométricas', fontsize=16)
plt.xlabel('x', fontsize=14)
plt.ylabel('y', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.show()
```

#### Gráfico de barras

```python
# Datos
categorias = ['A', 'B', 'C', 'D', 'E']
valores = [25, 40, 30, 55, 15]

plt.figure(figsize=(10, 6))
plt.bar(categorias, valores, color='skyblue', edgecolor='black', width=0.6)
plt.title('Gráfico de Barras', fontsize=16)
plt.xlabel('Categorías', fontsize=14)
plt.ylabel('Valores', fontsize=14)
plt.ylim(0, 60)
plt.grid(axis='y', alpha=0.3)

# Añadir etiquetas de valor
for i, v in enumerate(valores):
    plt.text(i, v + 1, str(v), ha='center', fontsize=12)

plt.show()
```

#### Histograma

```python
# Datos aleatorios con distribución normal
datos = np.random.normal(0, 1, 1000)

plt.figure(figsize=(10, 6))
plt.hist(datos, bins=30, color='green', alpha=0.7, edgecolor='black')
plt.title('Histograma', fontsize=16)
plt.xlabel('Valor', fontsize=14)
plt.ylabel('Frecuencia', fontsize=14)
plt.grid(True, alpha=0.3)
plt.show()
```

#### Gráfico circular (pie chart)

```python
# Datos
categorias = ['Producto A', 'Producto B', 'Producto C', 'Producto D']
ventas = [35, 25, 20, 20]
colores = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
explode = (0.1, 0, 0, 0)  # Destacar el primer elemento

plt.figure(figsize=(8, 8))
plt.pie(ventas, explode=explode, labels=categorias, colors=colores, autopct='%1.1f%%',
        shadow=True, startangle=90)
plt.title('Distribución de Ventas', fontsize=16)
plt.axis('equal')  # Para que el círculo sea redondo
plt.show()
```

### Subplots (múltiples gráficos en una figura)

```python
# Crear una figura con subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Primer subplot: línea
axes[0, 0].plot(x, np.sin(x), color='blue')
axes[0, 0].set_title('Gráfico de Línea')
axes[0, 0].set_xlabel('x')
axes[0, 0].set_ylabel('sen(x)')
axes[0, 0].grid(True)

# Segundo subplot: dispersión
axes[0, 1].scatter(x, y, color='red', alpha=0.7)
axes[0, 1].set_title('Gráfico de Dispersión')
axes[0, 1].set_xlabel('x')
axes[0, 1].set_ylabel('y')
axes[0, 1].grid(True)

# Tercer subplot: barras
axes[1, 0].bar(categorias, valores, color='skyblue', edgecolor='black')
axes[1, 0].set_title('Gráfico de Barras')
axes[1, 0].set_xlabel('Categorías')
axes[1, 0].set_ylabel('Valores')
axes[1, 0].grid(True, axis='y')

# Cuarto subplot: histograma
axes[1, 1].hist(datos, bins=20, color='green', alpha=0.7, edgecolor='black')
axes[1, 1].set_title('Histograma')
axes[1, 1].set_xlabel('Valor')
axes[1, 1].set_ylabel('Frecuencia')

# Ajustar diseño
plt.tight_layout()
plt.show()
```

### Personalización avanzada

```python
# Datos
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

# Crear figura con un cierto estilo
plt.style.use('seaborn-v0_8-darkgrid')
fig, ax = plt.subplots(figsize=(12, 7))

# Crear gráficos
ax.plot(x, y1, color='#ff6347', linestyle='-', linewidth=2.5, label='sen(x)')
ax.plot(x, y2, color='#4682b4', linestyle='--', linewidth=2.5, label='cos(x)')

# Personalizar ejes
ax.set_xlim(0, 10)
ax.set_ylim(-1.5, 1.5)
ax.set_xticks(np.arange(0, 11, 1))
ax.set_yticks(np.arange(-1, 1.1, 0.5))

# Añadir título y etiquetas
ax.set_title('Funciones Trigonométricas', fontsize=18, pad=20)
ax.set_xlabel('Ángulo (radianes)', fontsize=14, labelpad=10)
ax.set_ylabel('Valor', fontsize=14, labelpad=10)

# Personalizar grid y bordes
ax.grid(True, linestyle='--', alpha=0.7)
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)

# Añadir leyenda
ax.legend(fontsize=12, loc='upper right', frameon=True, 
          facecolor='white', edgecolor='gray', shadow=True)

# Añadir texto y anotaciones
ax.text(5, 0.8, 'Puntos de intersección', fontsize=12)
ax.annotate('π/2', xy=(np.pi/2, 1), xytext=(np.pi/2, 1.3),
            arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))

# Mostrar gráfico
plt.tight_layout()
plt.show()
```

## Visualización con Seaborn

Seaborn facilita la creación de gráficos estadísticos atractivos con menos código.

### Configuración de estilo

```python
# Estilos disponibles
print(plt.style.available)

# Establecer estilo de Seaborn
sns.set_theme(style="whitegrid")  # Opciones: darkgrid, whitegrid, dark, white, ticks
```

### Gráficos básicos de Seaborn

#### Gráfico de dispersión mejorado

```python
# Datos
tips = sns.load_dataset('tips')  # Dataset de propinas incluido en Seaborn
print(tips.head())

# Gráfico básico de dispersión
plt.figure(figsize=(10, 6))
sns.scatterplot(x='total_bill', y='tip', data=tips, hue='sex', 
                size='size', sizes=(20, 200), palette='viridis')
plt.title('Propinas vs. Total de la Cuenta', fontsize=16)
plt.xlabel('Total de la Cuenta', fontsize=14)
plt.ylabel('Propina', fontsize=14)
plt.show()
```

#### Gráfico de pares (pairplot)

```python
# Muestra relaciones entre múltiples variables
sns.pairplot(tips, hue='sex', palette='Set1', diag_kind='kde', 
             plot_kws={'alpha': 0.6, 's': 80, 'edgecolor': 'k'})
plt.suptitle('Relaciones entre Variables', y=1.02, fontsize=16)
plt.show()
```

#### Gráfico de barras

```python
# Contar por categoría
plt.figure(figsize=(10, 6))
sns.countplot(y='day', data=tips, palette='Blues_d', order=tips['day'].value_counts().index)
plt.title('Número de Clientes por Día', fontsize=16)
plt.xlabel('Número de Clientes', fontsize=14)
plt.ylabel('Día', fontsize=14)
plt.show()
```

#### Boxplot (diagrama de caja)

```python
plt.figure(figsize=(12, 7))
sns.boxplot(x='day', y='total_bill', data=tips, hue='sex', palette='Set2')
plt.title('Distribución del Total de la Cuenta por Día y Género', fontsize=16)
plt.xlabel('Día', fontsize=14)
plt.ylabel('Total de la Cuenta', fontsize=14)
plt.show()
```

#### Violinplot

```python
plt.figure(figsize=(12, 7))
sns.violinplot(x='day', y='total_bill', data=tips, hue='sex', 
               split=True, inner='quart', palette='Set3')
plt.title('Distribución del Total de la Cuenta por Día y Género', fontsize=16)
plt.xlabel('Día', fontsize=14)
plt.ylabel('Total de la Cuenta', fontsize=14)
plt.show()
```

#### Heatmap (mapa de calor)

```python
# Calcular matriz de correlación
corr = tips.corr()

# Crear heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, 
            fmt='.2f', linewidths=0.5)
plt.title('Matriz de Correlación', fontsize=16)
plt.show()
```

#### Gráficos de distribución

```python
# Histograma mejorado (distplot)
plt.figure(figsize=(10, 6))
sns.histplot(tips['total_bill'], kde=True, bins=20, color='skyblue')
plt.title('Distribución del Total de la Cuenta', fontsize=16)
plt.xlabel('Total de la Cuenta', fontsize=14)
plt.ylabel('Frecuencia', fontsize=14)
plt.show()

# Comparar distribuciones (kdeplot)
plt.figure(figsize=(10, 6))
sns.kdeplot(data=tips, x='total_bill', hue='sex', fill=True, alpha=0.5, palette='Set1')
plt.title('Distribución del Total de la Cuenta por Género', fontsize=16)
plt.xlabel('Total de la Cuenta', fontsize=14)
plt.ylabel('Densidad', fontsize=14)
plt.show()
```

### Gráficos estadísticos avanzados

#### Gráfico de regresión

```python
plt.figure(figsize=(10, 6))
sns.regplot(x='total_bill', y='tip', data=tips, 
            scatter_kws={'alpha':0.5, 's':50}, line_kws={'color':'red'})
plt.title('Relación entre Total de la Cuenta y Propina con Línea de Regresión', fontsize=16)
plt.xlabel('Total de la Cuenta', fontsize=14)
plt.ylabel('Propina', fontsize=14)
plt.show()
```

#### Jointplot

```python
# Combina scatterplot y distribuciones marginales
sns.jointplot(x='total_bill', y='tip', data=tips, kind='reg', 
              height=8, ratio=4, marginal_kws={'bins': 15})
plt.suptitle('Distribución Conjunta: Total de la Cuenta y Propina', y=1.02, fontsize=16)
plt.show()
```

#### Facetgrid (gráficos por categorías)

```python
g = sns.FacetGrid(tips, col='sex', row='time', height=4, aspect=1.5)
g.map_dataframe(sns.scatterplot, x='total_bill', y='tip')
g.add_legend()
g.fig.suptitle('Propinas por Género y Hora del Día', y=1.02, fontsize=16)
g.set_axis_labels('Total de la Cuenta', 'Propina')
g.set_titles(col_template='{col_name}', row_template='{row_name}')
plt.show()
```

#### Catplot (gráficos categóricos)

```python
plt.figure(figsize=(12, 10))
sns.catplot(x='day', y='total_bill', hue='sex', col='time', 
            kind='box', data=tips, height=5, aspect=0.8)
plt.suptitle('Distribución del Total por Día, Género y Hora', y=1.02, fontsize=16)
plt.show()
```

## Integrando Pandas con visualización

Pandas tiene integración directa con Matplotlib:

```python
# Crear DataFrame de ejemplo
np.random.seed(42)
fechas = pd.date_range('20230101', periods=12, freq='M')
datos = pd.DataFrame({
    'ventas': np.random.randint(100, 200, 12),
    'gastos': np.random.randint(50, 150, 12),
    'marketing': np.random.randint(10, 50, 12)
}, index=fechas)

# Método plot() de Pandas
datos.plot(figsize=(12, 6), title='Evolución Mensual')
plt.xlabel('Mes')
plt.ylabel('Cantidad')
plt.grid(True, alpha=0.3)
plt.show()

# Gráfico de barras
datos.plot.bar(figsize=(12, 6), title='Comparación Mensual', stacked=True)
plt.xlabel('Mes')
plt.ylabel('Cantidad')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3, axis='y')
plt.show()

# Scatter matriz
pd.plotting.scatter_matrix(datos, figsize=(12, 10), diagonal='kde', 
                          alpha=0.8, color='skyblue', edgecolor='black')
plt.suptitle('Matriz de Dispersión', y=0.98, fontsize=16)
plt.show()
```

## Visualizando para Machine Learning

### Visualización de clusters

```python
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Crear datos de ejemplo
X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Aplicar K-means
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

# Visualizar clusters
plt.figure(figsize=(10, 8))
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y_kmeans, palette='viridis', 
                s=100, alpha=0.8, edgecolor='black')

# Añadir centros
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], s=200, c='red', marker='X', edgecolor='black')

plt.title('K-means Clustering', fontsize=16)
plt.xlabel('Característica 1', fontsize=14)
plt.ylabel('Característica 2', fontsize=14)
plt.show()
```

### Matriz de confusión

```python
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# Crear datos de clasificación
X, y = make_classification(n_samples=1000, n_classes=2, n_features=10, random_state=42)

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar un clasificador
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Calcular matriz de confusión
cm = confusion_matrix(y_test, y_pred)

# Visualizar matriz de confusión
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Matriz de Confusión', fontsize=16)
plt.xlabel('Predicción', fontsize=14)
plt.ylabel('Valor Real', fontsize=14)
plt.xticks([0.5, 1.5], ['Negativo', 'Positivo'])
plt.yticks([0.5, 1.5], ['Negativo', 'Positivo'])
plt.show()
```

### Curva ROC

```python
from sklearn.metrics import roc_curve, auc

# Obtener probabilidades para clase positiva
y_scores = clf.predict_proba(X_test)[:, 1]

# Calcular tasa de falsos positivos (FPR) y tasa de verdaderos positivos (TPR)
fpr, tpr, thresholds = roc_curve(y_test, y_scores)
roc_auc = auc(fpr, tpr)

# Visualizar curva ROC
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.title('Curva ROC', fontsize=16)
plt.xlabel('Tasa de Falsos Positivos', fontsize=14)
plt.ylabel('Tasa de Verdaderos Positivos', fontsize=14)
plt.legend(loc='lower right', fontsize=12)
plt.grid(True, alpha=0.3)
plt.show()
```

### Importancia de características

```python
# Obtener importancia de características
importancias = clf.feature_importances_
indices = np.argsort(importancias)[::-1]

plt.figure(figsize=(10, 6))
sns.barplot(x=importancias[indices], y=range(len(indices)), palette='viridis')
plt.yticks(range(len(indices)), [f'Característica {i}' for i in indices])
plt.title('Importancia de Características', fontsize=16)
plt.xlabel('Importancia', fontsize=14)
plt.ylabel('Características', fontsize=14)
plt.tight_layout()
plt.show()
```

## Caso Práctico: Análisis Exploratorio de Datos (EDA)

Vamos a realizar un análisis completo de un dataset real:

```python
# Cargar dataset de Titanic
titanic = sns.load_dataset('titanic')
print(titanic.head())

# 1. Exploración inicial
print(titanic.info())
print(titanic.describe())

# 2. Histogramas para variables numéricas
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
sns.histplot(titanic['age'].dropna(), kde=True, bins=20)
plt.title('Distribución de Edad')

plt.subplot(2, 2, 2)
sns.histplot(titanic['fare'].dropna(), kde=True, bins=20)
plt.title('Distribución de Tarifa')

plt.subplot(2, 2, 3)
sns.countplot(x='class', data=titanic)
plt.title('Conteo por Clase')

plt.subplot(2, 2, 4)
sns.countplot(x='sex', data=titanic)
plt.title('Conteo por Género')

plt.tight_layout()
plt.show()

# 3. Análisis de supervivencia
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
sns.countplot(x='survived', data=titanic)
plt.title('Supervivencia General')

plt.subplot(2, 2, 2)
sns.countplot(x='survived', hue='sex', data=titanic)
plt.title('Supervivencia por Género')

plt.subplot(2, 2, 3)
sns.countplot(x='survived', hue='class', data=titanic)
plt.title('Supervivencia por Clase')

plt.subplot(2, 2, 4)
age_bins = [0, 12, 20, 40, 60, 100]
titanic['age_group'] = pd.cut(titanic['age'], bins=age_bins, 
                             labels=['Niño', 'Joven', 'Adulto', 'Mayor', 'Anciano'])
sns.countplot(x='survived', hue='age_group', data=titanic)
plt.title('Supervivencia por Grupo de Edad')

plt.tight_layout()
plt.show()

# 4. Relaciones entre variables
plt.figure(figsize=(12, 8))
sns.violinplot(x='class', y='age', hue='survived', data=titanic, split=True, inner='quart')
plt.title('Distribución de Edad por Clase y Supervivencia', fontsize=16)
plt.show()

# 5. Matriz de correlación para variables numéricas
corr = titanic.select_dtypes(include=[np.number]).corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f')
plt.title('Matriz de Correlación', fontsize=16)
plt.show()

# 6. Análisis de supervivencia con tarifas
plt.figure(figsize=(10, 6))
sns.boxplot(x='survived', y='fare', data=titanic)
plt.title('Distribución de Tarifas por Supervivencia', fontsize=16)
plt.show()

# 7. Análisis multivariado
g = sns.FacetGrid(titanic, col='sex', row='class', height=3.5, aspect=1.5)
g.map_dataframe(sns.histplot, x='age', hue='survived', multiple='stack')
g.fig.suptitle('Supervivencia por Edad, Género y Clase', y=1.02, fontsize=16)
g.set_axis_labels('Edad', 'Conteo')
g.add_legend()
plt.show()
```

## Guardando gráficos

```python
# Guardar un gráfico como imagen
plt.figure(figsize=(10, 6))
sns.histplot(titanic['age'].dropna(), kde=True, bins=20)
plt.title('Distribución de Edad de Pasajeros del Titanic', fontsize=16)
plt.savefig('distribucion_edad.png', dpi=300, bbox_inches='tight')
plt.show()
```

## Consejos para visualización efectiva

1. **Mantén la simplicidad**: No sobrecargues un gráfico con demasiada información
2. **Elige el gráfico adecuado**:
   - Líneas: para tendencias temporales
   - Barras: para comparar categorías
   - Dispersión: para relaciones entre variables
   - Histogramas: para distribuciones
   - Cajas/Violines: para comparar distribuciones entre grupos
3. **Utiliza colores con sentido**:
   - Coherentes y accesibles para personas con daltonismo
   - Con propósito (ej. rojo para valores negativos)
4. **Etiqueta correctamente**:
   - Títulos claros
   - Ejes con unidades
   - Leyendas explicativas
5. **Optimiza para tu audiencia**:
   - Técnica vs. no técnica
   - Impresión vs. pantalla

## Ejercicios prácticos

### Ejercicio 1: Visualización básica

Utilizando el dataset del Titanic:
1. Crea un histograma de la edad de los pasajeros
2. Realiza un gráfico de barras para contar pasajeros por clase
3. Haz un gráfico circular para mostrar la proporción de supervivientes

### Ejercicio 2: Visualización de relaciones

1. Crea un gráfico de dispersión entre 'edad' y 'tarifa'
2. Realiza un boxplot que muestre la distribución de tarifas por clase
3. Visualiza la supervivencia por género y clase

### Ejercicio 3: Análisis exploratorio completo

Elige uno de estos datasets (incluidos en Seaborn):
- `sns.load_dataset('iris')` - Flores iris
- `sns.load_dataset('diamonds')` - Precios de diamantes
- `sns.load_dataset('mpg')` - Consumo de combustible

Realiza un análisis exploratorio completo con al menos 5 visualizaciones diferentes que exploren:
- Distribuciones de variables numéricas
- Conteos de variables categóricas
- Relaciones entre variables
- Comparaciones entre grupos

## Recursos adicionales

- [Documentación de Matplotlib](https://matplotlib.org/stable/gallery/index.html)
- [Documentación de Seaborn](https://seaborn.pydata.org/examples/index.html)
- [Data Visualization Cheatsheet](https://www.python-graph-gallery.com/)
- [Data Visualization in Python](https://towardsdatascience.com/data-visualization-using-matplotlib-16f1aae5ce70)

Con estas herramientas de visualización, ahora estás preparado para explorar y comunicar efectivamente patrones en tus datos, un paso crucial antes de aplicar algoritmos de aprendizaje automático en los siguientes módulos.
