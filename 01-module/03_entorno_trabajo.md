# Preparación del Entorno de Trabajo

Antes de sumergirnos en el desarrollo de modelos de IA, necesitamos configurar nuestro entorno de trabajo. Este documento te guiará paso a paso para tener todo listo para los ejercicios prácticos.

## Opciones para tu entorno de desarrollo

Tienes varias opciones para configurar tu entorno:

1. **Instalación local**: Instalar todo en tu computadora
2. **Google Colab**: Usar notebooks en la nube (gratis, con GPUs)
3. **Entornos virtuales**: Usar ambientes aislados en tu computadora

Vamos a cubrir todas estas opciones.

## Opción 1: Instalación Local

### Paso 1: Instalar Python

Si aún no tienes Python, debes instalarlo:

#### Windows:
1. Visita [python.org](https://www.python.org/downloads/)
2. Descarga la última versión de Python (3.8 o superior)
3. **¡Importante!** Durante la instalación, marca la casilla "Add Python to PATH"
4. Completa la instalación

#### macOS:
1. Instala [Homebrew](https://brew.sh/) si no lo tienes
2. Abre Terminal y ejecuta:
   ```bash
   brew install python
   ```

#### Linux:
La mayoría de distribuciones ya tienen Python. Si no:
```bash
sudo apt update
sudo apt install python3 python3-pip
```

### Paso 2: Verificar la instalación

Abre una terminal o línea de comandos y ejecuta:

```bash
python --version  # o python3 --version en macOS/Linux
```

Deberías ver algo como `Python 3.8.10` (o una versión similar).

### Paso 3: Instalar bibliotecas esenciales

Para este curso, necesitaremos algunas bibliotecas de Python. Instálalas con:

```bash
pip install numpy pandas matplotlib scikit-learn jupyter
```

Si planeas trabajar con deep learning, también puedes instalar:

```bash
pip install tensorflow  # o 'tensorflow-gpu' si tienes GPU compatible
```

### Paso 4: Verificar la instalación

Crea un archivo `verificacion.py` con este contenido:

```python
print("Verificando entorno de IA...")

try:
    import numpy as np
    print(f"✅ NumPy: {np.__version__}")
except ImportError:
    print("❌ NumPy no está instalado")

try:
    import pandas as pd
    print(f"✅ Pandas: {pd.__version__}")
except ImportError:
    print("❌ Pandas no está instalado")

try:
    import matplotlib as mpl
    print(f"✅ Matplotlib: {mpl.__version__}")
except ImportError:
    print("❌ Matplotlib no está instalado")

try:
    import sklearn
    print(f"✅ Scikit-learn: {sklearn.__version__}")
except ImportError:
    print("❌ Scikit-learn no está instalado")

try:
    import tensorflow as tf
    print(f"✅ TensorFlow: {tf.__version__}")
    print(f"   GPU disponible: {'Sí' if tf.config.list_physical_devices('GPU') else 'No'}")
except ImportError:
    print("ℹ️ TensorFlow no está instalado (opcional para deep learning)")

print("\n¡Listo para comenzar con IA!")
```

Ejecútalo con:

```bash
python verificacion.py
```

## Opción 2: Google Colab (Recomendado para principiantes)

Google Colab es una plataforma gratuita que ofrece notebooks de Python con todas las bibliotecas ya instaladas y acceso a GPUs gratuitas.

### Paso 1: Acceder a Google Colab

1. Ve a [colab.research.google.com](https://colab.research.google.com/)
2. Inicia sesión con tu cuenta de Google
3. Haz clic en "Nuevo notebook"

### Paso 2: Verificar el entorno

Ejecuta este código en una celda:

```python
!python --version
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import tensorflow as tf

print(f"NumPy: {np.__version__}")
print(f"Pandas: {pd.__version__}")
print(f"Matplotlib: {matplotlib.__version__}")
print(f"Scikit-learn: {sklearn.__version__}")
print(f"TensorFlow: {tf.__version__}")
print(f"GPU disponible: {'Sí' if tf.config.list_physical_devices('GPU') else 'No'}")
```

### Paso 3: Explorar Google Colab

Colab tiene muchas ventajas:
- No requiere instalación
- Acceso a GPU gratuita (acelera el entrenamiento)
- Integración con Google Drive
- Fácil de compartir

Para usar GPU:
1. Menú > Entorno de ejecución > Cambiar tipo de entorno de ejecución
2. Selecciona GPU como acelerador por hardware

## Opción 3: Entornos Virtuales (Recomendado para desarrollo local)

Los entornos virtuales te permiten tener múltiples proyectos con diferentes versiones de bibliotecas sin conflictos.

### Paso 1: Instalar virtualenv

```bash
pip install virtualenv
```

### Paso 2: Crear un entorno virtual

```bash
# Windows
virtualenv ai-env

# macOS/Linux
python -m virtualenv ai-env
```

### Paso 3: Activar el entorno

#### Windows:
```bash
ai-env\Scripts\activate
```
