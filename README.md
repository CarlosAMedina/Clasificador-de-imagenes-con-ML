# Clasificador de Frutas y Verduras

Sistema de clasificación binaria de imágenes usando Machine Learning para distinguir entre frutas y verduras con una interfaz gráfica intuitiva.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)

## Descripción

Este proyecto implementa un clasificador de imágenes que utiliza **Gradient Boosting Classifier** con extracción avanzada de características para distinguir entre frutas y verduras. Incluye técnicas de aumento de datos (data augmentation) para mejorar la precisión del modelo.

### Características principales:

- **Extracción de características avanzadas**: color, textura, forma y estadísticas
- **Data Augmentation**: rotación, flip, ajuste de brillo/saturación, ruido, crop
- **Modelo robusto**: Gradient Boosting con 300 estimadores
- **Interfaz gráfica**: aplicación Tkinter para clasificación en tiempo real
- **Métricas detalladas**: accuracy, precision, recall, F1-score y matriz de confusión

## Instalación

### Prerrequisitos

- Python 3.8 o superior
- pip (gestor de paquetes de Python)

### Pasos de instalación

1. **Clonar el repositorio**
```bash
git clone https://github.com/tu-usuario/clasificador-frutas-verduras.git
cd clasificador-frutas-verduras
```

2. **Crear un entorno virtual (recomendado)**
```bash
python -m venv venv

# Activar en Windows
venv\Scripts\activate

# Activar en Linux/Mac
source venv/bin/activate
```

3. **Instalar dependencias**
```bash
pip install -r requirements.txt
```

## Estructura del Proyecto

```
clasificador-frutas-verduras/
│
├── FruityTrainer.py              # Script para entrenar el modelo
├── FruityTester.py               # Aplicación GUI para clasificar imágenes
├── fruit_classifier_utils.py    # Utilidades: FeatureExtractor y DataAugmentation
├── requirements.txt              # Dependencias del proyecto
├── README.md                     # Documentación del proyecto
├── .gitignore                    # Archivos ignorados por Git
│
├── VeggiesFruits/               # Dataset de entrenamiento
│   ├── Frutas/                  # Imágenes de frutas
│   └── Verduras/                # Imágenes de verduras
│
└── improved_classifier.pkl      # Modelo entrenado (generado)
```

## Uso

### 1. Preparar el Dataset

Organiza tus imágenes en la siguiente estructura:

```
VeggiesFruits/
├── Frutas/
│   ├── imagen1.jpg
│   ├── imagen2.png
│   └── ...
└── Verduras/
    ├── imagen1.jpg
    ├── imagen2.png
    └── ...
```

Formatos soportados: `.jpg`, `.jpeg`, `.png`

### 2. Entrenar el Modelo

```bash
python FruityTrainer.py
```

Esto generará:
- Métricas de entrenamiento, validación y test
- Matriz de confusión
- Análisis de confianza
- Archivo `improved_classifier.pkl` con el modelo entrenado

**Parámetros configurables** en `FruityTrainer.py`:
- `img_size`: Tamaño de las imágenes (default: 100x100)
- `augmentation_factor`: Factor de aumento de datos (default: 3)
- `test_size`: Proporción del conjunto de prueba (default: 0.2)
- `val_size`: Proporción del conjunto de validación (default: 0.1)

### 3. Clasificar Imágenes (Interfaz Gráfica)

```bash
python FruityTester.py
```

**Funcionalidades de la GUI:**
1. Click en "Browse Image" para seleccionar una imagen
2. Click en "Classify" para obtener la predicción
3. Visualiza resultados con:
   - Categoría predicha (Fruta o Verdura)
   - Nivel de confianza
   - Probabilidades individuales
   - Barra de confianza visual

## Tecnologías y Algoritmos

### Extracción de Características

El sistema extrae **múltiples tipos de características**:

1. **Color Features** (30 dimensiones):
   - Media y desviación estándar por canal RGB
   - Histogramas de color normalizados

2. **Texture Features** (6 dimensiones):
   - Bordes Sobel (horizontal y vertical)
   - Filtro Laplaciano

3. **Shape Features** (4 dimensiones):
   - Área normalizada
   - Perímetro normalizado
   - Circularidad
   - Aspect ratio

4. **Statistical Features** (6 dimensiones):
   - Media, mediana, min, max, std
   - Skewness

### Data Augmentation

Para aumentar la diversidad del dataset:
- Rotación aleatoria (-30° a +30°)
- Flip horizontal
- Ajuste de brillo (0.7x - 1.3x)
- Ajuste de saturación (0.7x - 1.3x)
- Ruido gaussiano
- Crop aleatorio y resize

### Modelo de Clasificación

**Gradient Boosting Classifier** con hiperparámetros optimizados:
- 300 estimadores
- Learning rate: 0.05
- Max depth: 6
- Subsample: 0.8

## Resultados Esperados

Con un dataset balanceado y diverso, el modelo puede alcanzar:
- **Accuracy**: 75-90%
- **Confianza promedio**: 80-90%
- **F1-Score**: 0.75-0.90 para ambas clases

## Personalización

### Ajustar el modelo

Edita los parámetros en `FruityTrainer.py`:

```python
classifier = ImprovedFruitClassifier(
    img_size=(100, 100),           # Tamaño de imagen
    augmentation_factor=3          # Factor de aumento
)
```

### Cambiar el modelo de ML

En la clase `ImprovedFruitClassifier`, reemplaza:

```python
from sklearn.ensemble import RandomForestClassifier

self.model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)
```

## Dependencias

- **numpy**: Operaciones numéricas y arrays
- **scikit-learn**: Modelos de machine learning
- **opencv-python**: Procesamiento de imágenes
- **Pillow**: Manejo de imágenes para GUI
- **tkinter**: Interfaz gráfica (incluido en Python)

## Solución de Problemas

### Error: "No module named 'fruit_classifier_utils'"
Asegúrate de que el archivo `fruit_classifier_utils.py` esté en el mismo directorio que los scripts principales.

### Error: "No images found"
Verifica que:
1. La carpeta `VeggiesFruits` existe
2. Contiene subcarpetas `Frutas/` y `Verduras/`
3. Las imágenes tienen extensiones `.jpg`, `.jpeg` o `.png`

### Baja precisión del modelo
- Aumenta el `augmentation_factor` (3-5)
- Colecta más imágenes diversas
- Balancea el número de imágenes por clase
- Aumenta `img_size` (150x150 o 200x200)

## Autor

**Carlos Armando Medina**

