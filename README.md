# Clasificador de prendas (Fashion-MNIST)

Este repositorio contiene un **clasificador de imágenes de ropa** entrenado con **TensorFlow/Keras** sobre el dataset **Fashion-MNIST**, y una **app web con Streamlit** para probar el modelo. Se desarrolló en el contexto de la evaluación del **Módulo 8 — Fundamentos de Deep Learning** del curso de Ciencia de Datos.

---

## Índice

1. [Qué hace el proyecto](#qué-hace-el-proyecto)
2. [Datos](#datos)
3. [Modelos entrenados](#modelos-entrenados)
4. [Notebook y salidas](#notebook-y-salidas)
5. [Aplicación Streamlit](#aplicación-streamlit)
6. [Estructura del repo](#estructura-del-repo)
7. [Requisitos (dependencias)](#requisitos-dependencias)
8. [Instalación y uso](#instalación-y-uso)

---

## Qué hace el proyecto

- Carga **Fashion-MNIST** desde los archivos binarios **IDX** (imágenes 28×28 en escala de grises, leídas con NumPy).
- Entrena una **red densa** sobre vectores aplanados (784 neuronas de entrada) con **Dropout**, **ReLU** y salida **softmax** para 10 clases; se comparan entrenamientos con distinta cantidad de **epochs** y se evalúa con **accuracy** en test.
- Entrena una **CNN** (bloques `Conv2D` + `MaxPooling2D`, capas densas finales y **Dropout**), que obtiene mejor desempeño que el modelo puramente denso; el resultado se guarda como `fashion_model.h5`.
- Ofrece una interfaz para **subir una imagen** o **recorrer el conjunto de prueba local** y ver **predicción** y **confianza**.

---

## Datos

**Fashion-MNIST** incluye 10 categorías (camiseta, pantalón, pulóver, vestido, abrigo, sandalia, camisa, zapato deportivo, bolso, bota). En la app las etiquetas se muestran en inglés (`T-shirt`, `Trouser`, etc.), alineadas con las convenciones habituales del dataset.

Los archivos crudos deben vivir en `data/` (por ejemplo `t10k-images-idx3-ubyte` y `t10k-labels-idx1-ubyte` para el test que usa la app).

---

## Modelos entrenados

| Enfoque | Idea clave |
|--------|------------|
| **Red densa** | Imagen aplanada a 784 valores → capas fully connected (p. ej. 128 → 64) con Dropout → 10 salidas softmax. Optimizador Adam, pérdida `sparse_categorical_crossentropy`. |
| **CNN** | Entrada 28×28×1 → convoluciones y pooling para patrones espaciales → `Flatten` → capas densas con Dropout → softmax. Es el modelo exportado en `fashion_model.h5` usado por Streamlit. |

---

## Notebook y salidas

El archivo **`main.ipynb`** concentra el flujo completo: carga de datos, normalización, definición de arquitecturas, entrenamiento con validación, evaluación en test y comparación entre configuraciones. También se generan figuras en **`images/`** (por ejemplo `images/accuracy.png`, `images/accuracy_2.png` y `images/comparacion_modelos.png`) para visualizar curvas de entrenamiento y validación.

---

## Aplicación Streamlit

**`app.py`** carga `fashion_model.h5`, normaliza píxeles a \([0,1]\) y aplica inferencia sobre:

- **Subir imagen**: se convierte a escala de grises, se redimensiona a 28×28 y se muestra clase predicha y confianza.
- **Imagen de prueba**: lee el test desde `data/` y permite navegar con slider y botones, mostrando además la **etiqueta real** frente a la predicción.

---

## Estructura del repo

| Archivo / carpeta | Rol |
|-------------------|-----|
| `main.ipynb` | Entrenamiento, métricas y comparación densa vs CNN |
| `app.py` | Interfaz Streamlit |
| `fashion_model.h5` | CNN guardada para producción local |
| `images/` | Figuras exportadas por el notebook (curvas, predicciones) |
| `data/` | Archivos IDX de Fashion-MNIST |
| `requirements.txt` | Dependencias Python |

---

## Requisitos (dependencias)

Instala todo con `pip install -r requirements.txt`. Paquetes incluidos:

| Paquete | Uso |
|---------|-----|
| **tensorflow** | Keras: entrenamiento e inferencia |
| **numpy** | Datos y tensores |
| **matplotlib** | Gráficos de entrenamiento en el notebook |
| **streamlit** | Interfaz web `app.py` |
| **pillow** | Carga y preprocesado de imágenes en Streamlit |
| **jupyter** | Ejecutar y editar `main.ipynb` |

Archivo fuente: **`requirements.txt`** en la raíz del repositorio.

---

## Instalación y uso

**Entorno:** Python 3.10+ recomendado.

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Notebook:** desde la raíz del proyecto:

```bash
jupyter notebook main.ipynb
```

**App Streamlit** (con `fashion_model.h5` y `data/` en su sitio):

```bash
streamlit run app.py
```

Abre el navegador en `http://localhost:8501` (puerto por defecto de Streamlit).
