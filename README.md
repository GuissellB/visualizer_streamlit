# Herramientas de Minería de Datos y Visualización ML

> **Producto final del curso de Minería de Datos Avanzada**
>
> Desarrollado por **Guissell Betancur Oviedo** y **Anyelin Arias Camacho**

---

## Descripción general

Este repositorio contiene todos los entregables del curso de Minería de Datos Avanzada, organizados como una serie de casos de estudio progresivos que culminan en un **toolkit reutilizable de ML** y una **interfaz interactiva para explorar cualquier dataset**.

El producto central del proyecto son los tres archivos en la raíz: `ml_toolkit.py`, `visualizer.py` y `explorador_ml.py`. Los casos de estudio y el proyecto final son entregas académicas que consumen ese mismo toolkit.

---

## Estructura del repositorio

```
visualizer_streamlit/
│
├── ml_toolkit.py           ← Toolkit principal de ML  (PRODUCTO CENTRAL)
├── visualizer.py           ← Librería de visualizaciones con Plotly
├── explorador_ml.py        ← Explorador interactivo para cualquier dataset
├── requirements.txt        ← Dependencias del proyecto
│
├── Caso_estudio_1/         ← Entregable 1: Predicción de consumo de agua
├── Caso_estudio_2/         ← Entregable 2: Web Mining + Redes Neuronales
├── Proyecto_Final/         ← Entregable 3: Dashboard de clasificación avanzado
└── Pruebas/                ← Interfaces experimentales (no entregable)
```

---

## Instalación

```bash
pip install -r requirements.txt
```

Se recomienda usar un entorno virtual:

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate

pip install -r requirements.txt
```

---

## Producto principal (raíz del proyecto)

### `ml_toolkit.py` — Toolkit de Machine Learning

Librería central del proyecto. Contiene todas las clases que implementan los flujos de ML utilizados por las interfaces Streamlit:

| Clase | Responsabilidad |
|---|---|
| `DataPreparer` | Carga, limpieza, codificación y escalado de datos |
| `SupervisedRunner` | Clasificación y regresión con validación cruzada |
| `NeuralNetworkRunner` | Entrenamiento y evaluación de redes neuronales |
| `UnsupervisedRunner` | Clustering (KMeans, Agglomerative) |
| `TimeSeriesRunner` | Evaluación de modelos de series de tiempo |
| `ARIMAForecaster` | Pronóstico con modelos ARIMA |
| `HoltWintersForecaster` | Suavizamiento exponencial (Holt-Winters) |
| `EDAExplorer` | Análisis exploratorio de datos |
| `AssociationRulesExplorer` | Reglas de asociación (market basket) |
| `WebMiningToolkit` | Web scraping con BeautifulSoup y Selenium |
| `ModelEvaluator` | Cálculo de métricas de evaluación |

---

### `visualizer.py` — Librería de visualizaciones

Wrapper de Plotly con métodos reutilizables para todos los dashboards del proyecto: gráficos de barras, donut charts, curvas ROC, mapas de calor de correlación, series de tiempo, matrices de confusión, entre otros.

---

### `explorador_ml.py` — Explorador interactivo de datasets

Interfaz Streamlit que permite cargar **cualquier archivo CSV** y explorarlo con todas las capacidades del toolkit: EDA automático, aprendizaje supervisado y no supervisado, series de tiempo, redes neuronales, reglas de asociación y web mining.

**Ejecutar:**

```bash
streamlit run explorador_ml.py
```

---

## Entregables del curso

Los tres entregables a continuación fueron desarrollados en ese orden durante el curso. Cada uno tiene su propia interfaz Streamlit y utiliza el toolkit de la raíz.

---

### Caso de Estudio 1 — Predicción de consumo de agua

**Carpeta:** `Caso_estudio_1/`

Dashboard para análisis y pronóstico de una serie de tiempo de consumo de agua. Aplica modelos ARIMA, Holt-Winters y redes neuronales, con visualización interactiva del EDA y comparación de métricas entre modelos.

**Dato utilizado:** `consumo_agua.csv`

**Ejecutar:**

```bash
streamlit run Caso_estudio_1/water_consumption_streamlit_p.py
```

---

### Caso de Estudio 2 — Web Mining, Reglas de Asociación y Redes Neuronales

**Carpeta:** `Caso_estudio_2/`

Pipeline completo que integra tres técnicas avanzadas:

1. **Web scraping** de un catálogo de productos (BeautifulSoup / Selenium)
2. **Minería de reglas de asociación** sobre transacciones sintéticas generadas a partir de los productos
3. **Clasificación con redes neuronales** para predecir el segmento de precio de un producto

El pipeline genera artefactos CSV/JSON en `Caso_estudio_2/outputs/` que el dashboard carga para visualizar resultados.

**Paso 1 — Ejecutar el pipeline de datos:**

```bash
python Caso_estudio_2/caso_estudio.py
```

**Paso 2 — Lanzar el dashboard:**

```bash
streamlit run Caso_estudio_2/streamlit_app.py
```

---

### Proyecto Final — Dashboard de clasificación avanzado

**Carpeta:** `Proyecto_Final/`

Dashboard de clasificación de nivel producción con las siguientes capacidades:

- Selección de modelo: Logistic Regression, Random Forest, SVM, XGBoost, LightGBM
- Técnicas de balanceo de clases: `class_weight`, NearMiss, RandomOverSampler, SMOTE+Tomek
- Ajuste de hiperparámetros configurable
- Validación cruzada con múltiples splits
- Comparación de métricas: Accuracy, Precision, Recall, F1, ROC-AUC
- Visualizaciones: curvas ROC, PR, matrices de confusión

La configuración de modelos y parámetros se centraliza en `Proyecto_Final/config.py`.

**Ejecutar:**

```bash
streamlit run Proyecto_Final/dashboard.py
```

---

## Pruebas (no entregable)

**Carpeta:** `Pruebas/`

Interfaces experimentales utilizadas durante el desarrollo para probar funcionalidades del toolkit y el módulo de web mining. No forman parte de los entregables académicos.

---

## Tecnologías utilizadas

- **Framework de interfaces:** Streamlit
- **ML:** scikit-learn, XGBoost, LightGBM, imbalanced-learn, statsmodels
- **Datos:** pandas, numpy, scipy
- **Visualización:** Plotly, seaborn, matplotlib
- **Web scraping:** BeautifulSoup4, Selenium, requests
- **Reglas de asociación:** mlxtend
- **Reducción de dimensionalidad:** UMAP
