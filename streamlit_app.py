import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np

# Tus imports (usa el toolkit limpio si lo estás usando)
# Si tu archivo se llama ml_toolkit_refactored_clean.py, cambia el import a ese.
from ml_toolkit import EDAExplorer, DataPreparer, SupervisedRunner
from visualizer import Visualizer

# Modelos (igual HTML)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Opcionales (si no los tienes instalados, comenta estas 2 líneas y quita los modelos abajo)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# =========================
# CONFIG (igual HTML)
# =========================
CSV_NAME = "Phishing_Websites_Data.csv"
TARGET = "Result"
RANDOM_STATE = 42
N_SPLITS = 10
SEMILLAS = [1, 7, 21, 42, 99]
CLASS_WEIGHT = "balanced"
DEFAULT_BEST_MODEL_CRITERION = "ROC_AUC_CV_mean"
BEST_MODEL_CRITERIA = [
    "ROC_AUC_CV_mean",
    "F1_CV_mean",
    "Accuracy_CV_mean",
    "ROC_AUC_Global",
    "F1_Global",
    "Accuracy_Global",
]


def crear_modelo(nombre: str, random_state: int):
    if nombre == "Regresión Logística":
        return LogisticRegression(random_state=random_state, solver="liblinear")
    if nombre == "Random Forest":
        return RandomForestClassifier(n_estimators=100, max_depth=8, random_state=random_state)
    if nombre == "XGBoost":
        return XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=random_state)
    if nombre == "LightGBM":
        return LGBMClassifier(random_state=random_state)
    if nombre == "SVM":
        return SVC(probability=True, random_state=random_state)
    raise ValueError("Modelo no reconocido")


def _score_positivo(model, X):
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1]
        return np.ravel(proba)
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return None

st.set_page_config(page_title="EDA + Modelos (HTML replica)", layout="wide")
st.title("EDA + Resultados de Modelos")

menu = st.sidebar.radio("Menú", ["EDA", "Resultados modelos"])

# =========================
# Cargar y preparar (cacheado)
# =========================
@st.cache_data(show_spinner=False)
def load_and_prepare(csv_name: str) -> pd.DataFrame:
    base_dir = Path(__file__).resolve().parent
    csv_path = base_dir / csv_name
    if not csv_path.exists():
        raise FileNotFoundError(f"No encontré {csv_name} en {base_dir}")

    eda = EDAExplorer(str(csv_path), num=1)

    # Mapeo del target (HTML)
    eda.df[TARGET] = eda.df[TARGET].map({-1: 0, 1: 1}).astype(int)

    # Limpieza (HTML)
    eda.valores_faltantes()
    eda.eliminarDuplicados()
    eda.eliminarNulos()

    # Dummies / conversión categóricas (HTML)
    eda.analisisCompleto()

    return eda.df


# =========================
# Entrenamiento/Evaluación (cacheado)
# NOTE: No pasamos df directamente para evitar problemas de hashing;
#       lo reconstruimos dentro usando csv_name + params.
# =========================
@st.cache_data(show_spinner=False)
def compute_model_results(
    csv_name: str, target: str, random_state: int, n_splits: int, best_model_criterion: str
) -> pd.DataFrame:
    df_local = load_and_prepare(csv_name)

    modelos = [
        ("Regresión Logística",
         LogisticRegression(random_state=random_state, solver="liblinear")),
        ("Random Forest",
         RandomForestClassifier(n_estimators=200, max_depth=10, random_state=random_state)),
        ("XGBoost",
         XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=random_state)),
        ("LightGBM",
         LGBMClassifier(random_state=random_state)),
        ("SVM",
         SVC(kernel="rbf", probability=True, random_state=random_state)),
    ]

    resultados = []
    for nombre, modelo in modelos:
        estandarizar = nombre in ["Regresión Logística", "SVM"]

        prep = DataPreparer(
            train_size=0.75,
            random_state=random_state,
            scale_X=estandarizar
        )

        runner = SupervisedRunner(
            df=df_local,
            target=target,
            model=modelo,
            task="classification",
            preparer=prep,
            pos_label=1,
            class_weight=CLASS_WEIGHT,
        )

        m = runner.evaluate()
        cv = runner.evaluate_cv(n_splits=n_splits)

        resultados.append({
            "Modelo": nombre,

            # ===== NORMAL =====
            "Accuracy_Global": m.get("Accuracy"),
            "F1_Global": m.get("F1_Pos"),
            "ROC_AUC_Global": m.get("ROC_AUC_Pos"),

            # ===== K-FOLD =====
            "Accuracy_CV_mean": cv.get("Accuracy"),
            "Accuracy_CV_std": cv.get("Accuracy_std"),

            "F1_CV_mean": cv.get("F1_Pos"),
            "F1_CV_std": cv.get("F1_Pos_std"),

            "ROC_AUC_CV_mean": cv.get("ROC_AUC_Pos"),
            "ROC_AUC_CV_std": cv.get("ROC_AUC_Pos_std"),
        })

    return pd.DataFrame(resultados).sort_values(best_model_criterion, ascending=False).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def compute_stability(csv_name: str, target: str, best_model_name: str, seeds: list, n_splits: int) -> pd.DataFrame:
    df_local = load_and_prepare(csv_name)

    resultados_stab = []
    for seed in seeds:
        model = crear_modelo(best_model_name, random_state=seed)
        estandarizar = best_model_name in ["Regresión Logística", "SVM"]

        prep = DataPreparer(
            train_size=0.75,
            random_state=seed,
            scale_X=estandarizar
        )

        runner = SupervisedRunner(
            df=df_local,
            target=target,
            model=model,
            task="classification",
            preparer=prep,
            pos_label=1,
            class_weight=CLASS_WEIGHT,
        )

        m = runner.evaluate()
        cv = runner.evaluate_cv(n_splits=n_splits)

        # EXACTO como el HTML (solo columnas que tú agregas)
        resultados_stab.append({
            "Seed": seed,
            "Accuracy": m.get("Accuracy"),
            "Error": m.get("Error"),
            "Recall": m.get("Recall_Pos"),
            "Precision": m.get("Precision_Pos"),
            "F1": m.get("F1_Pos"),
            "ROC_AUC": m.get("ROC_AUC_Pos"),
            "Accuracy_CV_mean": cv.get("Accuracy"),
            "F1_CV_mean": cv.get("F1_Pos"),
            "ROC_AUC_CV_mean": cv.get("ROC_AUC_Pos"),
        })

    return pd.DataFrame(resultados_stab).sort_values("F1", ascending=False).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def compute_best_model_balance_compare(
    csv_name: str, target: str, best_model_name: str, seed: int
):
    df_local = load_and_prepare(csv_name)
    estandarizar = best_model_name in ["Regresión Logística", "SVM"]

    scenarios = [
        ("Sin balanceo", None),
        ("Con balanceo", CLASS_WEIGHT),
    ]

    resultados = []
    curvas = {}
    y_true_roc = None

    for nombre_escenario, cw in scenarios:
        model = crear_modelo(best_model_name, random_state=seed)
        prep = DataPreparer(
            train_size=0.75,
            random_state=seed,
            scale_X=estandarizar
        )
        runner = SupervisedRunner(
            df=df_local,
            target=target,
            model=model,
            task="classification",
            preparer=prep,
            pos_label=1,
            class_weight=cw,
        )

        metricas = runner.evaluate()
        resultados.append({
            "Escenario": nombre_escenario,
            "Accuracy": metricas.get("Accuracy"),
            "Recall": metricas.get("Recall_Pos"),
            "Precision": metricas.get("Precision_Pos"),
            "F1": metricas.get("F1_Pos"),
            "ROC_AUC": metricas.get("ROC_AUC_Pos"),
        })

        y_score = _score_positivo(runner.model, runner.X_test)
        if y_score is not None:
            curvas[nombre_escenario] = (runner.y_test, y_score)
            y_true_roc = runner.y_test

    return pd.DataFrame(resultados), curvas, y_true_roc


# =========================
# Datos + Visualizer
# =========================
df = load_and_prepare(CSV_NAME)
viz = Visualizer()

# =========================
# EDA (hallazgos)
# =========================
if menu == "EDA":
    st.header("EDA (Hallazgos)")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Distribución de la clase")
        fig_class = viz.eda_histogramaClase(df, TARGET)
        if fig_class is not None:
            st.pyplot(fig_class)
        else:
            st.info(f"No pude graficar {TARGET} (no existe en df).")

    with col2:
        st.subheader("Correlación vs Target")
        fig_target = viz.eda_graficoCorrelacionTarget(df, TARGET)
        if fig_target is not None:
            st.pyplot(fig_target)
        else:
            st.info("No pude generar la correlación con el target.")

    st.subheader("Mapa de calor de correlaciones")
    st.pyplot(viz.eda_graficoCorrelacion(df))

# =========================
# Modelos (cacheados para NO re-entrenar al cambiar de pestaña)
# =========================
if menu == "Resultados modelos":
    st.header("Resultados de modelos")
    criterio_mejor_modelo = st.selectbox(
        "Criterio para seleccionar mejor modelo",
        options=BEST_MODEL_CRITERIA,
        index=BEST_MODEL_CRITERIA.index(DEFAULT_BEST_MODEL_CRITERION),
    )

    # 1) Tabla comparación (cacheada)
    with st.spinner("Cargando resultados cacheados (si es la primera vez, entrena)..."):
        resultados_df = compute_model_results(
            CSV_NAME, TARGET, RANDOM_STATE, N_SPLITS, criterio_mejor_modelo
        )

    st.subheader("Tabla de comparación de modelos")
    st.dataframe(resultados_df.round(4), use_container_width=True)

    resumen_criterios = []
    for crit in BEST_MODEL_CRITERIA:
        if crit in resultados_df.columns:
            top = resultados_df.sort_values(crit, ascending=False).iloc[0]
            resumen_criterios.append(
                {
                    "Criterio": crit,
                    "Mejor modelo": top["Modelo"],
                    "Valor": top[crit],
                }
            )
    if resumen_criterios:
        st.markdown("### Variación del mejor modelo según criterio")
        st.dataframe(pd.DataFrame(resumen_criterios).round(4), use_container_width=True)

    mejor_nombre = resultados_df.iloc[0]["Modelo"]
    st.success(f"Mejor modelo según {criterio_mejor_modelo}: {mejor_nombre}")

    # 2) Estabilidad (cacheada)
    st.subheader("Estabilidad por semilla (mejor modelo)")
    with st.spinner("Cargando estabilidad cacheada (si es la primera vez, calcula)..."):
        stab_df = compute_stability(CSV_NAME, TARGET, mejor_nombre, SEMILLAS, N_SPLITS)

    st.markdown("### --- ESTABILIDAD POR SEMILLA ---")
    st.dataframe(stab_df.round(4), use_container_width=True)

    st.markdown("### === RESUMEN ESTADÍSTICO ===")
    st.dataframe(stab_df.describe().round(4), use_container_width=True)

    mejor_seed = int(stab_df.iloc[0]["Seed"])
    st.info(f"Mejor semilla según estabilidad: {mejor_seed}")

    st.subheader("Mejor modelo (mejor semilla): con vs sin balanceo")
    with st.spinner("Comparando mejor modelo (mejor semilla) con y sin balanceo..."):
        balance_df, curvas_roc, _ = compute_best_model_balance_compare(
            CSV_NAME, TARGET, mejor_nombre, mejor_seed
        )

    st.dataframe(balance_df.round(4), use_container_width=True)

    fig_roc = viz.sup_plot_roc_compare(curvas_roc, title=f"Comparación Curva ROC - {mejor_nombre}")
    if fig_roc is not None:
        st.pyplot(fig_roc)
    else:
        st.info("No fue posible generar curva ROC para este modelo.")
