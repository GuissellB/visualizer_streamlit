import math
from datetime import datetime

import pandas as pd
import streamlit as st

from pathlib import Path
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)

from app_config import (
    BALANCE_METHOD_OPTIONS,
    CLASS_WEIGHT,
    CV_CRITERION_OPTIONS,
    DATA_PATH,
    DEFAULT_BEST_MODEL_CRITERION,
    DEFAULT_NO_CV_CRITERION,
    NO_CV_CRITERION_OPTIONS,
    N_SPLITS,
    RANDOM_STATE,
    SEMILLAS,
    TARGET_COL,
)
from ml_toolkit import EDAExplorer, DataPreparer, SupervisedRunner, get_positive_score
from model_config import MODEL_DEFAULT_PARAMS, MODEL_PARAM_SCHEMA, SHARED_PARAM_SCHEMA
from visualizer import Visualizer

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

@st.cache_data
def load_eda_data(path: str):
    # Carga el dataset con EDAExplorer del toolkit y prepara el resumen usado en la página EDA.
    eda = EDAExplorer(path, num=1)
    df = eda.df.copy()

    total_rows = len(df)
    total_cols = df.shape[1]
    null_count = int(eda.valores_faltantes().sum())
    dup_count = int(df.duplicated().sum())

    if TARGET_COL not in df.columns:
        raise ValueError(f"No se encontró la columna target '{TARGET_COL}' en el dataset.")

    class_counts = df[TARGET_COL].value_counts().sort_index()

    phishing_count = int(class_counts.get(-1, 0))
    legit_count = int(class_counts.get(1, 0))

    # Ratio simple entre clases
    ratio_text = f"1:{round(legit_count / phishing_count)}" if phishing_count > 0 else "N/A"

    # Resumen opcional para usar luego
    summary = {
        "total_rows": total_rows,
        "total_cols": total_cols,
        "null_count": null_count,
        "dup_count": dup_count,
        "phishing_count": phishing_count,
        "legit_count": legit_count,
        "ratio_text": ratio_text,
    }

    return df, summary

def crear_modelo(nombre: str, random_state: int):
    # Centraliza la construcción de modelos para reutilizarla en ranking y evaluación detallada.
    if nombre == "Regresión Logística":
        return LogisticRegression(random_state=random_state, **MODEL_DEFAULT_PARAMS[nombre])
    if nombre == "Random Forest":
        return RandomForestClassifier(random_state=random_state, **MODEL_DEFAULT_PARAMS[nombre])
    if nombre == "XGBoost":
        return XGBClassifier(random_state=random_state, **MODEL_DEFAULT_PARAMS[nombre])
    if nombre == "LightGBM":
        return LGBMClassifier(random_state=random_state, **MODEL_DEFAULT_PARAMS[nombre])
    if nombre == "SVM":
        return SVC(random_state=random_state, **MODEL_DEFAULT_PARAMS[nombre])
    raise ValueError(f"Modelo no reconocido: {nombre}")


def parse_optional_value(raw_value: str, value_type: str):
    if raw_value is None:
        return None

    raw_value = str(raw_value).strip()
    if raw_value == "":
        return None

    if value_type == "int":
        return int(raw_value)
    if value_type == "float":
        return float(raw_value)
    return raw_value


def parse_search_values(raw_value: str, value_type: str):
    values = [item.strip() for item in str(raw_value).split(",") if item.strip()]
    if not values:
        return []

    parsed = []
    for value in values:
        if value_type == "int":
            parsed.append(int(value))
        elif value_type == "float":
            parsed.append(float(value))
        else:
            parsed.append(normalize_param_value(value))
    return parsed


def normalize_param_value(value):
    if value == "None":
        return None
    return value


def parse_seed_list(raw_value: str) -> list:
    if raw_value is None:
        return list(SEMILLAS)
    values = [item.strip() for item in str(raw_value).split(",") if item.strip()]
    if not values:
        return list(SEMILLAS)
    return [int(item) for item in values]


def parse_seed_list_with_validation(raw_value: str) -> tuple[list, str | None]:
    if raw_value is None or str(raw_value).strip() == "":
        return list(SEMILLAS), None

    values = [item.strip() for item in str(raw_value).split(",") if item.strip()]
    invalid_values = [item for item in values if not item.lstrip("-").isdigit()]
    if invalid_values:
        return (
            list(SEMILLAS),
            f"Hay valores inválidos en la lista de semillas: {', '.join(invalid_values)}. Usa números enteros separados por comas.",
        )

    return [int(item) for item in values], None


def build_model_params(model_name: str, random_state: int, shared_overrides: dict, model_overrides: dict) -> dict:
    params = dict(MODEL_DEFAULT_PARAMS[model_name])
    compatible_shared = {k: v for k, v in shared_overrides.items() if k in params and v is not None}
    compatible_specific = {k: v for k, v in model_overrides.items() if v is not None}
    params.update(compatible_shared)
    params.update(compatible_specific)
    params["random_state"] = random_state
    return params


def crear_modelo_configurable(nombre: str, random_state: int, shared_overrides: dict, model_overrides: dict):
    params = build_model_params(nombre, random_state, shared_overrides, model_overrides)
    if nombre == "Regresión Logística":
        return LogisticRegression(**params)
    if nombre == "Random Forest":
        return RandomForestClassifier(**params)
    if nombre == "XGBoost":
        return XGBClassifier(**params)
    if nombre == "LightGBM":
        return LGBMClassifier(**params)
    if nombre == "SVM":
        return SVC(**params)
    raise ValueError(f"Modelo no reconocido: {nombre}")


def get_runner_balance_config(balance_method: str) -> dict:
    if balance_method == "class_weight":
        return {"class_weight": CLASS_WEIGHT, "sampling_method": None}
    if balance_method in {"undersample", "oversample", "smote_tomek"}:
        return {"class_weight": None, "sampling_method": balance_method}
    return {"class_weight": None, "sampling_method": None}


def get_available_criteria(use_cv: bool) -> dict:
    return CV_CRITERION_OPTIONS if use_cv else NO_CV_CRITERION_OPTIONS


def get_default_selection_criterion(use_cv: bool) -> str:
    return DEFAULT_BEST_MODEL_CRITERION if use_cv else DEFAULT_NO_CV_CRITERION


def get_selection_criterion_label(selected_criterion: str, use_cv: bool) -> str:
    return get_available_criteria(use_cv).get(selected_criterion, selected_criterion)


def get_balance_display(balance_method: str, model=None) -> dict:
    if balance_method == "class_weight" and model is not None:
        params = model.get_params()

        if "scale_pos_weight" in params and params["scale_pos_weight"] not in (None, 1):
            return {
                "name": "Scale Pos Weight",
                "value": f"{params['scale_pos_weight']:.4f}",
                "description": "Relación clase negativa/positiva para XGBoost",
            }

        return {
            "name": "Class Weight",
            "value": str(params.get("class_weight", CLASS_WEIGHT)),
            "description": "Balance de clases por pesos",
        }

    display_map = {
        "none": {
            "name": "Balanceo",
            "value": BALANCE_METHOD_OPTIONS["none"],
            "description": "Sin ajuste de clases",
        },
        "undersample": {
            "name": "Balanceo",
            "value": BALANCE_METHOD_OPTIONS["undersample"],
            "description": "Reduce ejemplos de la clase mayoritaria",
        },
        "oversample": {
            "name": "Balanceo",
            "value": BALANCE_METHOD_OPTIONS["oversample"],
            "description": "Aumenta ejemplos de la clase minoritaria",
        },
        "smote_tomek": {
            "name": "Balanceo",
            "value": BALANCE_METHOD_OPTIONS["smote_tomek"],
            "description": "Generación sintética y limpieza de clases solapadas",
        },
    }

    return display_map.get(balance_method, display_map["none"])


def render_text_param_input(widget_key: str, param_key: str, config: dict):
    return st.text_input(
        config["label"],
        value="",
        key=widget_key,
        placeholder="Default",
        help=config.get("help", f"Déjalo vacío para usar el valor por defecto de {param_key}."),
    )


def collect_shared_param_overrides() -> dict:
    overrides = {}
    with st.expander("Hiperparámetros compartidos", expanded=False):
        st.caption("Déjalos vacíos para usar los valores por defecto.")
        for param_key, config in SHARED_PARAM_SCHEMA.items():
            raw_value = render_text_param_input(f"shared_{param_key}", param_key, config)
            overrides[param_key] = parse_optional_value(raw_value, config["type"])
    return overrides


def collect_model_param_overrides() -> dict:
    overrides = {}
    with st.expander("Hiperparámetros por modelo", expanded=False):
        st.caption("Solo se aplican al modelo correspondiente. Si no llenas el campo, se conserva el default.")
        for model_name, schema in MODEL_PARAM_SCHEMA.items():
            with st.expander(model_name, expanded=False):
                model_values = {}
                for param_key, config in schema.items():
                    widget_key = f"{model_name}_{param_key}"
                    if config["type"] == "select":
                        value = st.selectbox(
                            config["label"],
                            options=["Default"] + config["options"],
                            key=widget_key,
                        )
                        model_values[param_key] = None if value == "Default" else normalize_param_value(value)
                    elif config["type"] == "select_or_text":
                        value = st.selectbox(
                            config["label"],
                            options=["Default"] + config["options"],
                            key=widget_key,
                        )
                        model_values[param_key] = None if value == "Default" else value
                    else:
                        raw_value = render_text_param_input(widget_key, param_key, config)
                        model_values[param_key] = parse_optional_value(raw_value, config["type"])
                overrides[model_name] = model_values
    return overrides


def get_configured_model(
    nombre: str,
    random_state: int,
    train_size: float,
    df_local: pd.DataFrame,
    target: str,
    balance_method: str,
    shared_param_overrides: dict,
    model_param_overrides: dict,
):
    # Delega al toolkit la aplicación del balanceo y devuelve el modelo ya configurado.
    estandarizar = nombre in ["Regresión Logística", "SVM"]
    balance_config = get_runner_balance_config(balance_method)
    prep = DataPreparer(
        train_size=train_size,
        random_state=random_state,
        scale_X=estandarizar
    )
    model = crear_modelo_configurable(
        nombre,
        random_state=random_state,
        shared_overrides=shared_param_overrides,
        model_overrides=model_param_overrides.get(nombre, {}),
    )
    runner = SupervisedRunner(
        df=df_local,
        target=target,
        model=model,
        task="classification",
        preparer=prep,
        pos_label=1,
        class_weight=balance_config["class_weight"],
        sampling_method=balance_config["sampling_method"],
    )
    return runner.get_model_for_current_split()


@st.cache_data(show_spinner=False)
def load_model_df(csv_name: str) -> pd.DataFrame:
    # Aplica la preparación base con EDAExplorer para que el dashboard use la misma data del modelado.
    base_dir = Path(__file__).resolve().parent
    csv_path = base_dir / csv_name
    if not csv_path.exists():
        raise FileNotFoundError(f"No encontré {csv_name} en {base_dir}")

    eda = EDAExplorer(str(csv_path), num=1)

    # Mantiene la misma preparación que streamlit_app.py para que el dashboard y el app usen la misma base.
    eda.df[TARGET_COL] = pd.to_numeric(eda.df[TARGET_COL], errors="coerce")

    unique_vals = set(eda.df[TARGET_COL].dropna().unique())

    if unique_vals == {-1, 1}:
        eda.df[TARGET_COL] = eda.df[TARGET_COL].map({-1: 0, 1: 1})
    elif unique_vals == {0, 1}:
        pass
    else:
        raise ValueError(f"Valores inesperados en {TARGET_COL}: {sorted(unique_vals)}")

    eda.df[TARGET_COL] = eda.df[TARGET_COL].astype(int)
    eda.eliminarDuplicados().eliminarNulos().analisisCompleto()

    return eda.df


@st.cache_data(show_spinner=False)
def compute_model_results_dashboard(
    csv_name: str,
    target: str,
    random_state: int,
    n_splits: int,
    train_size: float,
    use_cv: bool,
    balance_method: str,
    best_model_criterion: str,
    shared_param_overrides: dict,
    model_param_overrides: dict,
) -> pd.DataFrame:
    # Compara modelos con SupervisedRunner del toolkit y devuelve la tabla base del ranking.
    df_local = load_model_df(csv_name)

    modelos = [
        "Regresión Logística",
        "Random Forest",
        "SVM",
        "XGBoost",
        "LightGBM",
    ]

    resultados = []
    for nombre in modelos:
        estandarizar = nombre in ["Regresión Logística", "SVM"]
        balance_config = get_runner_balance_config(balance_method)
        modelo = crear_modelo_configurable(
            nombre,
            random_state=random_state,
            shared_overrides=shared_param_overrides,
            model_overrides=model_param_overrides.get(nombre, {}),
        )

        prep = DataPreparer(
            train_size=train_size,
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
            # El toolkit decide qué balanceo aplicar según la estrategia seleccionada.
            class_weight=balance_config["class_weight"],
            sampling_method=balance_config["sampling_method"],
        )

        m = runner.evaluate()
        cv = runner.evaluate_cv(n_splits=n_splits) if use_cv else {}

        resultados.append({
            "Modelo": nombre,
            "Accuracy_Global": m.get("Accuracy"),
            "F1_Global": m.get("F1_Pos"),
            "ROC_AUC_Global": m.get("ROC_AUC_Pos"),
            "Accuracy_CV_mean": cv.get("Accuracy"),
            "Accuracy_CV_std": cv.get("Accuracy_std"),
            "F1_CV_mean": cv.get("F1_Pos"),
            "F1_CV_std": cv.get("F1_Pos_std"),
            "ROC_AUC_CV_mean": cv.get("ROC_AUC_Pos"),
            "ROC_AUC_CV_std": cv.get("ROC_AUC_Pos_std"),
        })

    return pd.DataFrame(resultados).sort_values(best_model_criterion, ascending=False).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def compute_best_model_dashboard(
    csv_name: str,
    target: str,
    best_model_name: str,
    seed: int,
    train_size: float,
    balance_method: str,
    shared_param_overrides: dict,
    model_param_overrides: dict,
):
    # Evalúa en detalle el mejor modelo y arma las salidas que luego usa la página de rendimiento.
    df_local = load_model_df(csv_name)

    model = crear_modelo_configurable(
        best_model_name,
        random_state=seed,
        shared_overrides=shared_param_overrides,
        model_overrides=model_param_overrides.get(best_model_name, {}),
    )
    estandarizar = best_model_name in ["Regresión Logística", "SVM"]
    balance_config = get_runner_balance_config(balance_method)

    prep = DataPreparer(
        train_size=train_size,
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
    # El toolkit aplica la estrategia de balanceo seleccionada.
    class_weight=balance_config["class_weight"],
    sampling_method=balance_config["sampling_method"],
    )

    y_pred = runner.fit_predict()

    y_true = runner.y_test
    y_score = get_positive_score(runner.model, runner.X_test)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    rec = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

    try:
        auc = roc_auc_score(y_true, y_score) if y_score is not None else None
    except Exception:
        auc = None

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    roc_payload = None
    pr_payload = None

    if y_score is not None:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_score)

        roc_payload = pd.DataFrame({"fpr": fpr, "tpr": tpr})
        pr_payload = pd.DataFrame({"recall": recall_curve, "precision": precision_curve})

    return {
        "best_model_name": best_model_name,
        "seed": seed,
        "metrics": {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "roc_auc": auc,
            "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
            "error_rate": (fp + fn) / (tp + tn + fp + fn),
        },
        "confusion": {
            "TP": int(tp),
            "FP": int(fp),
            "TN": int(tn),
            "FN": int(fn),
        },
        "roc_df": roc_payload,
        "pr_df": pr_payload,
        "test_size": int(len(y_true)),
    }


@st.cache_data(show_spinner=False)
def compute_stability_dashboard(
    csv_name: str,
    target: str,
    best_model_name: str,
    seeds: tuple,
    train_size: float,
    use_cv: bool,
    n_splits: int,
    balance_method: str,
    shared_param_overrides: dict,
    model_param_overrides: dict,
) -> pd.DataFrame:
    df_local = load_model_df(csv_name)
    results = []

    for seed in seeds:
        model = crear_modelo_configurable(
            best_model_name,
            random_state=seed,
            shared_overrides=shared_param_overrides,
            model_overrides=model_param_overrides.get(best_model_name, {}),
        )
        estandarizar = best_model_name in ["Regresión Logística", "SVM"]
        balance_config = get_runner_balance_config(balance_method)

        prep = DataPreparer(
            train_size=train_size,
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
            class_weight=balance_config["class_weight"],
            sampling_method=balance_config["sampling_method"],
        )

        m = runner.evaluate()
        cv = runner.evaluate_cv(n_splits=n_splits) if use_cv else {}

        results.append({
            "Seed": seed,
            "Accuracy_Global": m.get("Accuracy"),
            "F1_Global": m.get("F1_Pos"),
            "ROC_AUC_Global": m.get("ROC_AUC_Pos"),
            "Accuracy_CV_mean": cv.get("Accuracy"),
            "F1_CV_mean": cv.get("F1_Pos"),
            "ROC_AUC_CV_mean": cv.get("ROC_AUC_Pos"),
        })

    return pd.DataFrame(results).sort_values("Seed").reset_index(drop=True)

st.set_page_config(
    page_title="Phishing Detection Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================
# Datos base (mockup fiel al Figma)
# =========================
preprocessing_steps = [
    ("1. Data Cleaning", "Eliminación de duplicados, URLs inválidas y valores nulos"),
    ("2. URL Parsing", "Extracción de componentes: protocolo, dominio, path, query parameters"),
    ("3. Feature Extraction", "Extracción de 47 features desde URLs y metadata de websites"),
    ("4. Feature Scaling", "StandardScaler para features numéricas"),
    ("5. Feature Selection", "SelectKBest con chi2 y eliminación de features correlacionadas"),
    ("6. Encoding", "One-Hot Encoding para variables categóricas como TLD y protocolo"),
    ("7. Train/Test Split", "80/20 split con estratificación por clase"),
]

feature_importance = pd.DataFrame(
    {
        "feature": [
            "url_length",
            "domain_age",
            "has_https",
            "suspicious_tld",
            "url_entropy",
            "subdomain_count",
            "has_ip_address",
            "special_chars_count",
            "domain_reputation",
            "redirect_count",
        ],
        "importance": [0.94, 0.89, 0.87, 0.82, 0.79, 0.76, 0.73, 0.68, 0.65, 0.61],
    }
)

roc_df = pd.DataFrame(
    {
        "fpr": [0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.13, 1.0],
        "tpr": [0.0, 0.78, 0.89, 0.94, 0.97, 0.985, 0.995, 1.0],
    }
)
pr_df = pd.DataFrame(
    {
        "recall": [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0],
        "precision": [1.0, 0.98, 0.96, 0.94, 0.91, 0.88, 0.85],
    }
)

confusion = {
    "TP": 243,
    "FP": 18,
    "TN": 12189,
    "FN": 4,
}

model_config = {
    "algorithm": "Random Forest Classifier",
    "version": "2.3.1",
    "training_date": "2026-02-15",
    "last_update": "2026-02-21",
    "status": "Producción",
}

hyperparameters = pd.DataFrame(
    [
        ("n_estimators", "500", "Número de árboles en el bosque"),
        ("max_depth", "15", "Profundidad máxima de cada árbol"),
        ("min_samples_split", "5", "Mínimo de muestras para dividir un nodo"),
        ("min_samples_leaf", "2", "Mínimo de muestras en hojas"),
        ("max_features", "sqrt", "Número de features para mejor división"),
        ("bootstrap", "True", "Usar bootstrap para muestras"),
        ("random_state", "42", "Semilla para reproducibilidad"),
        ("class_weight", "balanced", "Balance de clases automático"),
    ],
    columns=["Parámetro", "Valor", "Descripción"],
)

feature_groups = {
    "URL Analysis": [
        "url_length",
        "has_ip_address",
        "suspicious_tld",
        "url_entropy",
        "shortened_url",
        "special_chars_count",
    ],
    "Domain Analysis": [
        "domain_age",
        "domain_reputation",
        "whois_registered",
        "dns_records",
        "subdomain_count",
    ],
    "SSL/Security": [
        "has_https",
        "ssl_certificate_valid",
        "certificate_age",
        "issuer_reputation",
        "self_signed",
    ],
    "Page Content": [
        "page_rank",
        "external_links",
        "forms_count",
        "iframes_count",
        "hidden_elements",
    ],
    "Traffic & Reputation": [
        "alexa_rank",
        "google_index",
        "page_popularity",
        "domain_blacklist",
    ],
    "Behavioral": [
        "redirect_count",
        "right_click_disabled",
        "popup_count",
        "port_abnormal",
        "favicon_match",
    ],
}

training_metrics = [
    ("Training Set Size", "10,034 websites"),
    ("Validation Set Size", "2,509 websites"),
    ("Training Time", "4.3 minutos"),
    ("Cross-Validation Folds", "5-fold"),
    ("CV Mean Accuracy", "98.2%"),
    ("CV Std Deviation", "0.8%"),
]

# =========================
# Cálculos
# =========================
TOTAL = sum(confusion.values())
ACCURACY = ((confusion["TP"] + confusion["TN"]) / TOTAL) * 100
PRECISION = confusion["TP"] / (confusion["TP"] + confusion["FP"]) * 100
RECALL = confusion["TP"] / (confusion["TP"] + confusion["FN"]) * 100
F1 = 2 * (PRECISION * RECALL) / (PRECISION + RECALL)
SPECIFICITY = confusion["TN"] / (confusion["TN"] + confusion["FP"]) * 100
ERROR_RATE = ((confusion["FP"] + confusion["FN"]) / TOTAL) * 100

# =========================
# Estilo
# =========================
st.markdown(
    """
    <style>
    :root {
        --bg: #f4f7ff;
        --card: rgba(255,255,255,0.85);
        --text: #162033;
        --muted: #64748b;
        --line: rgba(15, 23, 42, 0.08);
        --blue: #2563eb;
        --indigo: #4f46e5;
        --purple: #7c3aed;
        --cyan: #06b6d4;
        --green: #10b981;
        --orange: #f59e0b;
        --red: #ef4444;
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(59,130,246,0.18), transparent 28%),
            radial-gradient(circle at top right, rgba(168,85,247,0.18), transparent 26%),
            linear-gradient(180deg, #eff6ff 0%, #f8fafc 100%);
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
        border-right: 1px solid rgba(37, 99, 235, 0.08);
    }

    .hero {
        background: linear-gradient(135deg, #2563eb 0%, #4f46e5 45%, #7c3aed 100%);
        color: white;
        padding: 1.4rem 1.6rem;
        border-radius: 24px;
        box-shadow: 0 18px 45px rgba(79, 70, 229, 0.22);
        margin-bottom: 1rem;
    }

    .hero h1 {
        margin: 0;
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: -0.03em;
    }

    .hero p {
        margin: 0.35rem 0 0 0;
        opacity: 0.92;
        font-size: 0.98rem;
    }

    .metric-card {
        background: var(--card);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.55);
        border-radius: 22px;
        padding: 1rem 1rem 1rem 1rem;
        box-shadow: 0 14px 34px rgba(15, 23, 42, 0.08);
        min-height: 145px;
        margin-bottom: 1rem;
    }

    .metric-label {
        color: var(--muted);
        font-size: 0.9rem;
        margin-bottom: 0.55rem;
        font-weight: 600;
    }

    .metric-value {
        font-size: 1.95rem;
        font-weight: 800;
        line-height: 1.05;
        color: var(--text);
    }

    .metric-sub {
        color: var(--muted);
        font-size: 0.83rem;
        margin-top: 0.45rem;
    }

    .panel {
    background: rgba(255,255,255,0.86);
    border: 1px solid rgba(255,255,255,0.6);
    backdrop-filter: blur(12px);
    border-radius: 24px;
    padding: 1.15rem 1.2rem;
    box-shadow: 0 14px 34px rgba(15, 23, 42, 0.08);
    margin-top: 0.35rem;
    margin-bottom: 1rem;
    }   

    .panel h3 {
        margin-top: 0;
        margin-bottom: 0.25rem;
        color: var(--text);
        font-size: 1.15rem;
        font-weight: 800;
    }

    .panel .sub {
        color: var(--muted);
        font-size: 0.88rem;
        margin-bottom: 0.8rem;
    }

    .step {
        background: linear-gradient(180deg, #f8fbff 0%, #f3f7ff 100%);
        border: 1px solid rgba(37,99,235,0.08);
        border-radius: 18px;
        padding: 0.85rem 1rem;
        margin-bottom: 0.65rem;
    }

    .step-num {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 34px;
        height: 34px;
        background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%);
        color: white;
        border-radius: 999px;
        font-weight: 800;
        margin-right: 0.6rem;
    }

    .pill {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        margin: 0.12rem 0.2rem 0.12rem 0;
        border-radius: 999px;
        background: linear-gradient(135deg, rgba(37,99,235,0.10), rgba(124,58,237,0.12));
        color: #314158;
        border: 1px solid rgba(99,102,241,0.12);
        font-size: 0.8rem;
        font-weight: 600;
    }

    .status-box {
        background: linear-gradient(135deg, rgba(16,185,129,0.12), rgba(37,99,235,0.08));
        border: 1px solid rgba(16,185,129,0.14);
        border-radius: 18px;
        padding: 0.9rem 1rem;
        color: #0f5132;
        font-weight: 600;
    }

    .threshold-card {
        border-radius: 20px;
        padding: 1rem;
        text-align: center;
        border: 1px solid rgba(15,23,42,0.08);
        background: white;
    }

    .threshold-card.active {
        background: linear-gradient(135deg, rgba(37,99,235,0.10), rgba(124,58,237,0.10));
        border: 2px solid rgba(37,99,235,0.5);
    }

    .cm-box {
        border-radius: 24px;
        padding: 1.4rem 1rem;
        text-align: center;
        font-weight: 700;
        min-height: 168px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }

    .small-muted {
        color: var(--muted);
        font-size: 0.8rem;
    }

    div[data-testid="stDataFrame"] div[role="table"] {
        border-radius: 18px;
        overflow: hidden;
    }
    div[data-testid="column"] {
    padding-bottom: 0.4rem;
    }

    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2rem;
    }

    .section-gap {
        height: 14px;
    }
    .chart-shell {
        background: rgba(248, 250, 252, 0.92);
        border: 1px solid rgba(148, 163, 184, 0.14);
        border-radius: 20px;
        padding: 0.35rem 0.8rem 0.6rem 0.8rem;
        margin-top: 0.35rem;
    }

    .chart-divider {
        height: 1px;
        background: linear-gradient(90deg, rgba(37,99,235,0.10), rgba(124,58,237,0.12));
        margin: 0.6rem 0 0.8rem 0;
        border-radius: 999px;
    }

    .metric-panel-value {
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 2.2rem;
        font-weight: 800;
        text-align: center;
        margin-top: 0.8rem;
    }

    .metric-panel {
        min-height: 160px;
        display: flex;
        flex-direction: column;
        padding: 1rem 1.1rem;
    }
    </style>
    """,
    
    unsafe_allow_html=True,
)

# =========================
# Renderiza bloques visuales del dashboard
# =========================
def hero(title: str, subtitle: str) -> None:
    # Renderiza el bloque principal de cabecera en cada página del dashboard.
    st.markdown(
        f"""
        <div class="hero">
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, sub: str, color: str = "#162033") -> None:
    # Renderiza una tarjeta compacta de métrica con el estilo visual del dashboard.
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value" style="color:{color};">{value}</div>
            <div class="metric-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_panel_card(title: str, subtitle: str, value: str, color: str) -> None:
    # Renderiza una tarjeta ampliada para resúmenes y métricas destacadas.
    st.markdown(
        f"""
        <div class="panel metric-panel">
            <h3>{title}</h3>
            <div class="sub">{subtitle}</div>
            <div class="metric-panel-value" style="color:{color};">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def confusion_box(value: int, label: str, percent: float, background: str, border: str, color: str) -> None:
    # Renderiza una celda visual de la matriz de confusión usando HTML/CSS del dashboard.
    st.markdown(
        f"<div class='cm-box' style='background:{background};border:2px solid {border};color:{color};'><div style='font-size:2rem'>{value}</div><div>{label}</div><div class='small-muted'>{percent:.2f}%</div></div>",
        unsafe_allow_html=True,
    )


def panel_open(title: str, subtitle: str = "", extra_class: str = "") -> None:
    # Abre un contenedor visual reutilizable para secciones del dashboard.
    st.markdown(
        f'<div class="panel {extra_class}"><h3>{title}</h3><div class="sub">{subtitle}</div>',
        unsafe_allow_html=True,
    )


def panel_close() -> None:
    # Cierra el contenedor HTML abierto con panel_open.
    st.markdown("</div>", unsafe_allow_html=True)


df_eda, eda_summary = load_eda_data(DATA_PATH)

df_model_base = load_model_df(DATA_PATH)

with st.sidebar:
    # La barra lateral solo controla navegación y configuración del modelado.
    st.markdown("## 🛡️ Phishing Detection")
    st.caption("Dashboard Analytics")
    page = st.radio(
        "Navegación",
        [
            "EDA - Análisis Exploratorio",
            "Rendimiento del Algoritmo",
            "Parametrización del Modelo",
            "Estabilidad del Modelo",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    selected_balance_method = st.selectbox(
        "Método de balanceo",
        options=list(BALANCE_METHOD_OPTIONS.keys()),
        format_func=lambda x: BALANCE_METHOD_OPTIONS[x],
        index=0,
    )
    use_cross_validation = st.toggle("Usar cross-validation", value=True)
    selected_n_splits = st.slider("Folds CV", min_value=3, max_value=10, value=N_SPLITS) if use_cross_validation else 0
    criterion_options = get_available_criteria(use_cross_validation)
    selected_criterion = st.selectbox(
        "Criterio para mejor modelo",
        options=list(criterion_options.keys()),
        format_func=lambda x: criterion_options[x],
        index=list(criterion_options.keys()).index(get_default_selection_criterion(use_cross_validation)),
    )
    with st.expander("Configuración experimental", expanded=False):
        selected_random_state = st.number_input("random_state", min_value=0, value=RANDOM_STATE, step=1)
        selected_train_size = st.slider("train_size", min_value=0.5, max_value=0.9, value=0.75, step=0.05)
    shared_param_overrides = collect_shared_param_overrides()
    model_param_overrides = collect_model_param_overrides()
    st.markdown(
        f"""
        <div class="status-box">
            <div>● Sistema Activo</div>
            <div class="small-muted" style="margin-top:0.35rem;">
                Última actualización: {datetime.now().strftime('%H:%M:%S')}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

model_results_df = compute_model_results_dashboard(
    DATA_PATH,
    TARGET_COL,
    int(selected_random_state),
    selected_n_splits,
    selected_train_size,
    use_cross_validation,
    selected_balance_method,
    selected_criterion,
    shared_param_overrides,
    model_param_overrides,
)
best_model_name = model_results_df.iloc[0]["Modelo"]

best_model_payload = compute_best_model_dashboard(
    DATA_PATH,
    TARGET_COL,
    best_model_name,
    int(selected_random_state),
    selected_train_size,
    selected_balance_method,
    shared_param_overrides,
    model_param_overrides,
)
@st.cache_data(show_spinner=False)
def get_best_model_row(results_df: pd.DataFrame) -> dict:
    return results_df.iloc[0].to_dict()


@st.cache_data(show_spinner=False)
def get_best_model_params_dict(
    best_model_name: str,
    seed: int,
    train_size: float,
    balance_method: str,
    shared_param_overrides: dict,
    model_param_overrides: dict,
) -> pd.DataFrame:
    model = get_configured_model(
        best_model_name,
        random_state=seed,
        train_size=train_size,
        df_local=df_model_base,
        target=TARGET_COL,
        balance_method=balance_method,
        shared_param_overrides=shared_param_overrides,
        model_param_overrides=model_param_overrides,
    )
    params = model.get_params()

    param_whitelist = {
        "Regresión Logística": [
            "C", "solver", "penalty", "max_iter", "class_weight", "random_state"
        ],
        "Random Forest": [
            "n_estimators", "max_depth", "min_samples_split", "min_samples_leaf",
            "max_features", "class_weight", "random_state"
        ],
        "SVM": [
            "C", "kernel", "gamma", "probability", "class_weight", "random_state"
        ],
        "XGBoost": [
            "n_estimators", "max_depth", "learning_rate", "subsample",
            "colsample_bytree", "objective", "eval_metric", "scale_pos_weight", "random_state"
        ],
        "LightGBM": [
            "n_estimators", "learning_rate", "max_depth", "num_leaves",
            "subsample", "colsample_bytree", "objective", "class_weight", "random_state"
        ],
    }

    selected_keys = param_whitelist.get(best_model_name, list(params.keys()))

    rows = []
    if balance_method in {"undersample", "oversample", "smote_tomek"}:
        rows.append({
            "Parámetro": "sampling_method",
            "Valor": balance_method,
        })

    for key in selected_keys:
        if key in params:
            rows.append({
                "Parámetro": key,
                "Valor": str(params[key])
            })

    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def get_real_feature_columns(csv_name: str, target: str) -> list:
    df_local = load_model_df(csv_name)
    return [c for c in df_local.columns if c != target]


def is_genetic_search_available() -> bool:
    try:
        import sklearn_genetic  # noqa: F401
        return True
    except ImportError:
        return False


def build_best_model_search_grid(best_model_name: str) -> tuple[dict, list]:
    param_grid = {}
    errors = []
    defaults = MODEL_DEFAULT_PARAMS.get(best_model_name, {})

    st.caption("Define listas separadas por comas. Si dejas un campo vacío, no se incluye en la búsqueda.")

    with st.expander("Grid de búsqueda", expanded=False):
        for param_key, config in SHARED_PARAM_SCHEMA.items():
            if param_key not in defaults:
                continue
            raw_value = st.text_input(
                f"{config['label']} (grid)",
                value="",
                key=f"search_shared_{best_model_name}_{param_key}",
                placeholder=f"Ej: {defaults.get(param_key)}",
                help="Valores separados por comas.",
            )
            if raw_value.strip():
                try:
                    values = parse_search_values(raw_value, config["type"])
                    if values:
                        param_grid[param_key] = values
                except Exception:
                    errors.append(f"Valores inválidos para {param_key}.")

        for param_key, config in MODEL_PARAM_SCHEMA.get(best_model_name, {}).items():
            raw_value = st.text_input(
                f"{config['label']} (grid)",
                value="",
                key=f"search_model_{best_model_name}_{param_key}",
                placeholder="Ej: valor1, valor2",
                help="Valores separados por comas.",
            )
            if raw_value.strip():
                parse_type = config["type"] if config["type"] in {"int", "float"} else "text"
                try:
                    values = parse_search_values(raw_value, parse_type)
                    if values:
                        param_grid[param_key] = values
                except Exception:
                    errors.append(f"Valores inválidos para {param_key}.")

    return param_grid, errors


def run_best_model_search(
    csv_name: str,
    target: str,
    best_model_name: str,
    seed: int,
    train_size: float,
    balance_method: str,
    shared_param_overrides: dict,
    model_param_overrides: dict,
    search_method: str,
    search_cv: int,
    scoring: str,
    param_grid: dict,
):
    df_local = load_model_df(csv_name)
    estandarizar = best_model_name in ["Regresión Logística", "SVM"]
    balance_config = get_runner_balance_config(balance_method)

    model = crear_modelo_configurable(
        best_model_name,
        random_state=seed,
        shared_overrides=shared_param_overrides,
        model_overrides=model_param_overrides.get(best_model_name, {}),
    )

    prep = DataPreparer(
        train_size=train_size,
        random_state=seed,
        scale_X=estandarizar,
    )

    runner = SupervisedRunner(
        df=df_local,
        target=target,
        model=model,
        task="classification",
        preparer=prep,
        pos_label=1,
        class_weight=balance_config["class_weight"],
        sampling_method=balance_config["sampling_method"],
    )

    evaluator = runner.build_evaluator(scoring=scoring, cv=search_cv)
    search_input = {
        best_model_name: {
            "estimator": runner.get_model_for_current_split(),
            "param_grid": param_grid,
        }
    }

    if search_method == "Genetic":
        search_results = evaluator.genetic_search(search_input)
    else:
        search_results = evaluator.exhaustive_search(search_input)

    result = search_results[best_model_name]
    tuned_model = result["estimator"]
    y_pred = tuned_model.predict(evaluator.X_test)
    y_score = get_positive_score(tuned_model, evaluator.X_test)

    metrics = {
        "Accuracy": float(accuracy_score(evaluator.y_test, y_pred)),
        "Precision": float(precision_score(evaluator.y_test, y_pred, pos_label=1, zero_division=0)),
        "Recall": float(recall_score(evaluator.y_test, y_pred, pos_label=1, zero_division=0)),
        "F1": float(f1_score(evaluator.y_test, y_pred, pos_label=1, zero_division=0)),
        "ROC_AUC": float(roc_auc_score(evaluator.y_test, y_score)) if y_score is not None else None,
    }

    return {
        "best_params": result["best_params"],
        "best_score": result["best_score"],
        "metrics": metrics,
    }

real_metrics = best_model_payload["metrics"]
real_confusion = best_model_payload["confusion"]
real_roc_df = best_model_payload["roc_df"]
real_pr_df = best_model_payload["pr_df"]
best_model_row = get_best_model_row(model_results_df)
best_model_params_df = get_best_model_params_dict(
    best_model_name,
    int(selected_random_state),
    selected_train_size,
    selected_balance_method,
    shared_param_overrides,
    model_param_overrides,
)
configured_best_model = get_configured_model(
    best_model_name,
    int(selected_random_state),
    selected_train_size,
    df_model_base,
    TARGET_COL,
    selected_balance_method,
    shared_param_overrides,
    model_param_overrides,
)
best_model_balance_display = get_balance_display(selected_balance_method, configured_best_model)
real_feature_cols = get_real_feature_columns(DATA_PATH, TARGET_COL)
viz = Visualizer()

# =========================
# Página 1: EDA
# =========================
if page == "EDA - Análisis Exploratorio":
    # Página EDA: usa Visualizer para gráficos y métricas resumidas del dataset cargado.
    hero(
        "Análisis Exploratorio de Datos (EDA)",
        "Insights y patrones identificados en el dataset de websites.",
    )

    c1, c2, c3, c4 = st.columns(4, gap="medium")
    with c1:
        metric_card(
            "Total de Websites",
            f"{eda_summary['total_rows']:,}",
            "Registros en el dataset",
            "#4f46e5",
        )
    with c2:
        metric_card(
            "Ratio Phishing/Legítimo",
            eda_summary["ratio_text"],
            f"{eda_summary['phishing_count']:,} phishing vs {eda_summary['legit_count']:,} legítimos",
            "#2563eb",
        )
    with c3:
        metric_card(
            "Features Extraídas",
            f"{eda_summary['total_cols'] - 1}",
            "Variables predictoras",
            "#06b6d4",
        )
    with c4:
        metric_card(
            "Duplicados",
            f"{eda_summary['dup_count']:,}",
            "Registros duplicados detectados",
            "#7c3aed",
        )

    left, right = st.columns([1.1, 0.9], gap="large")

    with left:
        panel_open(
            "Features más importantes (Top 10)",
            "Correlación absoluta con la variable objetivo (Result).",
        )

        fig = viz.top_target_correlation_bar(
            df_eda,
            target_col=TARGET_COL,
        )

        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        panel_close()

    with right:
        panel_open("Distribución general", "Balance real entre clases del dataset.")

        fig_pie = viz.target_distribution_donut(
            df_eda,
            target_col=TARGET_COL,
        )

        st.plotly_chart(fig_pie, width="stretch", config={"displayModeBar": False})
        panel_close()

    panel_open("Resumen de calidad de datos", "Estado real del dataset cargado.")
    q1, q2, q3 = st.columns(3, gap="medium")

    with q1:
        metric_panel_card("Valores nulos", "Conteo total de valores faltantes.", str(eda_summary["null_count"]), "#10b981")

    with q2:
        metric_panel_card("Duplicados", "Filas repetidas detectadas en el dataset.", f"{eda_summary['dup_count']:,}", "#f59e0b")

    with q3:
        metric_panel_card("Variable objetivo", "Columna usada como clase a predecir.", TARGET_COL, "#2563eb")
    panel_close()

# =========================
# Página 2: Performance
# =========================
elif page == "Rendimiento del Algoritmo":
    # Página de rendimiento: combina métricas, curvas y tablas derivadas del toolkit.
    hero(
        "Rendimiento del Algoritmo",
        f"Métricas reales del mejor modelo seleccionado: {best_model_name}.",
    )

    cols = st.columns(5, gap="medium")
    metrics = [
        ("Exactitud (Accuracy)", f"{real_metrics['accuracy'] * 100:.2f}%", "Desempeño global", "#4f46e5"),
        ("Precisión (Precision)", f"{real_metrics['precision'] * 100:.2f}%", "Control de falsos positivos", "#10b981"),
        ("Sensibilidad (Recall)", f"{real_metrics['recall'] * 100:.2f}%", "Detección de phishing", "#2563eb"),
        ("F1-Score", f"{real_metrics['f1'] * 100:.2f}%", "Balance precisión-recall", "#7c3aed"),
        ("Especificidad", f"{real_metrics['specificity'] * 100:.2f}%", "Detección de legítimos", "#f59e0b"),
    ]
    for col, m in zip(cols, metrics):
        with col:
            metric_card(*m)

    roc_col, pr_col = st.columns(2, gap="large")

    with roc_col:
        panel_open(
            "Curva ROC (Receiver Operating Characteristic)",
            f"AUC (Área Bajo la Curva): {real_metrics['roc_auc']:.3f}" if real_metrics["roc_auc"] is not None else "AUC no disponible",
        )
        if real_roc_df is not None:
            roc_fig = viz.roc_curve_plot(
                real_roc_df,
                auc_value=real_metrics["roc_auc"],
            )
            st.plotly_chart(roc_fig, width="stretch", config={"displayModeBar": False})
        else:
            st.info("No fue posible calcular la curva ROC para este modelo.")
        panel_close()

    with pr_col:
        panel_open("Curva Precision-Recall", "Curva calculada sobre el conjunto de prueba.")
        if real_pr_df is not None:
            pr_fig = viz.precision_recall_plot(
                real_pr_df,
            )
            st.plotly_chart(pr_fig, width="stretch", config={"displayModeBar": False})
        else:
            st.info("No fue posible calcular la curva Precision-Recall para este modelo.")
        panel_close()

    panel_open("Matriz de confusión", f"Basada en {best_model_payload['test_size']:,} registros del conjunto de prueba.")

    left_spacer, center_block, right_spacer = st.columns([0.12, 0.76, 0.12])

    with center_block:
        st.markdown(
            """
            <div style="display:grid;grid-template-columns:180px 1fr 1fr;gap:18px;align-items:center;margin-top:0.5rem;">
                <div></div>
                <div style="text-align:center;font-weight:800;color:#334155;font-size:1rem;">Predicción: Phishing</div>
                <div style="text-align:center;font-weight:800;color:#334155;font-size:1rem;">Predicción: Legítimo</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        row_labels, matrix_cols = st.columns([0.28, 0.72], gap="medium")

        with row_labels:
            st.markdown(
                "<div style='height:168px;display:flex;align-items:center;justify-content:flex-end;font-weight:800;color:#334155;font-size:1rem;'>Real: Phishing</div>",
                unsafe_allow_html=True,
            )
            st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
            st.markdown(
                "<div style='height:168px;display:flex;align-items:center;justify-content:flex-end;font-weight:800;color:#334155;font-size:1rem;'>Real: Legítimo</div>",
                unsafe_allow_html=True,
            )

        with matrix_cols:
            a, b = st.columns(2, gap="medium")

            with a:
                confusion_box(real_confusion["TP"], "Verdaderos Positivos", real_confusion["TP"] / best_model_payload["test_size"] * 100, "#dcfce7", "#22c55e", "#166534")
                st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
                confusion_box(real_confusion["FP"], "Falsos Positivos", real_confusion["FP"] / best_model_payload["test_size"] * 100, "#ffedd5", "#f59e0b", "#9a3412")

            with b:
                confusion_box(real_confusion["FN"], "Falsos Negativos", real_confusion["FN"] / best_model_payload["test_size"] * 100, "#fee2e2", "#ef4444", "#991b1b")
                st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
                confusion_box(real_confusion["TN"], "Verdaderos Negativos", real_confusion["TN"] / best_model_payload["test_size"] * 100, "#dbeafe", "#3b82f6", "#1d4ed8")

    panel_close()

    err1, err2, err3 = st.columns(3, gap="medium")

    with err1:
        metric_panel_card("Falsos Negativos", "Phishing no detectado (crítico).", str(real_confusion["FN"]), "#ef4444")

    with err2:
        metric_panel_card("Falsos Positivos", "Website legítimo marcado como phishing.", str(real_confusion["FP"]), "#f59e0b")

    with err3:
        metric_panel_card("Tasa de Error Total", "Porcentaje total de clasificaciones incorrectas.", f"{real_metrics['error_rate'] * 100:.3f}%", "#2563eb")

    ranking_subtitle = (
        f"Ranking de modelos según {get_selection_criterion_label(selected_criterion, use_cross_validation)}."
    )
    panel_open("Comparación de modelos", ranking_subtitle)
    st.dataframe(model_results_df.round(4), width="stretch", hide_index=True)
    panel_close()
# =========================
# Página 3: Parámetros
# =========================
elif page == "Parametrización del Modelo":
    # Página de parametrización: muestra cómo quedó configurado el modelo ganador.
    hero(
        "Parametrización del Modelo",
        f"Configuración real del mejor modelo seleccionado: {best_model_name}.",
    )

    c1, c2, c3, c4, c5 = st.columns(5, gap="medium")
    with c1:
        winner_caption = "Seleccionado por desempeño CV" if use_cross_validation else "Seleccionado por desempeño global"
        metric_card("Modelo ganador", best_model_name, winner_caption, "#2563eb")
    with c2:
        metric_card("Semilla", str(int(selected_random_state)), "Reproducibilidad", "#7c3aed")
    with c3:
        cv_value = f"{selected_n_splits}-fold" if use_cross_validation else "Desactivado"
        cv_description = "Evaluación cruzada" if use_cross_validation else "Se usan métricas globales"
        metric_card("Cross-Validation", cv_value, cv_description, "#06b6d4")
    with c4:
        metric_card("Train Size", f"{selected_train_size:.2f}", "Proporción usada para entrenamiento", "#f59e0b")
    with c5:
        metric_card(
            best_model_balance_display["name"],
            best_model_balance_display["value"],
            best_model_balance_display["description"],
            "#10b981",
        )

    panel_open("Resumen de selección del modelo", "Información usada para elegir el modelo ganador.")
    s1, s2, s3 = st.columns(3, gap="medium")

    with s1:
        metric_panel_card(
            "Criterio principal",
            "Métrica usada para ordenar el ranking.",
            get_selection_criterion_label(selected_criterion, use_cross_validation),
            "#2563eb",
        )

    with s2:
        metric_2_label = "ROC_AUC_CV_mean" if use_cross_validation else "ROC_AUC_Global"
        metric_2_desc = "Valor promedio del mejor modelo en validación cruzada." if use_cross_validation else "Valor global del mejor modelo."
        metric_2_value = best_model_row["ROC_AUC_CV_mean"] if use_cross_validation else best_model_row["ROC_AUC_Global"]
        metric_panel_card(metric_2_label, metric_2_desc, f"{metric_2_value:.4f}", "#7c3aed")

    with s3:
        metric_3_label = "Accuracy_CV_mean" if use_cross_validation else "Accuracy_Global"
        metric_3_desc = "Exactitud promedio del mejor modelo en validación cruzada." if use_cross_validation else "Exactitud global del mejor modelo."
        metric_3_value = best_model_row["Accuracy_CV_mean"] if use_cross_validation else best_model_row["Accuracy_Global"]
        metric_panel_card(metric_3_label, metric_3_desc, f"{metric_3_value:.4f}", "#06b6d4")
    panel_close()

    panel_open("Hiperparámetros reales del modelo ganador", "Parámetros obtenidos directamente desde get_params().")
    st.dataframe(best_model_params_df, width="stretch", hide_index=True)
    panel_close()

    panel_open("Búsqueda de hiperparámetros", "Optimiza el mejor modelo actual bajo demanda.")
    genetic_available = is_genetic_search_available()
    tuning_method_options = ["Exhaustive"] + (["Genetic"] if genetic_available else [])

    tune_col_1, tune_col_2, tune_col_3 = st.columns(3, gap="medium")
    with tune_col_1:
        tuning_method = st.selectbox("Método de búsqueda", options=tuning_method_options, index=0)
    with tune_col_2:
        tuning_cv = st.slider("Folds para búsqueda", min_value=3, max_value=10, value=5)
    with tune_col_3:
        tuning_scoring = st.selectbox(
            "Scoring",
            options=["f1", "roc_auc", "accuracy"],
            index=0,
        )

    if not genetic_available:
        st.caption("La búsqueda genética se habilita instalando `sklearn-genetic-opt`.")

    search_grid, search_grid_errors = build_best_model_search_grid(best_model_name)
    if search_grid_errors:
        for error in search_grid_errors:
            st.error(error)

    if search_grid:
        st.caption(f"Parámetros incluidos en la búsqueda: {', '.join(search_grid.keys())}")
    else:
        st.info("Agrega al menos un parámetro con varios valores para ejecutar la búsqueda.")

    run_tuning = st.button("Ejecutar búsqueda sobre el mejor modelo", use_container_width=True)

    if run_tuning:
        if search_grid_errors:
            st.error("Corrige los parámetros inválidos antes de ejecutar la búsqueda.")
        elif not search_grid:
            st.warning("Debes indicar al menos un parámetro para buscar.")
        else:
            with st.spinner("Buscando mejores hiperparámetros..."):
                tuning_result = run_best_model_search(
                    DATA_PATH,
                    TARGET_COL,
                    best_model_name,
                    int(selected_random_state),
                    selected_train_size,
                    selected_balance_method,
                    shared_param_overrides,
                    model_param_overrides,
                    tuning_method,
                    tuning_cv,
                    tuning_scoring,
                    search_grid,
                )

            best_score_value = tuning_result["best_score"]
            tuned_metrics = tuning_result["metrics"]
            tuned_params_df = pd.DataFrame(
                [
                    {"Parámetro": key, "Valor": str(value)}
                    for key, value in tuning_result["best_params"].items()
                ]
            )

            tuning_metrics_cols = st.columns(5, gap="medium")
            tuning_cards = [
                ("Best CV Score", f"{best_score_value:.4f}", f"Scoring: {tuning_scoring}", "#2563eb"),
                ("Accuracy", f"{tuned_metrics['Accuracy']:.4f}", "Conjunto de prueba", "#7c3aed"),
                ("Precision", f"{tuned_metrics['Precision']:.4f}", "Conjunto de prueba", "#10b981"),
                ("Recall", f"{tuned_metrics['Recall']:.4f}", "Conjunto de prueba", "#06b6d4"),
                ("F1", f"{tuned_metrics['F1']:.4f}", "Conjunto de prueba", "#f59e0b"),
            ]
            for col, card in zip(tuning_metrics_cols, tuning_cards):
                with col:
                    metric_card(*card)

            st.markdown("#### Mejores parámetros encontrados")
            st.dataframe(tuned_params_df, width="stretch", hide_index=True)
    panel_close()

# =========================
# Página 4: Estabilidad
# =========================
else:
    hero(
        "Estabilidad del Modelo",
        f"Variación del mejor modelo seleccionado ({best_model_name}) a través de distintas semillas.",
    )

    if "stability_seed_text" not in st.session_state:
        st.session_state["stability_seed_text"] = ", ".join(str(seed) for seed in SEMILLAS)

    seed_text = st.text_input(
        "Semillas para estabilidad",
        key="stability_seed_text",
        help="Separadas por comas. Se usan solo en esta página para el análisis de estabilidad.",
    )
    selected_seed_list, seed_validation_error = parse_seed_list_with_validation(seed_text)
    if seed_validation_error:
        st.error(seed_validation_error)
        st.caption(f"Usando temporalmente la lista por defecto: {', '.join(str(seed) for seed in SEMILLAS)}")
    stability_df = compute_stability_dashboard(
        DATA_PATH,
        TARGET_COL,
        best_model_name,
        tuple(selected_seed_list),
        selected_train_size,
        use_cross_validation,
        selected_n_splits,
        selected_balance_method,
        shared_param_overrides,
        model_param_overrides,
    )

    stability_metric_col_1, stability_metric_col_2, stability_metric_col_3 = st.columns(3, gap="medium")

    if use_cross_validation:
        stability_primary_series = stability_df["ROC_AUC_CV_mean"].dropna()
        stability_secondary_series = stability_df["Accuracy_CV_mean"].dropna()
        stability_primary_label = "ROC AUC CV"
        stability_secondary_label = "Accuracy CV"
    else:
        stability_primary_series = stability_df["ROC_AUC_Global"].dropna()
        stability_secondary_series = stability_df["Accuracy_Global"].dropna()
        stability_primary_label = "ROC AUC Global"
        stability_secondary_label = "Accuracy Global"

    with stability_metric_col_1:
        metric_card("Modelo evaluado", best_model_name, "Mejor modelo actual", "#2563eb")
    with stability_metric_col_2:
        metric_card("Semillas", str(len(selected_seed_list)), "Cantidad de ejecuciones evaluadas", "#7c3aed")
    with stability_metric_col_3:
        spread_value = stability_primary_series.std() if not stability_primary_series.empty else 0.0
        metric_card("Desv. estándar", f"{spread_value:.4f}", f"Variación de {stability_primary_label}", "#06b6d4")

    panel_open("Resumen de estabilidad", "Promedio y variación de las métricas por semilla.")
    sum_col_1, sum_col_2, sum_col_3 = st.columns(3, gap="medium")

    with sum_col_1:
        metric_panel_card(
            f"Promedio {stability_primary_label}",
            "Media entre semillas evaluadas.",
            f"{stability_primary_series.mean():.4f}" if not stability_primary_series.empty else "N/A",
            "#2563eb",
        )
    with sum_col_2:
        metric_panel_card(
            f"Promedio {stability_secondary_label}",
            "Comportamiento medio del modelo entre semillas.",
            f"{stability_secondary_series.mean():.4f}" if not stability_secondary_series.empty else "N/A",
            "#7c3aed",
        )
    with sum_col_3:
        metric_panel_card(
            "Semillas usadas",
            "Lista de semillas configuradas para el análisis.",
            ", ".join(str(seed) for seed in selected_seed_list),
            "#10b981",
        )
    panel_close()

    panel_open("Resultados por semilla", "Cada fila representa una corrida completa del mejor modelo con una semilla distinta.")
    st.dataframe(stability_df.round(4), width="stretch", hide_index=True)
    panel_close()

