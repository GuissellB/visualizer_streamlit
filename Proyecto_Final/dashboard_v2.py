import math
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
import streamlit as st

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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

from config import (
    BALANCE_METHOD_OPTIONS,
    CLASS_WEIGHT,
    CONFIG_DIR,
    CV_CRITERION_OPTIONS,
    DATA_PATH,
    DEFAULT_BEST_MODEL_CRITERION,
    DEFAULT_NO_CV_CRITERION,
    NO_CV_CRITERION_OPTIONS,
    N_SPLITS,
    RANDOM_STATE,
    SEMILLAS,
    TARGET_COL,
    TUNING_FLOAT_FACTORS,
    TUNING_INT_STEPS,
    MODEL_DEFAULT_PARAMS,
    MODEL_PARAM_SCHEMA,
    SHARED_PARAM_SCHEMA,
)
from ml_toolkit import EDAExplorer, DataPreparer, SupervisedRunner, get_positive_score
from visualizer import Visualizer

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


# =========================
# Helpers de parseo y modelos
# =========================

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


def get_default_search_text(param_key: str, current_value, value_type: str) -> str:
    if current_value is None or current_value == "":
        return ""
    if value_type == "int":
        value = int(current_value)
        if param_key == "max_iter" or TUNING_INT_STEPS.get(param_key) is None:
            options = sorted({max(50, value // 2), value, value * 2})
        elif param_key == "max_depth":
            if value <= 0:
                options = [value]
            else:
                step = TUNING_INT_STEPS.get(param_key, 1)
                options = sorted({max(1, value - step), value, value + step})
        else:
            step = TUNING_INT_STEPS.get(param_key, 1)
            min_value = 10 if param_key == "n_estimators" else 2 if param_key == "num_leaves" else 1
            options = sorted({max(min_value, value - step), value, value + step})
        return ", ".join(str(option) for option in options)
    if value_type == "float":
        value = float(current_value)
        if 0 < value <= 1:
            factors = TUNING_FLOAT_FACTORS.get("default_fractional", (0.8, 1.0, 1.2))
            options = sorted({
                round(max(0.01, value * factors[0]), 4),
                round(value * factors[1], 4),
                round(min(1.0, value * factors[2]), 4),
            })
        else:
            factors = TUNING_FLOAT_FACTORS.get("default_continuous", (0.5, 1.0, 1.5))
            options = sorted({
                round(max(0.0001, value * factors[0]), 4),
                round(value * factors[1], 4),
                round(value * factors[2], 4),
            })
        return ", ".join(str(option) for option in options)
    return str(current_value)


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
            f"Valores inválidos en semillas: {', '.join(invalid_values)}. Usa enteros separados por comas.",
        )
    return [int(item) for item in values], None


def build_model_params(model_name: str, random_state: int, shared_overrides: dict, model_overrides: dict) -> dict:
    params = dict(MODEL_DEFAULT_PARAMS[model_name])
    compatible_shared = {k: v for k, v in shared_overrides.items() if k in params and v is not None}
    compatible_specific = {k: v for k, v in model_overrides.items() if v is not None}
    params.update(compatible_shared)
    params.update(compatible_specific)
    if model_name == "Random Forest" and params.get("max_depth") == -1:
        params["max_depth"] = None
    if model_name == "XGBoost" and params.get("max_depth") == -1:
        params["max_depth"] = 0
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
        "none": {"name": "Balanceo", "value": BALANCE_METHOD_OPTIONS["none"], "description": "Sin ajuste de clases"},
        "undersample": {"name": "Balanceo", "value": BALANCE_METHOD_OPTIONS["undersample"], "description": "Reduce ejemplos de la clase mayoritaria"},
        "oversample": {"name": "Balanceo", "value": BALANCE_METHOD_OPTIONS["oversample"], "description": "Aumenta ejemplos de la clase minoritaria"},
        "smote_tomek": {"name": "Balanceo", "value": BALANCE_METHOD_OPTIONS["smote_tomek"], "description": "Generación sintética y limpieza de clases solapadas"},
    }
    return display_map.get(balance_method, display_map["none"])


def render_text_param_input(widget_key: str, param_key: str, config: dict, value: str = ""):
    return st.text_input(
        config["label"],
        value=value,
        key=widget_key,
        placeholder="Default",
        help=config.get("help", f"Déjalo vacío para usar el valor por defecto de {param_key}."),
    )


def collect_shared_param_overrides(current_overrides: dict | None = None) -> dict:
    current_overrides = current_overrides or {}
    overrides = {}
    with st.expander("Hiperparámetros compartidos", expanded=False):
        st.caption("Déjalos vacíos para usar los valores por defecto.")
        st.caption("Nota: algunos valores compartidos pueden adaptarse según el modelo.")
        for param_key, config in SHARED_PARAM_SCHEMA.items():
            current_value = current_overrides.get(param_key)
            raw_value = render_text_param_input(
                f"v2_shared_{param_key}", param_key, config,
                value="" if current_value is None else str(current_value),
            )
            overrides[param_key] = parse_optional_value(raw_value, config["type"])
    return overrides


def collect_model_param_overrides(current_overrides: dict | None = None) -> dict:
    current_overrides = current_overrides or {}
    overrides = {}
    with st.expander("Hiperparámetros por modelo", expanded=False):
        st.caption("Solo se aplican al modelo correspondiente.")
        for model_name, schema in MODEL_PARAM_SCHEMA.items():
            with st.expander(model_name, expanded=False):
                model_values = {}
                current_model_values = current_overrides.get(model_name, {})
                for param_key, config in schema.items():
                    widget_key = f"v2_{model_name}_{param_key}"
                    current_value = current_model_values.get(param_key)
                    if config["type"] in {"select", "select_or_text"}:
                        options = ["Default"] + config["options"]
                        selected_value = "Default" if current_value is None else str(current_value)
                        selected_index = options.index(selected_value) if selected_value in options else 0
                        value = st.selectbox(config["label"], options=options, key=widget_key, index=selected_index)
                        model_values[param_key] = None if value == "Default" else normalize_param_value(value)
                    else:
                        raw_value = render_text_param_input(
                            widget_key, param_key, config,
                            value="" if current_value is None else str(current_value),
                        )
                        model_values[param_key] = parse_optional_value(raw_value, config["type"])
                overrides[model_name] = model_values
    return overrides


def is_genetic_search_available() -> bool:
    try:
        import sklearn_genetic  # noqa: F401
        return True
    except ImportError:
        return False


def build_best_model_search_grid(best_model_name: str, current_params: dict) -> tuple[dict, list]:
    param_grid = {}
    errors = []
    st.caption("Los campos cargan un mini rango alrededor del valor actual del mejor modelo.")
    with st.expander("Grid de búsqueda", expanded=False):
        for param_key, config in SHARED_PARAM_SCHEMA.items():
            if param_key not in current_params:
                continue
            default_text = get_default_search_text(param_key, current_params.get(param_key), config["type"])
            raw_value = st.text_input(
                f"{config['label']} (grid)", value=default_text,
                key=f"v2_search_shared_{best_model_name}_{param_key}",
                placeholder=f"Ej: {default_text}", help="Valores separados por comas.",
            )
            try:
                values = parse_search_values(raw_value, config["type"])
                if values:
                    param_grid[param_key] = values
            except Exception:
                errors.append(f"Valores inválidos para {param_key}.")
        for param_key, config in MODEL_PARAM_SCHEMA.get(best_model_name, {}).items():
            parse_type = config["type"] if config["type"] in {"int", "float"} else "text"
            default_text = get_default_search_text(param_key, current_params.get(param_key, ""), parse_type)
            raw_value = st.text_input(
                f"{config['label']} (grid)", value=default_text,
                key=f"v2_search_model_{best_model_name}_{param_key}",
                placeholder=f"Ej: {default_text}", help="Valores separados por comas.",
            )
            try:
                values = parse_search_values(raw_value, parse_type)
                if values:
                    param_grid[param_key] = values
            except Exception:
                errors.append(f"Valores inválidos para {param_key}.")
    return param_grid, errors


# =========================
# Cache — carga y cómputo (genérico)
# =========================

@st.cache_data(show_spinner=False)
def load_eda_data_v2(path: str, target: str):
    eda = EDAExplorer(path, num=1)
    df = eda.df.copy()

    if target not in df.columns:
        raise ValueError(f"No se encontró la columna objetivo '{target}' en el dataset.")

    total_rows = len(df)
    total_cols = df.shape[1]
    null_count = int(eda.valores_faltantes().sum())
    dup_count = int(df.duplicated().sum())
    class_counts = df[target].value_counts().sort_index()

    summary = {
        "total_rows": total_rows,
        "total_cols": total_cols,
        "null_count": null_count,
        "dup_count": dup_count,
        "class_counts": class_counts.to_dict(),
        "n_classes": int(len(class_counts)),
    }
    return df, summary


@st.cache_data(show_spinner=False)
def load_model_df_v2(csv_name: str, target: str) -> pd.DataFrame:
    csv_path = Path(csv_name)
    if not csv_path.is_absolute():
        csv_path = CONFIG_DIR / csv_path.name
    if not csv_path.exists():
        raise FileNotFoundError(f"No encontré {csv_name}")

    eda = EDAExplorer(str(csv_path), num=1)

    if target in eda.df.columns:
        eda.df[target] = pd.to_numeric(eda.df[target], errors="coerce")
        unique_vals = set(eda.df[target].dropna().unique())
        if unique_vals == {-1, 1}:
            eda.df[target] = eda.df[target].map({-1: 0, 1: 1})
        eda.df[target] = eda.df[target].astype("Int64").astype(int, errors="ignore")

    eda.eliminarDuplicados().eliminarNulos().analisisCompleto()
    return eda.df


@st.cache_data(show_spinner=False)
def compute_model_results_v2(
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
    df_local = load_model_df_v2(csv_name, target)
    modelos = ["Regresión Logística", "Random Forest", "SVM", "XGBoost", "LightGBM"]
    resultados = []
    for nombre in modelos:
        estandarizar = nombre in ["Regresión Logística", "SVM"]
        balance_config = get_runner_balance_config(balance_method)
        modelo = crear_modelo_configurable(nombre, random_state=random_state,
                                           shared_overrides=shared_param_overrides,
                                           model_overrides=model_param_overrides.get(nombre, {}))
        prep = DataPreparer(train_size=train_size, random_state=random_state, scale_X=estandarizar)
        runner = SupervisedRunner(
            df=df_local, target=target, model=modelo, task="classification",
            preparer=prep, pos_label=1,
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
def compute_best_model_v2(
    csv_name: str,
    target: str,
    best_model_name: str,
    seed: int,
    train_size: float,
    balance_method: str,
    shared_param_overrides: dict,
    model_param_overrides: dict,
):
    df_local = load_model_df_v2(csv_name, target)
    model = crear_modelo_configurable(best_model_name, random_state=seed,
                                      shared_overrides=shared_param_overrides,
                                      model_overrides=model_param_overrides.get(best_model_name, {}))
    estandarizar = best_model_name in ["Regresión Logística", "SVM"]
    balance_config = get_runner_balance_config(balance_method)
    prep = DataPreparer(train_size=train_size, random_state=seed, scale_X=estandarizar)
    runner = SupervisedRunner(
        df=df_local, target=target, model=model, task="classification",
        preparer=prep, pos_label=1,
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

    roc_payload, pr_payload = None, None
    if y_score is not None:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_score)
        roc_payload = pd.DataFrame({"fpr": fpr, "tpr": tpr})
        pr_payload = pd.DataFrame({"recall": recall_curve, "precision": precision_curve})

    return {
        "best_model_name": best_model_name,
        "seed": seed,
        "metrics": {
            "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
            "roc_auc": auc,
            "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
            "error_rate": (fp + fn) / (tp + tn + fp + fn),
        },
        "confusion": {"TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn)},
        "roc_df": roc_payload,
        "pr_df": pr_payload,
        "test_size": int(len(y_true)),
    }


@st.cache_data(show_spinner=False)
def compute_stability_v2(
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
    df_local = load_model_df_v2(csv_name, target)
    results = []
    for seed in seeds:
        model = crear_modelo_configurable(best_model_name, random_state=seed,
                                          shared_overrides=shared_param_overrides,
                                          model_overrides=model_param_overrides.get(best_model_name, {}))
        estandarizar = best_model_name in ["Regresión Logística", "SVM"]
        balance_config = get_runner_balance_config(balance_method)
        prep = DataPreparer(train_size=train_size, random_state=seed, scale_X=estandarizar)
        runner = SupervisedRunner(
            df=df_local, target=target, model=model, task="classification",
            preparer=prep, pos_label=1,
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


@st.cache_data(show_spinner=False)
def get_real_feature_columns_v2(csv_name: str, target: str) -> list:
    df_local = load_model_df_v2(csv_name, target)
    return [c for c in df_local.columns if c != target]


@st.cache_data(show_spinner=False)
def get_best_model_params_df_v2(
    csv_name: str,
    target: str,
    best_model_name: str,
    seed: int,
    train_size: float,
    balance_method: str,
    shared_param_overrides: dict,
    model_param_overrides: dict,
) -> pd.DataFrame:
    df_local = load_model_df_v2(csv_name, target)
    estandarizar = best_model_name in ["Regresión Logística", "SVM"]
    balance_config = get_runner_balance_config(balance_method)
    prep = DataPreparer(train_size=train_size, random_state=seed, scale_X=estandarizar)
    model = crear_modelo_configurable(best_model_name, random_state=seed,
                                      shared_overrides=shared_param_overrides,
                                      model_overrides=model_param_overrides.get(best_model_name, {}))
    runner = SupervisedRunner(
        df=df_local, target=target, model=model, task="classification",
        preparer=prep, pos_label=1,
        class_weight=balance_config["class_weight"],
        sampling_method=balance_config["sampling_method"],
    )
    configured_model = runner.get_model_for_current_split()
    params = configured_model.get_params()

    param_whitelist = {
        "Regresión Logística": ["C", "solver", "penalty", "max_iter", "class_weight", "random_state"],
        "Random Forest": ["n_estimators", "max_depth", "min_samples_split", "min_samples_leaf", "max_features", "class_weight", "random_state"],
        "SVM": ["C", "kernel", "gamma", "probability", "class_weight", "random_state"],
        "XGBoost": ["n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree", "objective", "eval_metric", "scale_pos_weight", "random_state"],
        "LightGBM": ["n_estimators", "learning_rate", "max_depth", "num_leaves", "subsample", "colsample_bytree", "objective", "class_weight", "random_state"],
    }
    selected_keys = param_whitelist.get(best_model_name, list(params.keys()))
    rows = []
    if balance_method in {"undersample", "oversample", "smote_tomek"}:
        rows.append({"Parámetro": "sampling_method", "Valor": balance_method})
    for key in selected_keys:
        if key in params:
            rows.append({"Parámetro": key, "Valor": str(params[key])})
    return pd.DataFrame(rows)


def run_best_model_search_v2(
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
    df_local = load_model_df_v2(csv_name, target)
    estandarizar = best_model_name in ["Regresión Logística", "SVM"]
    balance_config = get_runner_balance_config(balance_method)
    model = crear_modelo_configurable(best_model_name, random_state=seed,
                                      shared_overrides=shared_param_overrides,
                                      model_overrides=model_param_overrides.get(best_model_name, {}))
    prep = DataPreparer(train_size=train_size, random_state=seed, scale_X=estandarizar)
    runner = SupervisedRunner(
        df=df_local, target=target, model=model, task="classification",
        preparer=prep, pos_label=1,
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
    return {"best_params": result["best_params"], "best_score": result["best_score"], "metrics": metrics}


# =========================
# Configuración de página
# =========================

st.set_page_config(
    page_title="ML Dashboard v2",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================
# CSS
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

    .hero h1 { margin: 0; font-size: 2rem; font-weight: 800; letter-spacing: -0.03em; }
    .hero p { margin: 0.35rem 0 0 0; opacity: 0.92; font-size: 0.98rem; }

    .metric-card {
        background: var(--card);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.55);
        border-radius: 22px;
        padding: 1rem;
        box-shadow: 0 14px 34px rgba(15, 23, 42, 0.08);
        min-height: 145px;
        margin-bottom: 1rem;
    }

    .metric-label { color: var(--muted); font-size: 0.9rem; margin-bottom: 0.55rem; font-weight: 600; }
    .metric-value { font-size: 1.95rem; font-weight: 800; line-height: 1.05; color: var(--text); }
    .metric-sub { color: var(--muted); font-size: 0.83rem; margin-top: 0.45rem; }

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

    .panel h3 { margin-top: 0; margin-bottom: 0.25rem; color: var(--text); font-size: 1.15rem; font-weight: 800; }
    .panel .sub { color: var(--muted); font-size: 0.88rem; margin-bottom: 0.8rem; }

    .step {
        background: linear-gradient(180deg, #f8fbff 0%, #f3f7ff 100%);
        border: 1px solid rgba(37,99,235,0.08);
        border-radius: 18px;
        padding: 0.85rem 1rem;
        margin-bottom: 0.65rem;
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

    .small-muted { color: var(--muted); font-size: 0.8rem; }

    div[data-testid="stDataFrame"] div[role="table"] { border-radius: 18px; overflow: hidden; }
    div[data-testid="column"] { padding-bottom: 0.4rem; }

    .block-container { padding-top: 1.4rem; padding-bottom: 2rem; }

    .chart-shell {
        background: rgba(248, 250, 252, 0.92);
        border: 1px solid rgba(148, 163, 184, 0.14);
        border-radius: 20px;
        padding: 0.35rem 0.8rem 0.6rem 0.8rem;
        margin-top: 0.35rem;
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

    .unsup-placeholder {
        background: linear-gradient(135deg, rgba(37,99,235,0.04), rgba(124,58,237,0.06));
        border: 2px dashed rgba(99,102,241,0.25);
        border-radius: 20px;
        padding: 2rem 1.5rem;
        text-align: center;
        color: var(--muted);
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
    }

    .unsup-placeholder .placeholder-icon { font-size: 2.5rem; margin-bottom: 0.5rem; }
    .unsup-placeholder p { margin: 0; font-size: 0.95rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# Helpers visuales
# =========================

def hero(title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, sub: str, color: str = "#162033") -> None:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value" style="color:{color};">{value}</div>'
        f'<div class="metric-sub">{sub}</div></div>',
        unsafe_allow_html=True,
    )


def metric_panel_card(title: str, subtitle: str, value: str, color: str) -> None:
    st.markdown(
        f'<div class="panel metric-panel"><h3>{title}</h3><div class="sub">{subtitle}</div>'
        f'<div class="metric-panel-value" style="color:{color};">{value}</div></div>',
        unsafe_allow_html=True,
    )


def confusion_box(value: int, label: str, percent: float, background: str, border: str, color: str) -> None:
    st.markdown(
        f"<div class='cm-box' style='background:{background};border:2px solid {border};color:{color};'>"
        f"<div style='font-size:2rem'>{value}</div><div>{label}</div>"
        f"<div class='small-muted'>{percent:.2f}%</div></div>",
        unsafe_allow_html=True,
    )


def panel_open(title: str, subtitle: str = "", extra_class: str = "") -> None:
    st.markdown(
        f'<div class="panel {extra_class}"><h3>{title}</h3><div class="sub">{subtitle}</div>',
        unsafe_allow_html=True,
    )


def panel_close() -> None:
    st.markdown("</div>", unsafe_allow_html=True)


def unsup_placeholder(icon: str, message: str) -> None:
    st.markdown(
        f'<div class="unsup-placeholder"><div class="placeholder-icon">{icon}</div><p>{message}</p></div>',
        unsafe_allow_html=True,
    )


# =========================
# Session state — valores iniciales
# =========================

_SS_DEFAULTS = {
    "v2_uploaded_file_path": None,
    "v2_delimiter": ",",
    "v2_problem_type": "Supervisado",
    "v2_target_col": TARGET_COL,
    "v2_columns": [],
    "applied_balance_method": "none",
    "applied_use_cross_validation": True,
    "applied_n_splits": N_SPLITS,
    "applied_selected_criterion": get_default_selection_criterion(True),
    "applied_random_state": RANDOM_STATE,
    "applied_train_size": 0.75,
    "applied_shared_param_overrides": {},
    "applied_model_param_overrides": {},
    "v2_stability_seed_text": ", ".join(str(s) for s in SEMILLAS),
}
for _k, _v in _SS_DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# =========================
# Sidebar
# =========================

with st.sidebar:
    st.markdown("## 🔬 ML Dashboard v2")
    st.caption("Análisis supervisado y no supervisado")
    st.markdown("---")

    # --- Carga de datos ---
    st.markdown("### Datos")
    uploaded_file = st.file_uploader("Subir CSV", type=["csv"], label_visibility="collapsed")

    delimiter_options = {",": "Coma (,)", ";": "Punto y coma (;)", "\t": "Tabulador (\\t)", "|": "Pipe (|)"}
    selected_delimiter = st.selectbox(
        "Separador",
        options=list(delimiter_options.keys()),
        format_func=lambda x: delimiter_options[x],
        index=list(delimiter_options.keys()).index(st.session_state["v2_delimiter"]),
        key="v2_delimiter_selector",
    )

    if uploaded_file is not None:
        tmp_dir = Path(tempfile.gettempdir()) / "ml_dashboard_v2"
        tmp_dir.mkdir(exist_ok=True)
        tmp_path = str(tmp_dir / uploaded_file.name)
        with open(tmp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        try:
            _preview_df = pd.read_csv(tmp_path, sep=selected_delimiter, nrows=0)
            st.session_state["v2_uploaded_file_path"] = tmp_path
            st.session_state["v2_delimiter"] = selected_delimiter
            st.session_state["v2_columns"] = list(_preview_df.columns)
        except Exception as e:
            st.error(f"No se pudo leer el archivo: {e}")
    else:
        st.session_state["v2_delimiter"] = selected_delimiter

    active_path = st.session_state["v2_uploaded_file_path"] or DATA_PATH
    _is_custom_file = st.session_state["v2_uploaded_file_path"] is not None

    if _is_custom_file:
        st.success(f"Archivo cargado: {Path(active_path).name}")
    else:
        st.caption(f"Usando dataset por defecto: {Path(DATA_PATH).name}")

    # --- Tipo de problema ---
    st.markdown("### Tipo de problema")
    problem_type = st.radio(
        "Tipo de análisis",
        ["Supervisado", "No Supervisado"],
        index=0 if st.session_state["v2_problem_type"] == "Supervisado" else 1,
        label_visibility="collapsed",
        key="v2_problem_type_radio",
    )
    st.session_state["v2_problem_type"] = problem_type

    st.markdown("---")

    # --- Navegación ---
    st.markdown("### Navegación")
    if problem_type == "Supervisado":
        available_columns = st.session_state.get("v2_columns") or [TARGET_COL]
        default_target_idx = available_columns.index(st.session_state["v2_target_col"]) if st.session_state["v2_target_col"] in available_columns else 0
        selected_target = st.selectbox(
            "Variable objetivo",
            options=available_columns,
            index=default_target_idx,
            key="v2_target_selector",
        )
        st.session_state["v2_target_col"] = selected_target
        active_target = selected_target

        page = st.radio(
            "Página",
            ["EDA - Análisis Exploratorio", "Rendimiento del Algoritmo", "Parametrización del Modelo", "Estabilidad del Modelo"],
            label_visibility="collapsed",
            key="v2_page_sup",
        )
    else:
        active_target = st.session_state["v2_target_col"]
        page = st.radio(
            "Página",
            ["EDA - Análisis Exploratorio", "Clustering", "Reducción Dimensional", "Reglas de Asociación"],
            label_visibility="collapsed",
            key="v2_page_unsup",
        )

    st.markdown("---")

    # --- Configuración del modelo (solo Supervisado) ---
    if problem_type == "Supervisado":
        with st.form("v2_sidebar_model_config_form"):
            selected_balance_method = st.selectbox(
                "Método de balanceo",
                options=list(BALANCE_METHOD_OPTIONS.keys()),
                format_func=lambda x: BALANCE_METHOD_OPTIONS[x],
                index=list(BALANCE_METHOD_OPTIONS.keys()).index(st.session_state["applied_balance_method"]),
            )
            use_cross_validation = st.toggle("Usar cross-validation", value=st.session_state["applied_use_cross_validation"])
            selected_n_splits = (
                st.slider("Folds CV", min_value=3, max_value=10, value=st.session_state["applied_n_splits"])
                if use_cross_validation else 0
            )
            criterion_options = get_available_criteria(use_cross_validation)
            default_criterion = st.session_state["applied_selected_criterion"]
            if default_criterion not in criterion_options:
                default_criterion = get_default_selection_criterion(use_cross_validation)
            selected_criterion = st.selectbox(
                "Criterio para mejor modelo",
                options=list(criterion_options.keys()),
                format_func=lambda x: criterion_options[x],
                index=list(criterion_options.keys()).index(default_criterion),
            )
            with st.expander("Configuración experimental", expanded=False):
                selected_random_state = st.number_input("random_state", min_value=0, value=st.session_state["applied_random_state"], step=1)
                selected_train_size = st.slider("train_size", min_value=0.5, max_value=0.9, value=float(st.session_state["applied_train_size"]), step=0.05)
            shared_param_overrides = collect_shared_param_overrides(st.session_state["applied_shared_param_overrides"])
            model_param_overrides = collect_model_param_overrides(st.session_state["applied_model_param_overrides"])
            apply_config = st.form_submit_button("Aplicar configuración", use_container_width=True)

        if apply_config:
            st.session_state["applied_balance_method"] = selected_balance_method
            st.session_state["applied_use_cross_validation"] = use_cross_validation
            st.session_state["applied_n_splits"] = selected_n_splits
            st.session_state["applied_selected_criterion"] = selected_criterion
            st.session_state["applied_random_state"] = int(selected_random_state)
            st.session_state["applied_train_size"] = float(selected_train_size)
            st.session_state["applied_shared_param_overrides"] = shared_param_overrides
            st.session_state["applied_model_param_overrides"] = model_param_overrides

        selected_balance_method = st.session_state["applied_balance_method"]
        use_cross_validation = st.session_state["applied_use_cross_validation"]
        selected_n_splits = st.session_state["applied_n_splits"] if use_cross_validation else 0
        selected_criterion = st.session_state["applied_selected_criterion"]
        selected_random_state = st.session_state["applied_random_state"]
        selected_train_size = st.session_state["applied_train_size"]
        shared_param_overrides = st.session_state["applied_shared_param_overrides"]
        model_param_overrides = st.session_state["applied_model_param_overrides"]
    else:
        selected_balance_method = st.session_state["applied_balance_method"]
        use_cross_validation = st.session_state["applied_use_cross_validation"]
        selected_n_splits = st.session_state["applied_n_splits"] if use_cross_validation else 0
        selected_criterion = st.session_state["applied_selected_criterion"]
        selected_random_state = st.session_state["applied_random_state"]
        selected_train_size = st.session_state["applied_train_size"]
        shared_param_overrides = st.session_state["applied_shared_param_overrides"]
        model_param_overrides = st.session_state["applied_model_param_overrides"]

    st.markdown(
        f'<div class="status-box"><div>● Sistema Activo</div>'
        f'<div class="small-muted" style="margin-top:0.35rem;">Última actualización: {datetime.now().strftime("%H:%M:%S")}</div></div>',
        unsafe_allow_html=True,
    )

viz = Visualizer()

# =========================
# SUPERVISADO — Cómputos compartidos entre páginas
# =========================

if problem_type == "Supervisado":
    with st.spinner("Calculando resultados del ranking de modelos..."):
        model_results_df = compute_model_results_v2(
            active_path, active_target,
            int(selected_random_state), selected_n_splits,
            selected_train_size, use_cross_validation,
            selected_balance_method, selected_criterion,
            shared_param_overrides, model_param_overrides,
        )
    best_model_name = model_results_df.iloc[0]["Modelo"]

    with st.spinner(f"Evaluando {best_model_name}..."):
        best_model_payload = compute_best_model_v2(
            active_path, active_target,
            best_model_name, int(selected_random_state),
            selected_train_size, selected_balance_method,
            shared_param_overrides, model_param_overrides,
        )

    real_metrics = best_model_payload["metrics"]
    real_confusion = best_model_payload["confusion"]
    real_roc_df = best_model_payload["roc_df"]
    real_pr_df = best_model_payload["pr_df"]
    best_model_row = model_results_df.iloc[0].to_dict()
    best_model_params_df = get_best_model_params_df_v2(
        active_path, active_target, best_model_name,
        int(selected_random_state), selected_train_size,
        selected_balance_method, shared_param_overrides, model_param_overrides,
    )

# =========================
# PÁGINA EDA — Supervisado
# =========================

if page == "EDA - Análisis Exploratorio" and problem_type == "Supervisado":
    try:
        df_eda, eda_summary = load_eda_data_v2(active_path, active_target)
    except Exception as e:
        st.error(f"Error cargando datos para EDA: {e}")
        st.stop()

    hero(
        "Análisis Exploratorio de Datos (EDA)",
        f"Insights del dataset · Variable objetivo: {active_target}",
    )

    c1, c2, c3, c4 = st.columns(4, gap="medium")
    with c1:
        metric_card("Total de registros", f"{eda_summary['total_rows']:,}", "Filas en el dataset", "#4f46e5")
    with c2:
        n_classes = eda_summary["n_classes"]
        metric_card("Clases detectadas", str(n_classes), f"Valores únicos en '{active_target}'", "#2563eb")
    with c3:
        metric_card("Features disponibles", str(eda_summary["total_cols"] - 1), "Variables predictoras", "#06b6d4")
    with c4:
        metric_card("Duplicados", f"{eda_summary['dup_count']:,}", "Registros duplicados detectados", "#7c3aed")

    left, right = st.columns([1.1, 0.9], gap="large")
    with left:
        panel_open("Features más correlacionadas (Top 10)", f"Correlación absoluta con '{active_target}'.")
        fig = viz.top_target_correlation_bar(df_eda, target_col=active_target)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        panel_close()
    with right:
        panel_open("Distribución de la variable objetivo", f"Balance de clases en '{active_target}'.")
        fig_pie = viz.target_distribution_donut(df_eda, target_col=active_target)
        st.plotly_chart(fig_pie, width="stretch", config={"displayModeBar": False})
        panel_close()

    panel_open("Resumen de calidad de datos", "Estado del dataset cargado.")
    q1, q2, q3 = st.columns(3, gap="medium")
    with q1:
        metric_panel_card("Valores nulos", "Conteo total de valores faltantes.", str(eda_summary["null_count"]), "#10b981")
    with q2:
        metric_panel_card("Duplicados", "Filas repetidas detectadas.", f"{eda_summary['dup_count']:,}", "#f59e0b")
    with q3:
        metric_panel_card("Variable objetivo", "Columna usada como clase a predecir.", active_target, "#2563eb")
    panel_close()

    panel_open("Distribución por clase", "Conteo de registros en cada clase.")
    class_df = pd.DataFrame(
        [{"Clase": str(k), "Conteo": v, "Porcentaje (%)": round(v / eda_summary["total_rows"] * 100, 2)}
         for k, v in eda_summary["class_counts"].items()]
    )
    st.dataframe(class_df, width="stretch", hide_index=True)
    panel_close()

# =========================
# PÁGINA EDA — No Supervisado
# =========================

elif page == "EDA - Análisis Exploratorio" and problem_type == "No Supervisado":
    try:
        df_eda_unsup, eda_summary_unsup = load_eda_data_v2(active_path, active_target if active_target in pd.read_csv(active_path, nrows=0).columns else list(pd.read_csv(active_path, nrows=0).columns)[0])
    except Exception:
        try:
            _tmp = pd.read_csv(active_path, nrows=0)
            df_eda_unsup = pd.read_csv(active_path)
            eda_summary_unsup = {
                "total_rows": len(df_eda_unsup), "total_cols": df_eda_unsup.shape[1],
                "null_count": int(df_eda_unsup.isnull().sum().sum()),
                "dup_count": int(df_eda_unsup.duplicated().sum()),
                "class_counts": {}, "n_classes": 0,
            }
        except Exception as e:
            st.error(f"Error cargando datos: {e}")
            st.stop()

    hero(
        "Análisis Exploratorio de Datos (EDA)",
        "Exploración de la estructura del dataset para análisis no supervisado.",
    )

    c1, c2, c3, c4 = st.columns(4, gap="medium")
    with c1:
        metric_card("Total de registros", f"{eda_summary_unsup['total_rows']:,}", "Filas en el dataset", "#4f46e5")
    with c2:
        metric_card("Variables", str(eda_summary_unsup["total_cols"]), "Columnas disponibles", "#2563eb")
    with c3:
        metric_card("Valores nulos", str(eda_summary_unsup["null_count"]), "Total de valores faltantes", "#10b981")
    with c4:
        metric_card("Duplicados", f"{eda_summary_unsup['dup_count']:,}", "Filas repetidas detectadas", "#7c3aed")

    panel_open("Correlación entre variables", "Mapa de calor de correlaciones.")
    numeric_cols = df_eda_unsup.select_dtypes(include="number").columns.tolist()
    if len(numeric_cols) >= 2:
        fig_corr = viz.correlation_heatmap(df_eda_unsup[numeric_cols])
        st.plotly_chart(fig_corr, width="stretch", config={"displayModeBar": False})
    else:
        st.info("Se necesitan al menos 2 columnas numéricas para mostrar el mapa de correlación.")
    panel_close()

    panel_open("Vista previa del dataset", "Primeras filas del archivo cargado.")
    st.dataframe(df_eda_unsup.head(20), width="stretch")
    panel_close()

# =========================
# PÁGINA Rendimiento — Supervisado
# =========================

elif page == "Rendimiento del Algoritmo":
    hero(
        "Rendimiento del Algoritmo",
        f"Métricas reales del mejor modelo seleccionado: {best_model_name}.",
    )

    cols = st.columns(5, gap="medium")
    metrics_list = [
        ("Exactitud (Accuracy)", f"{real_metrics['accuracy'] * 100:.2f}%", "Desempeño global", "#4f46e5"),
        ("Precisión (Precision)", f"{real_metrics['precision'] * 100:.2f}%", "Control de falsos positivos", "#10b981"),
        ("Sensibilidad (Recall)", f"{real_metrics['recall'] * 100:.2f}%", "Cobertura de la clase positiva", "#2563eb"),
        ("F1-Score", f"{real_metrics['f1'] * 100:.2f}%", "Balance precisión-recall", "#7c3aed"),
        ("Especificidad", f"{real_metrics['specificity'] * 100:.2f}%", "Detección de negativos", "#f59e0b"),
    ]
    for col, m in zip(cols, metrics_list):
        with col:
            metric_card(*m)

    roc_col, pr_col = st.columns(2, gap="large")
    with roc_col:
        panel_open(
            "Curva ROC",
            f"AUC: {real_metrics['roc_auc']:.3f}" if real_metrics["roc_auc"] is not None else "AUC no disponible",
        )
        if real_roc_df is not None:
            roc_fig = viz.roc_curve_plot(real_roc_df, auc_value=real_metrics["roc_auc"])
            st.plotly_chart(roc_fig, width="stretch", config={"displayModeBar": False})
        else:
            st.info("No fue posible calcular la curva ROC para este modelo.")
        panel_close()
    with pr_col:
        panel_open("Curva Precision-Recall", "Calculada sobre el conjunto de prueba.")
        if real_pr_df is not None:
            pr_fig = viz.precision_recall_plot(real_pr_df)
            st.plotly_chart(pr_fig, width="stretch", config={"displayModeBar": False})
        else:
            st.info("No fue posible calcular la curva Precision-Recall.")
        panel_close()

    panel_open("Matriz de confusión", f"Basada en {best_model_payload['test_size']:,} registros del conjunto de prueba.")
    left_spacer, center_block, right_spacer = st.columns([0.12, 0.76, 0.12])
    with center_block:
        st.markdown(
            '<div style="display:grid;grid-template-columns:180px 1fr 1fr;gap:18px;align-items:center;margin-top:0.5rem;">'
            '<div></div>'
            '<div style="text-align:center;font-weight:800;color:#334155;font-size:1rem;">Predicción Positiva</div>'
            '<div style="text-align:center;font-weight:800;color:#334155;font-size:1rem;">Predicción Negativa</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        row_labels, matrix_cols = st.columns([0.28, 0.72], gap="medium")
        with row_labels:
            st.markdown("<div style='height:168px;display:flex;align-items:center;justify-content:flex-end;font-weight:800;color:#334155;font-size:1rem;'>Real Positivo</div>", unsafe_allow_html=True)
            st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
            st.markdown("<div style='height:168px;display:flex;align-items:center;justify-content:flex-end;font-weight:800;color:#334155;font-size:1rem;'>Real Negativo</div>", unsafe_allow_html=True)
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
        metric_panel_card("Falsos Negativos", "Positivos no detectados.", str(real_confusion["FN"]), "#ef4444")
    with err2:
        metric_panel_card("Falsos Positivos", "Negativos marcados como positivos.", str(real_confusion["FP"]), "#f59e0b")
    with err3:
        metric_panel_card("Tasa de Error Total", "Clasificaciones incorrectas totales.", f"{real_metrics['error_rate'] * 100:.3f}%", "#2563eb")

    panel_open("Comparación de modelos", f"Ranking según {get_selection_criterion_label(selected_criterion, use_cross_validation)}.")
    st.dataframe(model_results_df.round(4), width="stretch", hide_index=True)
    panel_close()

# =========================
# PÁGINA Parametrización — Supervisado
# =========================

elif page == "Parametrización del Modelo":
    hero("Parametrización del Modelo", f"Configuración real del mejor modelo: {best_model_name}.")

    c1, c2, c3, c4, c5 = st.columns(5, gap="medium")
    with c1:
        winner_caption = "Seleccionado por CV" if use_cross_validation else "Seleccionado por métrica global"
        metric_card("Modelo ganador", best_model_name, winner_caption, "#2563eb")
    with c2:
        metric_card("Semilla", str(int(selected_random_state)), "Reproducibilidad", "#7c3aed")
    with c3:
        cv_value = f"{selected_n_splits}-fold" if use_cross_validation else "Desactivado"
        cv_desc = "Evaluación cruzada" if use_cross_validation else "Se usan métricas globales"
        metric_card("Cross-Validation", cv_value, cv_desc, "#06b6d4")
    with c4:
        metric_card("Train Size", f"{selected_train_size:.2f}", "Proporción de entrenamiento", "#f59e0b")
    with c5:
        balance_display = get_balance_display(selected_balance_method)
        metric_card(balance_display["name"], balance_display["value"], balance_display["description"], "#10b981")

    panel_open("Resumen de selección del modelo", "Información usada para elegir el modelo ganador.")
    s1, s2, s3 = st.columns(3, gap="medium")
    with s1:
        metric_panel_card("Criterio principal", "Métrica usada para ordenar el ranking.", get_selection_criterion_label(selected_criterion, use_cross_validation), "#2563eb")
    with s2:
        m2_label = "ROC_AUC_CV_mean" if use_cross_validation else "ROC_AUC_Global"
        m2_desc = "Promedio del mejor modelo en CV." if use_cross_validation else "Valor global del mejor modelo."
        m2_value = best_model_row["ROC_AUC_CV_mean"] if use_cross_validation else best_model_row["ROC_AUC_Global"]
        metric_panel_card(m2_label, m2_desc, f"{m2_value:.4f}", "#7c3aed")
    with s3:
        m3_label = "Accuracy_CV_mean" if use_cross_validation else "Accuracy_Global"
        m3_desc = "Exactitud promedio en CV." if use_cross_validation else "Exactitud global."
        m3_value = best_model_row["Accuracy_CV_mean"] if use_cross_validation else best_model_row["Accuracy_Global"]
        metric_panel_card(m3_label, m3_desc, f"{m3_value:.4f}", "#06b6d4")
    panel_close()

    panel_open("Hiperparámetros reales del modelo ganador", "Parámetros obtenidos directamente desde get_params().")
    st.dataframe(best_model_params_df, width="stretch", hide_index=True)
    panel_close()

    panel_open("Búsqueda de hiperparámetros", "Optimiza el mejor modelo actual bajo demanda.")
    genetic_available = is_genetic_search_available()
    tuning_method_options = ["Exhaustive"] + (["Genetic"] if genetic_available else [])

    try:
        _df_for_params = load_model_df_v2(active_path, active_target)
        estandarizar_param = best_model_name in ["Regresión Logística", "SVM"]
        balance_config_param = get_runner_balance_config(selected_balance_method)
        _model_for_params = crear_modelo_configurable(best_model_name, random_state=int(selected_random_state),
                                                       shared_overrides=shared_param_overrides,
                                                       model_overrides=model_param_overrides.get(best_model_name, {}))
        _prep_param = DataPreparer(train_size=selected_train_size, random_state=int(selected_random_state), scale_X=estandarizar_param)
        _runner_param = SupervisedRunner(df=_df_for_params, target=active_target, model=_model_for_params,
                                          task="classification", preparer=_prep_param, pos_label=1,
                                          class_weight=balance_config_param["class_weight"],
                                          sampling_method=balance_config_param["sampling_method"])
        configured_best_model = _runner_param.get_model_for_current_split()

        with st.form("v2_best_model_tuning_form"):
            tune_col_1, tune_col_2, tune_col_3 = st.columns(3, gap="medium")
            with tune_col_1:
                tuning_method = st.selectbox("Método de búsqueda", options=tuning_method_options, index=0)
            with tune_col_2:
                tuning_cv = st.slider("Folds para búsqueda", min_value=3, max_value=10, value=5)
            with tune_col_3:
                tuning_scoring = st.selectbox("Scoring", options=["f1", "roc_auc", "accuracy"], index=0)
            if not genetic_available:
                st.caption("La búsqueda genética se habilita instalando `sklearn-genetic-opt`.")
            st.caption("El tuning evalúa el mejor modelo con cross-validation sobre entrenamiento y luego en test.")
            search_grid, search_grid_errors = build_best_model_search_grid(best_model_name, configured_best_model.get_params())
            if search_grid_errors:
                for error in search_grid_errors:
                    st.error(error)
            st.caption(f"Parámetros incluidos: {', '.join(search_grid.keys())}")
            run_tuning = st.form_submit_button("Ejecutar búsqueda sobre el mejor modelo", use_container_width=True)

        if run_tuning:
            if search_grid_errors:
                st.error("Corrige los parámetros inválidos antes de ejecutar.")
            else:
                with st.spinner("Buscando mejores hiperparámetros..."):
                    tuning_result = run_best_model_search_v2(
                        active_path, active_target, best_model_name,
                        int(selected_random_state), selected_train_size,
                        selected_balance_method, shared_param_overrides, model_param_overrides,
                        tuning_method, tuning_cv, tuning_scoring, search_grid,
                    )
                tuned_metrics = tuning_result["metrics"]
                tuned_params_df = pd.DataFrame([{"Parámetro": k, "Valor": str(v)} for k, v in tuning_result["best_params"].items()])
                tuning_cols = st.columns(5, gap="medium")
                for col, card in zip(tuning_cols, [
                    ("Best CV Score", f"{tuning_result['best_score']:.4f}", f"Scoring: {tuning_scoring}", "#2563eb"),
                    ("Accuracy", f"{tuned_metrics['Accuracy']:.4f}", "Conjunto de prueba", "#7c3aed"),
                    ("Precision", f"{tuned_metrics['Precision']:.4f}", "Conjunto de prueba", "#10b981"),
                    ("Recall", f"{tuned_metrics['Recall']:.4f}", "Conjunto de prueba", "#06b6d4"),
                    ("F1", f"{tuned_metrics['F1']:.4f}", "Conjunto de prueba", "#f59e0b"),
                ]):
                    with col:
                        metric_card(*card)
                st.markdown("#### Mejores parámetros encontrados")
                st.dataframe(tuned_params_df, width="stretch", hide_index=True)
    except Exception as e:
        st.error(f"Error preparando la búsqueda de hiperparámetros: {e}")
    panel_close()

# =========================
# PÁGINA Estabilidad — Supervisado
# =========================

elif page == "Estabilidad del Modelo":
    hero("Estabilidad del Modelo", f"Variación de {best_model_name} a través de distintas semillas.")

    seed_text = st.text_input(
        "Semillas para estabilidad",
        key="v2_stability_seed_text",
        help="Separadas por comas.",
    )
    selected_seed_list, seed_validation_error = parse_seed_list_with_validation(seed_text)
    if seed_validation_error:
        st.error(seed_validation_error)
        st.caption(f"Usando lista por defecto: {', '.join(str(s) for s in SEMILLAS)}")

    stability_df = compute_stability_v2(
        active_path, active_target, best_model_name,
        tuple(selected_seed_list), selected_train_size,
        use_cross_validation, selected_n_splits,
        selected_balance_method, shared_param_overrides, model_param_overrides,
    )

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

    sc1, sc2, sc3 = st.columns(3, gap="medium")
    with sc1:
        metric_card("Modelo evaluado", best_model_name, "Mejor modelo actual", "#2563eb")
    with sc2:
        metric_card("Semillas", str(len(selected_seed_list)), "Cantidad de ejecuciones", "#7c3aed")
    with sc3:
        spread_value = stability_primary_series.std() if not stability_primary_series.empty else 0.0
        metric_card("Desv. estándar", f"{spread_value:.4f}", f"Variación de {stability_primary_label}", "#06b6d4")

    panel_open("Resumen de estabilidad", "Promedio y variación de métricas por semilla.")
    sum1, sum2, sum3 = st.columns(3, gap="medium")
    with sum1:
        metric_panel_card(f"Promedio {stability_primary_label}", "Media entre semillas evaluadas.",
                          f"{stability_primary_series.mean():.4f}" if not stability_primary_series.empty else "N/A", "#2563eb")
    with sum2:
        metric_panel_card(f"Promedio {stability_secondary_label}", "Comportamiento medio entre semillas.",
                          f"{stability_secondary_series.mean():.4f}" if not stability_secondary_series.empty else "N/A", "#7c3aed")
    with sum3:
        metric_panel_card("Semillas usadas", "Lista de semillas del análisis.",
                          ", ".join(str(s) for s in selected_seed_list), "#10b981")
    panel_close()

    panel_open("Resultados por semilla", "Cada fila: una ejecución completa con una semilla distinta.")
    st.dataframe(stability_df.round(4), width="stretch", hide_index=True)
    panel_close()

# =========================
# PÁGINA Clustering — No Supervisado
# =========================

elif page == "Clustering":
    hero("Clustering", "Agrupamiento no supervisado de los datos.")

    tab_analisis, tab_param, tab_estab = st.tabs(["Análisis", "Parametrización", "Estabilidad"])

    with tab_analisis:
        c1, c2, c3 = st.columns(3, gap="medium")
        with c1:
            metric_card("Algoritmo", "KMeans", "Seleccionado en Parametrización", "#2563eb")
        with c2:
            metric_card("Clusters (k)", "—", "Configurado en Parametrización", "#7c3aed")
        with c3:
            metric_card("Silhouette Score", "—", "Calculado al ejecutar", "#10b981")

        panel_open("Visualización de clusters", "Proyección 2D de los grupos encontrados.")
        unsup_placeholder("🔵", "Ejecuta el análisis en la pestaña Parametrización para ver los clusters aquí.")
        panel_close()

        panel_open("Estadísticas por cluster", "Tamaño y centroide de cada grupo.")
        unsup_placeholder("📊", "Los resultados del clustering aparecerán aquí tras ejecutar el análisis.")
        panel_close()

    with tab_param:
        panel_open("Configuración del algoritmo de clustering", "Ajusta los parámetros antes de ejecutar.")

        p1, p2 = st.columns(2, gap="medium")
        with p1:
            clustering_algo = st.selectbox(
                "Algoritmo",
                options=["KMeans", "HAC (Agrupamiento Jerárquico Aglomerativo)"],
                key="v2_clustering_algo",
            )
        with p2:
            n_clusters = st.slider("Número de clusters (k)", min_value=2, max_value=20, value=3, key="v2_n_clusters")

        if "KMeans" in clustering_algo:
            km1, km2 = st.columns(2, gap="medium")
            with km1:
                km_init = st.selectbox("Inicialización", options=["k-means++", "random"], key="v2_km_init")
            with km2:
                km_max_iter = st.number_input("max_iter", min_value=50, max_value=1000, value=300, key="v2_km_max_iter")
            km_n_init = st.slider("n_init (repeticiones)", min_value=1, max_value=20, value=10, key="v2_km_n_init")
        else:
            hac1, hac2 = st.columns(2, gap="medium")
            with hac1:
                hac_linkage = st.selectbox("Linkage", options=["ward", "complete", "average", "single"], key="v2_hac_linkage")
            with hac2:
                hac_affinity = st.selectbox("Métrica de distancia", options=["euclidean", "manhattan", "cosine"], key="v2_hac_affinity")

        with st.expander("Preprocesamiento de variables", expanded=False):
            num_cols_available = []
            try:
                _df_preview = pd.read_csv(active_path, nrows=5)
                num_cols_available = _df_preview.select_dtypes(include="number").columns.tolist()
            except Exception:
                pass
            if num_cols_available:
                selected_features = st.multiselect(
                    "Variables a incluir (vacío = todas las numéricas)",
                    options=num_cols_available,
                    key="v2_clustering_features",
                )
            scale_data = st.toggle("Estandarizar datos (StandardScaler)", value=True, key="v2_clustering_scale")

        panel_close()

        st.button("Ejecutar clustering", use_container_width=True, key="v2_run_clustering", disabled=True)
        st.caption("La conexión con UnsupervisedRunner se habilitará en la siguiente etapa del proyecto.")

    with tab_estab:
        panel_open("Estabilidad del clustering", "Variación del Silhouette Score a través de distintas semillas.")

        es1, es2 = st.columns(2, gap="medium")
        with es1:
            stability_seeds_text = st.text_input(
                "Semillas",
                value=", ".join(str(s) for s in SEMILLAS),
                key="v2_clustering_stability_seeds",
                help="Enteros separados por comas.",
            )
        with es2:
            stability_k_range = st.text_input(
                "Rango de k a evaluar",
                value="2, 3, 4, 5",
                key="v2_clustering_k_range",
                help="Valores de k separados por comas.",
            )

        st.button("Ejecutar análisis de estabilidad", use_container_width=True, key="v2_run_clustering_stability", disabled=True)
        panel_close()

        panel_open("Resultados de estabilidad", "Silhouette promedio por k y semilla.")
        unsup_placeholder("📈", "Los resultados de estabilidad aparecerán aquí tras ejecutar el análisis.")
        panel_close()

# =========================
# PÁGINA Reducción Dimensional — No Supervisado
# =========================

elif page == "Reducción Dimensional":
    hero("Reducción Dimensional", "Visualización de la estructura interna del dataset.")

    tab_analisis, tab_param = st.tabs(["Análisis", "Parametrización"])

    with tab_analisis:
        rd1, rd2, rd3 = st.columns(3, gap="medium")
        with rd1:
            metric_card("Método", "PCA", "Seleccionado en Parametrización", "#2563eb")
        with rd2:
            metric_card("Componentes", "—", "Configurado en Parametrización", "#7c3aed")
        with rd3:
            metric_card("Varianza explicada", "—", "Calculada al ejecutar (PCA)", "#10b981")

        panel_open("Proyección 2D", "Distribución de los puntos en el espacio reducido.")
        unsup_placeholder("🌐", "Ejecuta la reducción dimensional en Parametrización para ver la proyección aquí.")
        panel_close()

        panel_open("Varianza explicada acumulada", "Solo disponible para PCA.")
        unsup_placeholder("📉", "El gráfico de varianza acumulada aparecerá aquí tras ejecutar PCA.")
        panel_close()

    with tab_param:
        panel_open("Configuración del método de reducción", "Ajusta los parámetros antes de ejecutar.")

        rd_method = st.selectbox(
            "Método",
            options=["PCA", "t-SNE", "UMAP"],
            key="v2_rd_method",
        )
        rd_n_components = st.slider("Número de componentes", min_value=2, max_value=10, value=2, key="v2_rd_components")

        if rd_method == "PCA":
            st.caption("PCA es determinístico. El número de componentes controla cuántas dimensiones se retienen.")

        elif rd_method == "t-SNE":
            ts1, ts2 = st.columns(2, gap="medium")
            with ts1:
                tsne_perplexity = st.slider("Perplexity", min_value=5, max_value=100, value=30, key="v2_tsne_perplexity")
            with ts2:
                tsne_lr = st.number_input("Learning rate", min_value=10.0, max_value=1000.0, value=200.0, key="v2_tsne_lr")
            tsne_n_iter = st.number_input("n_iter", min_value=250, max_value=5000, value=1000, key="v2_tsne_n_iter", step=250)

        elif rd_method == "UMAP":
            um1, um2 = st.columns(2, gap="medium")
            with um1:
                umap_neighbors = st.slider("n_neighbors", min_value=2, max_value=100, value=15, key="v2_umap_neighbors")
            with um2:
                umap_min_dist = st.slider("min_dist", min_value=0.0, max_value=1.0, value=0.1, step=0.05, key="v2_umap_min_dist")
            umap_metric = st.selectbox("Métrica", options=["euclidean", "manhattan", "cosine", "correlation"], key="v2_umap_metric")

        with st.expander("Preprocesamiento de variables", expanded=False):
            try:
                _df_preview_rd = pd.read_csv(active_path, nrows=5)
                num_cols_rd = _df_preview_rd.select_dtypes(include="number").columns.tolist()
            except Exception:
                num_cols_rd = []
            if num_cols_rd:
                selected_features_rd = st.multiselect(
                    "Variables a incluir (vacío = todas las numéricas)",
                    options=num_cols_rd,
                    key="v2_rd_features",
                )
            scale_rd = st.toggle("Estandarizar datos (StandardScaler)", value=True, key="v2_rd_scale")

        panel_close()

        st.button("Ejecutar reducción dimensional", use_container_width=True, key="v2_run_rd", disabled=True)
        st.caption("La conexión con UnsupervisedRunner se habilitará en la siguiente etapa del proyecto.")

# =========================
# PÁGINA Reglas de Asociación — No Supervisado
# =========================

elif page == "Reglas de Asociación":
    hero("Reglas de Asociación", "Descubrimiento de patrones frecuentes en los datos.")

    tab_analisis, tab_param = st.tabs(["Análisis", "Parametrización"])

    with tab_analisis:
        ra1, ra2, ra3 = st.columns(3, gap="medium")
        with ra1:
            metric_card("Itemsets frecuentes", "—", "Calculados al ejecutar", "#2563eb")
        with ra2:
            metric_card("Reglas generadas", "—", "Calculadas al ejecutar", "#7c3aed")
        with ra3:
            metric_card("Confianza máx.", "—", "Mejor regla encontrada", "#10b981")

        panel_open("Reglas de asociación encontradas", "Top reglas ordenadas por lift.")
        unsup_placeholder("🔗", "Las reglas de asociación aparecerán aquí tras ejecutar el análisis en Parametrización.")
        panel_close()

        panel_open("Dispersión soporte vs confianza", "Visualización del trade-off entre soporte y confianza.")
        unsup_placeholder("📊", "El gráfico de dispersión aparecerá aquí tras ejecutar el análisis.")
        panel_close()

    with tab_param:
        panel_open("Configuración del análisis de reglas", "Ajusta los parámetros antes de ejecutar.")

        ra_algo = st.selectbox("Algoritmo", options=["Apriori", "FP-Growth"], key="v2_ra_algo")

        rp1, rp2, rp3 = st.columns(3, gap="medium")
        with rp1:
            min_support = st.slider("Soporte mínimo", min_value=0.01, max_value=1.0, value=0.05, step=0.01, key="v2_ra_min_support")
        with rp2:
            min_confidence = st.slider("Confianza mínima", min_value=0.1, max_value=1.0, value=0.5, step=0.05, key="v2_ra_min_confidence")
        with rp3:
            min_lift = st.number_input("Lift mínimo", min_value=1.0, max_value=20.0, value=1.0, step=0.5, key="v2_ra_min_lift")

        top_n_rules = st.slider("Top N reglas a mostrar", min_value=5, max_value=100, value=20, key="v2_ra_top_n")

        st.markdown("#### Formato de los datos")
        data_format = st.radio(
            "Formato de transacciones",
            options=[
                "Tabla larga (columna de transacción + columna de ítem)",
                "Matriz binaria (columnas 0/1 por ítem)",
                "Columna con lista de ítems separados por delimitador",
            ],
            key="v2_ra_format",
            label_visibility="collapsed",
        )

        try:
            _df_preview_ra = pd.read_csv(active_path, nrows=5)
            _all_cols_ra = list(_df_preview_ra.columns)
        except Exception:
            _all_cols_ra = []

        if "larga" in data_format and _all_cols_ra:
            ra_c1, ra_c2 = st.columns(2, gap="medium")
            with ra_c1:
                transaction_col = st.selectbox("Columna de transacción (ID)", options=_all_cols_ra, key="v2_ra_tx_col")
            with ra_c2:
                item_col = st.selectbox("Columna de ítem", options=_all_cols_ra, key="v2_ra_item_col")

        elif "binaria" in data_format and _all_cols_ra:
            binary_cols = st.multiselect("Columnas binarias (0/1)", options=_all_cols_ra, key="v2_ra_binary_cols")

        elif "lista" in data_format and _all_cols_ra:
            list_col = st.selectbox("Columna con la lista de ítems", options=_all_cols_ra, key="v2_ra_list_col")
            list_sep = st.text_input("Separador de ítems en la lista", value=",", key="v2_ra_list_sep")

        panel_close()

        st.button("Ejecutar análisis de reglas", use_container_width=True, key="v2_run_ra", disabled=True)
        st.caption("La conexión con AssociationRulesExplorer se habilitará en la siguiente etapa del proyecto.")
