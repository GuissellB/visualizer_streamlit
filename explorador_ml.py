import sys
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve,
    mean_absolute_error, mean_squared_error, r2_score,
)

from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

from ml_toolkit import (
    DataPreparer, SupervisedRunner, NeuralNetworkRunner,
    UnsupervisedRunner, AssociationRulesExplorer,
    get_positive_score,
    ARIMAForecaster, EDAExplorer, HoltWintersForecaster, TimeSeriesRunner,
)
from visualizer import Visualizer

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Explorador ML",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
:root {
    --card: rgba(255,255,255,0.85); --text: #162033;
    --muted: #64748b; --blue: #2563eb; --indigo: #4f46e5;
    --purple: #7c3aed; --cyan: #06b6d4; --green: #10b981;
    --orange: #f59e0b; --red: #ef4444;
}
.stApp {
    background:
        radial-gradient(circle at top left, rgba(59,130,246,0.18), transparent 28%),
        radial-gradient(circle at top right, rgba(168,85,247,0.18), transparent 26%),
        linear-gradient(180deg, #eff6ff 0%, #f8fafc 100%);
}
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#ffffff 0%,#f8fbff 100%);
    border-right: 1px solid rgba(37,99,235,0.08);
}
.hero {
    background: linear-gradient(135deg,#2563eb 0%,#4f46e5 45%,#7c3aed 100%);
    color: white; padding: 1.4rem 1.6rem; border-radius: 24px;
    box-shadow: 0 18px 45px rgba(79,70,229,0.22); margin-bottom: 1rem;
}
.hero h1 { margin:0; font-size:2rem; font-weight:800; letter-spacing:-0.03em; }
.hero p  { margin:0.35rem 0 0 0; opacity:0.92; font-size:0.98rem; }
.metric-card {
    background: var(--card); backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.55); border-radius: 22px;
    padding: 1rem; box-shadow: 0 14px 34px rgba(15,23,42,0.08);
    min-height: 130px; margin-bottom: 1rem;
}
.metric-label { color:var(--muted); font-size:0.9rem; margin-bottom:0.55rem; font-weight:600; }
.metric-value { font-size:1.95rem; font-weight:800; line-height:1.05; color:var(--text); }
.metric-sub   { color:var(--muted); font-size:0.83rem; margin-top:0.45rem; }
.panel {
    background: rgba(255,255,255,0.86); border: 1px solid rgba(255,255,255,0.6);
    backdrop-filter: blur(12px); border-radius: 24px; padding: 1.15rem 1.2rem;
    box-shadow: 0 14px 34px rgba(15,23,42,0.08); margin-top:0.35rem; margin-bottom:1rem;
}
.panel h3 { margin-top:0; margin-bottom:0.25rem; color:var(--text); font-size:1.15rem; font-weight:800; }
.panel .sub { color:var(--muted); font-size:0.88rem; margin-bottom:0.8rem; }
.cm-box {
    border-radius:24px; padding:1.4rem 1rem; text-align:center; font-weight:700;
    min-height:140px; display:flex; flex-direction:column; justify-content:center;
}
.small-muted { color:var(--muted); font-size:0.8rem; }
.block-container { padding-top:1.4rem; padding-bottom:2rem; }
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
_DEFAULTS: dict = {
    "df": None, "df_name": "",
    "_prep_cfg_exclude": [], "_prep_cfg_impute": "No imputar", "_prep_cfg_ohe": [],
    "sup_results": None, "sup_best_payload": None,
    "sup_target": None, "sup_task": "classification",
    "sup_features": [], "sup_balance": "none",
    "sup_train_size": 0.75, "sup_rs": 42,
    "sup_use_cv": True, "sup_n_splits": 5, "sup_pos_label": "1",
    "nn_result": None, "tuning_result": None, "stability_df": None,
    "unsup_cluster_result": None, "unsup_dim_result": None,
    "assoc_rules": None, "assoc_itemsets": None,
    "ts_analysis": None,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ── Preprocessing aplicado globalmente en cada render ──────────────────────────
def _apply_preprocessing() -> None:
    """Reconstruye session_state['df'] desde df_raw + estados de widgets de preparación."""
    if st.session_state.get("df_raw") is None:
        return
    _raw = st.session_state["df_raw"]

    # 1) Excluir columnas
    _exclude = [c for c in st.session_state.get("_prep_cfg_exclude", []) if c in _raw.columns]
    _df = _raw.drop(columns=_exclude) if _exclude else _raw.copy()

    # 1.5) Eliminar duplicados siempre
    _df = _df.drop_duplicates().reset_index(drop=True)

    # 2) Imputación
    _strategy = st.session_state.get("_prep_cfg_impute", "No imputar")
    if _strategy == "Eliminar filas con nulos":
        _df = _df.dropna()
    elif _strategy == "Media (numéricas)":
        _num = _df.select_dtypes(include=np.number).columns
        _df[_num] = _df[_num].fillna(_df[_num].mean())
    elif _strategy == "Mediana (numéricas)":
        _num = _df.select_dtypes(include=np.number).columns
        _df[_num] = _df[_num].fillna(_df[_num].median())
    elif _strategy == "Moda (todas)":
        _df = _df.fillna(_df.mode().iloc[0])

    # 3) One-hot encoding
    _ohe = [c for c in st.session_state.get("_prep_cfg_ohe", []) if c in _df.columns]
    if _ohe:
        _df = pd.get_dummies(_df, columns=_ohe, drop_first=False, dtype=int)

    st.session_state["df"] = _df

_apply_preprocessing()

def _build_supervised_df(target: str, features: list[str] | None = None) -> tuple[pd.DataFrame, dict]:
    """Construye un dataset de modelado evitando imputación global previa al split.

    Reglas:
    - parte del archivo original (df_raw)
    - excluye columnas marcadas por el usuario
    - elimina duplicados exactos
    - elimina filas con target nulo
    - codifica variables categóricas de entrada a numérico
    - elimina filas con nulos restantes en las columnas realmente usadas
    """
    raw = st.session_state.get("df_raw")
    if raw is None:
        return pd.DataFrame(), {"rows_initial": 0, "rows_final": 0, "dropped_null_rows": 0}

    df_model = raw.copy()
    exclude = [c for c in st.session_state.get("_prep_cfg_exclude", []) if c in df_model.columns]
    if exclude:
        df_model = df_model.drop(columns=exclude)

    rows_initial = len(df_model)
    original_duplicates = int(df_model.duplicated().sum())
    if original_duplicates:
        df_model = df_model.drop_duplicates().reset_index(drop=True)

    if target not in df_model.columns:
        return pd.DataFrame(), {
            "rows_initial": rows_initial,
            "rows_final": 0,
            "dropped_duplicates": original_duplicates,
            "dropped_null_rows": 0,
            "auto_ohe_columns": [],
        }

    # Target siempre sin nulos
    before_target = len(df_model)
    df_model = df_model.dropna(subset=[target]).copy()
    dropped_target_nulls = before_target - len(df_model)

    selected_features = [c for c in (features or []) if c in df_model.columns and c != target]
    feature_cols = selected_features or [c for c in df_model.columns if c != target]

    # En supervisado, toda feature categórica se codifica para evitar fallos del estimador.
    cat_features = [c for c in feature_cols if str(df_model[c].dtype) in ("object", "category") or pd.api.types.is_object_dtype(df_model[c]) or pd.api.types.is_categorical_dtype(df_model[c])]
    user_ohe = [c for c in st.session_state.get("_prep_cfg_ohe", []) if c in feature_cols]
    auto_ohe_columns = sorted(set(cat_features) | set(user_ohe))
    if auto_ohe_columns:
        df_model = pd.get_dummies(df_model, columns=auto_ohe_columns, drop_first=False, dtype=int)

    feature_cols_after = [c for c in df_model.columns if c != target]

    # Para evitar leakage, no se imputa globalmente antes del split.
    # Se eliminan filas con nulos en las columnas realmente usadas.
    used_cols = [target] + feature_cols_after
    before_dropna = len(df_model)
    df_model = df_model.dropna(subset=used_cols).reset_index(drop=True)
    dropped_null_rows = before_dropna - len(df_model)

    meta = {
        "rows_initial": rows_initial,
        "rows_final": len(df_model),
        "dropped_duplicates": original_duplicates,
        "dropped_target_nulls": dropped_target_nulls,
        "dropped_null_rows": dropped_null_rows,
        "auto_ohe_columns": auto_ohe_columns,
        "feature_count": len(feature_cols_after),
    }
    return df_model, meta

# ── UI helpers ─────────────────────────────────────────────────────────────────
viz = Visualizer()

def hero(title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )

def metric_card(label: str, value: str, sub: str, color: str = "#162033") -> None:
    st.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value" style="color:{color};">{value}</div>'
        f'<div class="metric-sub">{sub}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

def panel_open(title: str, sub: str = "") -> None:
    st.markdown(
        f'<div class="panel"><h3>{title}</h3><div class="sub">{sub}</div>',
        unsafe_allow_html=True,
    )

def panel_close() -> None:
    st.markdown("</div>", unsafe_allow_html=True)

def confusion_box(value: int, label: str, pct: float, bg: str, border: str, color: str) -> None:
    st.markdown(
        f"<div class='cm-box' style='background:{bg};border:2px solid {border};color:{color};'>"
        f"<div style='font-size:1.8rem'>{value}</div><div>{label}</div>"
        f"<div class='small-muted'>{pct:.1f}%</div></div>",
        unsafe_allow_html=True,
    )

def _require_df() -> bool:
    if st.session_state["df"] is None:
        st.info("Primero carga un archivo CSV en la sección **Datos**.")
        return False
    return True

def _fig_layout(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(248,250,252,0.92)",
        height=height,
        margin=dict(l=10, r=10, t=20, b=10),
    )
    return fig

# ── Balance helper ─────────────────────────────────────────────────────────────
_BALANCE_LABELS = {
    "none": "Sin balanceo", "class_weight": "Class Weight",
    "undersample": "Undersample", "oversample": "Oversample",
    "smote_tomek": "SMOTE+Tomek",
}

def _balance_cfg(method: str) -> dict:
    if method == "class_weight":
        return {"class_weight": "balanced", "sampling_method": None}
    if method in {"undersample", "oversample", "smote_tomek"}:
        return {"class_weight": None, "sampling_method": method}
    return {"class_weight": None, "sampling_method": None}

# ── Model factories ────────────────────────────────────────────────────────────
_SCALE_MODELS = {"Regresión Logística", "SVM", "Regresión Lineal", "Ridge"}

def _clf_model(name: str, rs: int):
    return {
        "Regresión Logística": LogisticRegression(random_state=rs, max_iter=1000),
        "Random Forest": RandomForestClassifier(random_state=rs, n_estimators=100),
        "SVM": SVC(random_state=rs, probability=True),
        "XGBoost": XGBClassifier(random_state=rs, eval_metric="logloss", verbosity=0),
        "LightGBM": LGBMClassifier(random_state=rs, verbose=-1),
    }[name]

def _reg_model(name: str, rs: int):
    return {
        "Regresión Lineal": LinearRegression(),
        "Ridge": Ridge(),
        "Random Forest": RandomForestRegressor(random_state=rs, n_estimators=100),
        "XGBoost": XGBRegressor(random_state=rs, verbosity=0),
        "LightGBM": LGBMRegressor(random_state=rs, verbose=-1),
    }[name]

MODEL_PARAM_WHITELIST = {
    "Regresión Logística": ["C", "solver", "penalty", "max_iter", "class_weight", "random_state"],
    "Random Forest": ["n_estimators", "max_depth", "min_samples_split", "min_samples_leaf", "max_features", "class_weight", "random_state"],
    "SVM": ["C", "kernel", "gamma", "probability", "class_weight", "random_state"],
    "XGBoost": ["n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree", "objective", "eval_metric", "scale_pos_weight", "random_state"],
    "LightGBM": ["n_estimators", "learning_rate", "max_depth", "num_leaves", "subsample", "colsample_bytree", "objective", "class_weight", "random_state"],
    "Regresión Lineal": ["fit_intercept", "copy_X", "tol", "positive"],
    "Ridge": ["alpha", "fit_intercept", "solver", "random_state"],
}

def _build_configured_model(nombre: str, rs: int, train_size: float, balance: str, task: str):
    bcfg = _balance_cfg(balance)
    prep = DataPreparer(
        train_size=train_size,
        random_state=rs,
        scale_X=(nombre in _SCALE_MODELS),
    )
    model = _clf_model(nombre, rs) if task == "classification" else _reg_model(nombre, rs)

    runner = SupervisedRunner(
        df=_build_supervised_df(st.session_state["sup_target"], st.session_state["sup_features"] or [])[0],
        target=st.session_state["sup_target"],
        model=model,
        task=task,
        features=st.session_state["sup_features"] or [],
        preparer=prep,
        pos_label=_coerce_pos_label(st.session_state["sup_pos_label"]),
        class_weight=bcfg["class_weight"] if task == "classification" else None,
        sampling_method=bcfg["sampling_method"] if task == "classification" else None,
    )
    return runner.get_model_for_current_split()

def _get_best_model_params_df(nombre: str, rs: int, train_size: float, balance: str, task: str) -> pd.DataFrame:
    model = _build_configured_model(nombre, rs, train_size, balance, task)
    params = model.get_params()

    selected_keys = MODEL_PARAM_WHITELIST.get(nombre, list(params.keys()))
    rows = []

    if task == "classification" and balance in {"undersample", "oversample", "smote_tomek"}:
        rows.append({"Parámetro": "sampling_method", "Valor": balance})

    for key in selected_keys:
        if key in params:
            rows.append({"Parámetro": key, "Valor": str(params[key])})

    return pd.DataFrame(rows)
def _is_genetic_search_available() -> bool:
    try:
        import sklearn_genetic  # noqa: F401
        return True
    except ImportError:
        return False

def _normalize_param_value(value):
    return None if value == "None" else value

def _parse_search_values(raw_value: str):
    values = [item.strip() for item in str(raw_value).split(",") if item.strip()]
    parsed = []
    for value in values:
        try:
            if "." in value:
                parsed.append(float(value))
            else:
                parsed.append(int(value))
        except Exception:
            parsed.append(_normalize_param_value(value))
    return parsed

def _default_search_text(current_value):
    if current_value is None or current_value == "":
        return ""
    if isinstance(current_value, bool):
        return str(current_value)
    if isinstance(current_value, int):
        if current_value <= 0:
            return str(current_value)
        vals = sorted({max(1, current_value // 2), current_value, current_value * 2})
        return ", ".join(str(v) for v in vals)
    if isinstance(current_value, float):
        vals = sorted({
            round(max(0.0001, current_value * 0.8), 4),
            round(current_value, 4),
            round(current_value * 1.2, 4),
        })
        return ", ".join(str(v) for v in vals)
    return str(current_value)

def _build_best_model_search_grid(best_model_name: str, current_params: dict) -> tuple[dict, list]:
    param_grid = {}
    errors = []

    st.caption("Los campos cargan un mini rango alrededor del valor actual del mejor modelo.")

    with st.expander("Grid de búsqueda", expanded=False):
        selected_keys = MODEL_PARAM_WHITELIST.get(best_model_name, list(current_params.keys()))
        for param_key in selected_keys:
            if param_key not in current_params:
                continue

            default_text = _default_search_text(current_params.get(param_key))
            raw_value = st.text_input(
                f"{param_key} (grid)",
                value=default_text,
                key=f"search_{best_model_name}_{param_key}",
                placeholder=f"Ej: {default_text}",
                help="Valores separados por comas.",
            )

            try:
                values = _parse_search_values(raw_value)
                if values:
                    param_grid[param_key] = values
            except Exception:
                errors.append(f"Valores inválidos para {param_key}.")

    return param_grid, errors

def _run_best_model_search(best_model_name: str, seed: int, train_size: float, balance: str, task: str,
                           search_method: str, search_cv: int, scoring: str, param_grid: dict):
    df_local = _build_supervised_df(st.session_state["sup_target"], st.session_state["sup_features"] or [])[0]
    target = st.session_state["sup_target"]
    features = st.session_state["sup_features"] or []
    pos_label = _coerce_pos_label(st.session_state["sup_pos_label"])
    bcfg = _balance_cfg(balance)

    model = _clf_model(best_model_name, seed) if task == "classification" else _reg_model(best_model_name, seed)
    prep = DataPreparer(train_size=train_size, random_state=seed, scale_X=(best_model_name in _SCALE_MODELS))

    runner = SupervisedRunner(
        df=df_local,
        target=target,
        model=model,
        task=task,
        features=features,
        preparer=prep,
        pos_label=pos_label,
        class_weight=bcfg["class_weight"] if task == "classification" else None,
        sampling_method=bcfg["sampling_method"] if task == "classification" else None,
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

    if task == "classification":
        y_true = evaluator.y_test
        y_score = get_positive_score(tuned_model, evaluator.X_test)

        resolved_pos_label = _resolve_valid_pos_label(y_true, pos_label)
        unique_labels = sorted(pd.Series(y_true).dropna().unique().tolist())

        if len(unique_labels) == 2 and resolved_pos_label is not None:
            tuned_metrics = {
                "Accuracy": float(accuracy_score(y_true, y_pred)),
                "Precision": float(precision_score(y_true, y_pred, pos_label=resolved_pos_label, zero_division=0)),
                "Recall": float(recall_score(y_true, y_pred, pos_label=resolved_pos_label, zero_division=0)),
                "F1": float(f1_score(y_true, y_pred, pos_label=resolved_pos_label, zero_division=0)),
                "ROC_AUC": float(roc_auc_score(y_true, y_score)) if y_score is not None else None,
            }
        else:
            tuned_metrics = {
                "Accuracy": float(accuracy_score(y_true, y_pred)),
                "Precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
                "Recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
                "F1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
                "ROC_AUC": None,
            }

    return {
        "best_params": result["best_params"],
        "best_score": result["best_score"],
        "metrics": tuned_metrics,
    }

def _coerce_pos_label(s: str):
    try:
        return int(s)
    except Exception:
        try:
            return float(s)
        except Exception:
            return s

def _resolve_valid_pos_label(y_true, desired_pos_label):
    unique_labels = sorted(pd.Series(y_true).dropna().unique().tolist())

    if desired_pos_label in unique_labels:
        return desired_pos_label

    if len(unique_labels) == 2:
        return unique_labels[-1]

    return None

def _avg_strategy(y_true) -> str:
    return "binary" if len(np.unique(y_true)) == 2 else "macro"

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔍 Explorador ML")
    st.caption("Herramienta genérica de análisis de datos")
    page = st.radio(
        "Sección",
        ["Datos", "EDA", "Supervisado", "No Supervisado", "Series de Tiempo"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    if st.session_state["df"] is not None:
        _d = st.session_state["df"]
        st.markdown(
            f'<div style="background:rgba(16,185,129,0.12);border:1px solid rgba(16,185,129,0.25);'
            f'border-radius:14px;padding:0.7rem 0.9rem;font-weight:600;color:#0f5132;">'
            f'● Dataset activo<br><span style="font-size:0.82rem;font-weight:400;">'
            f'{st.session_state["df_name"]}<br>{_d.shape[0]:,} filas × {_d.shape[1]} cols'
            f'</span></div>',
            unsafe_allow_html=True,
        )
    else:
        st.caption("Sin dataset cargado.")


# ══════════════════════════════════════════════════════════════════════════════
# 1) DATOS
# ══════════════════════════════════════════════════════════════════════════════
if page == "Datos":
    hero("Carga de Datos", "Sube un archivo CSV para comenzar la exploración.")

    up_col, sep_col = st.columns([3, 1])
    with up_col:
        uploaded = st.file_uploader("Archivo CSV", type=["csv"], label_visibility="collapsed")
    with sep_col:
        _sep_opts = {",": 'Coma  (",")', ";": 'Punto y coma  (";")',
                     "\t": "Tabulador", "|": 'Pipe  ("|")', " ": "Espacio"}
        sep = st.selectbox("Separador", list(_sep_opts.keys()), format_func=lambda x: _sep_opts[x])

    if uploaded:
        try:
            df = pd.read_csv(uploaded, sep=sep)
            st.session_state.update({
                "df": df, "df_raw": df, "df_name": uploaded.name,
                "_prep_cfg_exclude": [], "_prep_cfg_impute": "No imputar", "_prep_cfg_ohe": [],
                "sup_results": None, "sup_best_payload": None, "sup_target": None,
                "nn_result": None, "tuning_result": None, "stability_df": None,
                "unsup_cluster_result": None, "unsup_dim_result": None,
                "assoc_rules": None, "assoc_itemsets": None,
            })
            st.success(f"Dataset cargado: **{df.shape[0]:,}** filas × **{df.shape[1]}** columnas")
        except Exception as e:
            st.error(f"Error al leer el archivo: {e}")

    if st.session_state["df"] is not None:
        df = st.session_state["df"]
        c1, c2, c3, c4 = st.columns(4, gap="medium")
        raw_df = st.session_state.get("df_raw", df)
        original_dupes = int(raw_df.duplicated().sum())
        current_dupes = int(df.duplicated().sum())
        with c1: metric_card("Filas", f"{df.shape[0]:,}", "registros", "#4f46e5")
        with c2: metric_card("Columnas", str(df.shape[1]), "variables", "#2563eb")
        with c3: metric_card("Valores nulos", str(int(df.isnull().sum().sum())), "total en el dataset", "#f59e0b")
        with c4: metric_card("Duplicados originales", str(original_dupes), f"actuales: {current_dupes}", "#ef4444")

        panel_open("Vista previa", f"Primeras 20 filas de {st.session_state['df_name']}")
        st.dataframe(df.head(20), use_container_width=True)
        panel_close()

        panel_open("Tipos de datos y calidad")
        dtype_df = pd.DataFrame({
            "Columna": df.columns,
            "Tipo": df.dtypes.astype(str).values,
            "No nulos": df.count().values,
            "Nulos": df.isnull().sum().values,
            "Únicos": df.nunique().values,
        })
        st.dataframe(dtype_df, use_container_width=True, hide_index=True)
        panel_close()

        with st.expander("⚙️ Preparar datos", expanded=False):
            st.caption("Los cambios aplican a todas las secciones (EDA, Supervisado, No Supervisado).")
            _df_raw = st.session_state["df_raw"]

            # 1) Excluir columnas
            st.markdown("**Excluir columnas** (índices, IDs, columnas irrelevantes)")
            _excl_val = st.multiselect(
                "Columnas a excluir", _df_raw.columns.tolist(),
                default=st.session_state["_prep_cfg_exclude"],
                help="Selecciona columnas que no aportan al análisis.",
            )
            st.session_state["_prep_cfg_exclude"] = _excl_val

            st.divider()

            # 2) Imputación de nulos
            _null_counts_raw = _df_raw.isnull().sum()
            _cols_with_nulls = _null_counts_raw[_null_counts_raw > 0].index.tolist()
            if _cols_with_nulls:
                st.markdown(f"**Imputar nulos** — {len(_cols_with_nulls)} columna(s) con valores faltantes")
                _opts_impute = ["No imputar", "Eliminar filas con nulos", "Media (numéricas)",
                                "Mediana (numéricas)", "Moda (todas)"]
                _impute_val = st.selectbox(
                    "Estrategia global", _opts_impute,
                    index=_opts_impute.index(st.session_state["_prep_cfg_impute"]),
                )
                st.session_state["_prep_cfg_impute"] = _impute_val
            else:
                st.success("No hay valores nulos en el dataset.")

            st.divider()

            # 3) One-hot encoding
            st.markdown("**One-Hot Encoding** *(opcional)*")
            _cat_cols_ohe = [c for c in _df_raw.select_dtypes(include=["object", "category"]).columns
                             if c not in _excl_val]
            if _cat_cols_ohe:
                _valid_ohe = [c for c in st.session_state["_prep_cfg_ohe"] if c in _cat_cols_ohe]
                _ohe_val = st.multiselect(
                    "Columnas categóricas a codificar", _cat_cols_ohe,
                    default=_valid_ohe,
                    help="Se crean columnas 0/1 por cada categoría.",
                )
                st.session_state["_prep_cfg_ohe"] = _ohe_val
            else:
                st.caption("No hay columnas categóricas para codificar.")

            # Resumen del df resultante
            _proc_df = st.session_state["df"]
            st.info(f"Dataset procesado: {_proc_df.shape[0]:,} filas × {_proc_df.shape[1]} columnas")


# ══════════════════════════════════════════════════════════════════════════════
# 2) EDA
# ══════════════════════════════════════════════════════════════════════════════
elif page == "EDA":
    if not _require_df():
        st.stop()

    df = st.session_state["df"]
    hero("Análisis Exploratorio de Datos", f"Dataset: {st.session_state['df_name']}")

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    c1, c2, c3, c4 = st.columns(4, gap="medium")
    raw_df = st.session_state.get("df_raw", df)
    original_dupes = int(raw_df.duplicated().sum())
    current_dupes = int(df.duplicated().sum())
    with c1: metric_card("Numéricas", str(len(num_cols)), "columnas numéricas", "#4f46e5")
    with c2: metric_card("Categóricas", str(len(cat_cols)), "columnas no numéricas", "#7c3aed")
    with c3: metric_card("Valores nulos", str(int(df.isnull().sum().sum())), "total", "#f59e0b")
    with c4: metric_card("Duplicados originales", str(original_dupes), f"actuales: {current_dupes}", "#ef4444")

    tab_stats, tab_dist, tab_corr, tab_cat = st.tabs(
        ["Estadísticas", "Distribuciones", "Correlación", "Categóricas"]
    )

    with tab_stats:
        panel_open("Estadísticas descriptivas", "Variables numéricas")
        if num_cols:
            st.dataframe(df[num_cols].describe().T.round(4), use_container_width=True)
        else:
            st.info("No hay columnas numéricas.")
        panel_close()

        panel_open("Valores nulos por columna")
        null_df = df.isnull().sum().reset_index()
        null_df.columns = ["Columna", "Nulos"]
        null_df["Porcentaje (%)"] = (null_df["Nulos"] / len(df) * 100).round(2)
        null_df = null_df[null_df["Nulos"] > 0].sort_values("Nulos", ascending=False)
        if null_df.empty:
            st.success("Sin valores nulos en el dataset.")
        else:
            st.dataframe(null_df, use_container_width=True, hide_index=True)
        panel_close()

    with tab_dist:
        if not num_cols:
            st.info("No hay columnas numéricas para mostrar distribuciones.")
        else:
            d1, d2 = st.columns([2, 1])
            with d1:
                sel_col = st.selectbox("Variable", num_cols, key="eda_dist_col")
            with d2:
                nbins = st.slider("Bins", 10, 100, 30, key="eda_bins")

            color_col = st.selectbox("Colorear por (opcional)", ["—"] + df.columns.tolist(), key="eda_color")

            fig_h = px.histogram(
                df, x=sel_col, nbins=nbins,
                color=color_col if color_col != "—" else None,
                barmode="overlay" if color_col != "—" else "relative",
                opacity=0.8,
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            st.plotly_chart(_fig_layout(fig_h), use_container_width=True, config={"displayModeBar": False})

            box_col = st.selectbox("Boxplot — colorear por", ["—"] + df.columns.tolist(), key="eda_box_color")
            fig_box = px.box(df, y=sel_col, color=box_col if box_col != "—" else None,
                             color_discrete_sequence=px.colors.qualitative.Set2)
            st.plotly_chart(_fig_layout(fig_box, height=320), use_container_width=True, config={"displayModeBar": False})

    with tab_corr:
        if len(num_cols) < 2:
            st.info("Se necesitan al menos 2 columnas numéricas.")
        else:
            target_corr = st.selectbox("Target para correlación con features", ["—"] + num_cols, key="eda_corr_t")
            if target_corr != "—":
                fig_bar = viz.top_target_correlation_bar(df, target_col=target_corr, top_n=15)
                if fig_bar:
                    panel_open(f"Top correlaciones con '{target_corr}'")
                    st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})
                    panel_close()

            panel_open("Mapa de calor — correlación", "Solo variables numéricas")
            fig_heat = viz.correlation_heatmap(df[num_cols])
            if fig_heat:
                st.plotly_chart(fig_heat, use_container_width=True, config={"displayModeBar": False})
            panel_close()

    with tab_cat:
        if not cat_cols:
            st.info("No hay columnas categóricas.")
        else:
            cat_sel = st.selectbox("Variable categórica", cat_cols, key="eda_cat")
            top_n = st.slider("Top N categorías", 5, 30, 15, key="eda_cat_n")

            vc = df[cat_sel].value_counts().head(top_n).reset_index()
            vc.columns = [cat_sel, "Frecuencia"]
            fig_cat = px.bar(
                vc.sort_values("Frecuencia"), x="Frecuencia", y=cat_sel,
                orientation="h", color="Frecuencia",
                color_continuous_scale=["#60a5fa", "#7c3aed"],
            )
            fig_cat.update_layout(coloraxis_showscale=False)
            st.plotly_chart(_fig_layout(fig_cat, height=max(300, top_n * 28)),
                            use_container_width=True, config={"displayModeBar": False})

            target_donut = st.selectbox("Distribución de target (donut)", ["—"] + df.columns.tolist(), key="eda_donut")
            if target_donut != "—":
                label_map = {v: str(v) for v in df[target_donut].unique()}
                fig_donut = viz.target_distribution_donut(df, target_col=target_donut, label_map=label_map)
                if fig_donut:
                    st.plotly_chart(fig_donut, use_container_width=True, config={"displayModeBar": False})


# ══════════════════════════════════════════════════════════════════════════════
# 3) SUPERVISADO
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Supervisado":
    if not _require_df():
        st.stop()

    df = st.session_state["df"]
    hero("Aprendizaje Supervisado", "Clasificación, regresión, redes neuronales, tuning y estabilidad.")

    with st.form("sup_config"):
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            target_col = st.selectbox("Columna target", df.columns.tolist())
        with fc2:
            task = st.selectbox("Tarea", ["classification", "regression"],
                                format_func=lambda x: "Clasificación" if x == "classification" else "Regresión")
        with fc3:
            balance = st.selectbox("Balanceo (solo clasificación)",
                                   list(_BALANCE_LABELS.keys()), format_func=lambda x: _BALANCE_LABELS[x])

        feat_opts = [c for c in df.columns if c != target_col]
        features_sel = st.multiselect("Features (vacío = todas excepto target)", feat_opts)

        sc1, sc2, sc3, sc4 = st.columns(4)
        with sc1: train_size = st.slider("Train size", 0.5, 0.9, 0.75, 0.05)
        with sc2: rs = int(st.number_input("Random state", 0, value=42, step=1))
        with sc3: use_cv = st.toggle("Cross-validation", value=True)
        with sc4: n_splits = st.slider("Folds", 3, 10, 5) if use_cv else 5

        unique_vals = (sorted(df[target_col].dropna().unique().tolist())
                       if target_col in df.columns else [1])
        pos_opts = [str(v) for v in unique_vals]
        pl_col, _ = st.columns([1, 3])
        with pl_col:
            pos_label_str = (st.selectbox("Clase positiva (clasificación)", pos_opts)
                             if len(pos_opts) <= 20 else st.text_input("Clase positiva", "1"))

        apply_btn = st.form_submit_button("Aplicar configuración", use_container_width=True)

    if apply_btn:
        st.session_state.update({
            "sup_target": target_col, "sup_task": task,
            "sup_features": features_sel, "sup_balance": balance,
            "sup_train_size": train_size, "sup_rs": rs,
            "sup_use_cv": use_cv, "sup_n_splits": n_splits,
            "sup_pos_label": pos_label_str,
            "sup_results": None, "sup_best_payload": None,
            "nn_result": None, "tuning_result": None, "stability_df": None,
        })

    target = st.session_state.get("sup_target")
    if target is None:
        st.info("Configura y aplica los parámetros arriba.")
        st.stop()

    task        = st.session_state["sup_task"]
    features    = st.session_state["sup_features"] or []
    balance     = st.session_state["sup_balance"]
    train_size  = st.session_state["sup_train_size"]
    rs          = st.session_state["sup_rs"]
    use_cv      = st.session_state["sup_use_cv"]
    n_splits    = st.session_state["sup_n_splits"]
    pos_label   = _coerce_pos_label(st.session_state["sup_pos_label"])

    df_model, sup_meta = _build_supervised_df(target, features)
    if df_model.empty:
        st.error("No hay suficientes datos válidos para entrenar después del preprocesamiento de supervisado.")
        st.stop()

    st.info(
        f"Modelado supervisado sin imputación global previa al split: "
        f"{sup_meta['rows_final']:,} filas listas | "
        f"duplicados eliminados: {sup_meta['dropped_duplicates']:,} | "
        f"filas con target nulo: {sup_meta['dropped_target_nulls']:,} | "
        f"filas con nulos restantes eliminadas: {sup_meta['dropped_null_rows']:,}."
    )
    if sup_meta.get("auto_ohe_columns"):
        st.caption("Codificación automática para supervisado: " + ", ".join(sup_meta["auto_ohe_columns"][:12]) + ("..." if len(sup_meta["auto_ohe_columns"]) > 12 else ""))

    clf_names = ["Regresión Logística", "Random Forest", "SVM", "XGBoost", "LightGBM"]
    reg_names = ["Regresión Lineal", "Ridge", "Random Forest", "XGBoost", "LightGBM"]
    model_names = clf_names if task == "classification" else reg_names

    tab_models, tab_nn, tab_tuning, tab_stability = st.tabs(
        ["Comparación de modelos", "Redes Neuronales", "Tuning", "Estabilidad"]
    )

    # ── Tab: Comparación ──────────────────────────────────────────────────────
    with tab_models:
        if st.button("Ejecutar comparación de modelos", type="primary", key="run_compare"):
            bcfg = _balance_cfg(balance)
            results, errs = [], []
            prog = st.progress(0, text="Entrenando modelos...")
            for i, name in enumerate(model_names):
                try:
                    model = _clf_model(name, rs) if task == "classification" else _reg_model(name, rs)
                    prep  = DataPreparer(train_size=train_size, random_state=rs,
                                        scale_X=(name in _SCALE_MODELS))
                    runner = SupervisedRunner(
                        df=df_model, target=target, model=model, task=task,
                        features=features, preparer=prep, pos_label=pos_label,
                        class_weight=bcfg["class_weight"] if task == "classification" else None,
                        sampling_method=bcfg["sampling_method"] if task == "classification" else None,
                    )
                    m  = runner.evaluate()
                    cv = runner.evaluate_cv(n_splits=n_splits) if use_cv else {}
                    row = {"Modelo": name}
                    row.update({k: v for k, v in m.items()  if isinstance(v, (int, float, np.floating))})
                    row.update({f"{k}_CV": v for k, v in cv.items() if isinstance(v, (int, float, np.floating))})
                    results.append(row)
                except Exception as e:
                    errs.append(f"{name}: {e}")
                prog.progress((i + 1) / len(model_names), text=f"Listo: {name}")
            prog.empty()
            st.session_state["sup_results"] = results
            st.session_state["sup_best_payload"] = None
            for err in errs:
                st.warning(err)

        if st.session_state["sup_results"]:
            res_df = pd.DataFrame(st.session_state["sup_results"])
            sort_opts = [c for c in res_df.columns if c != "Modelo"
                         and pd.api.types.is_numeric_dtype(res_df[c])]
            sort_col = st.selectbox("Ordenar por", sort_opts, key="sup_sort") if sort_opts else None
            if sort_col:
                res_df = res_df.sort_values(sort_col, ascending=False).reset_index(drop=True)

            panel_open("Ranking de modelos")
            st.dataframe(res_df.round(4), use_container_width=True, hide_index=True)
            panel_close()

            best_name = res_df.iloc[0]["Modelo"]

            if st.button(f"Ver detalle: {best_name}", key="detail_btn"):
                try:
                    bcfg = _balance_cfg(balance)
                    model = _clf_model(best_name, rs) if task == "classification" else _reg_model(best_name, rs)
                    prep  = DataPreparer(train_size=train_size, random_state=rs,
                                        scale_X=(best_name in _SCALE_MODELS))
                    runner = SupervisedRunner(
                        df=df_model, target=target, model=model, task=task,
                        features=features, preparer=prep, pos_label=pos_label,
                        class_weight=bcfg["class_weight"] if task == "classification" else None,
                        sampling_method=bcfg["sampling_method"] if task == "classification" else None,
                    )
                    y_pred = runner.fit_predict()
                    y_true = runner.y_test

                    if task == "classification":
                        avg = _avg_strategy(y_true)
                        y_score = get_positive_score(runner.model, runner.X_test)
                        metrics_payload = {
                            "Accuracy":  float(accuracy_score(y_true, y_pred)),
                            "Precision": float(precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0, average=avg)),
                            "Recall":    float(recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0, average=avg)),
                            "F1":        float(f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0, average=avg)),
                        }
                        roc_df_p = pr_df_p = auc = None
                        if y_score is not None and len(np.unique(y_true)) == 2:
                            try:
                                auc = float(roc_auc_score(y_true, y_score))
                                fpr, tpr, _ = roc_curve(y_true, y_score)
                                pc, rc, _   = precision_recall_curve(y_true, y_score)
                                roc_df_p = pd.DataFrame({"fpr": fpr, "tpr": tpr})
                                pr_df_p  = pd.DataFrame({"recall": rc, "precision": pc})
                            except Exception:
                                pass
                        metrics_payload["ROC_AUC"] = auc
                        cm = confusion_matrix(y_true, y_pred)
                        st.session_state["sup_best_payload"] = {
                            "name": best_name, "task": "classification",
                            "metrics": metrics_payload, "auc": auc,
                            "roc_df": roc_df_p, "pr_df": pr_df_p,
                            "cm": cm, "y_true": y_true, "y_pred": y_pred, "n": len(y_true),
                        }
                    else:
                        st.session_state["sup_best_payload"] = {
                            "name": best_name, "task": "regression",
                            "MAE":  float(mean_absolute_error(y_true, y_pred)),
                            "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
                            "R2":   float(r2_score(y_true, y_pred)),
                            "y_true": y_true, "y_pred": y_pred, "n": len(y_true),
                        }
                except Exception as e:
                    st.error(f"Error: {e}")

            payload = st.session_state.get("sup_best_payload")
            if payload and payload.get("name") == best_name:
                if payload["task"] == "classification":
                    m = payload["metrics"]
                    auc_str = f"{payload['auc']:.4f}" if payload.get("auc") is not None else "N/A"
                    c1,c2,c3,c4,c5 = st.columns(5, gap="medium")
                    with c1: metric_card("Accuracy",  f"{m['Accuracy']:.4f}",  "test", "#4f46e5")
                    with c2: metric_card("Precision", f"{m['Precision']:.4f}", "test", "#10b981")
                    with c3: metric_card("Recall",    f"{m['Recall']:.4f}",    "test", "#2563eb")
                    with c4: metric_card("F1",        f"{m['F1']:.4f}",        "test", "#7c3aed")
                    with c5: metric_card("ROC-AUC",   auc_str,                 "test", "#f59e0b")

                    rc1, rc2 = st.columns(2, gap="large")
                    with rc1:
                        if payload.get("roc_df") is not None:
                            panel_open("Curva ROC", f"AUC = {payload['auc']:.4f}")
                            st.plotly_chart(viz.roc_curve_plot(payload["roc_df"], auc_value=payload["auc"]),
                                            use_container_width=True, config={"displayModeBar": False})
                            panel_close()
                    with rc2:
                        if payload.get("pr_df") is not None:
                            panel_open("Curva Precision-Recall")
                            st.plotly_chart(viz.precision_recall_plot(payload["pr_df"]),
                                            use_container_width=True, config={"displayModeBar": False})
                            panel_close()

                    cm = payload["cm"]
                    n  = payload["n"]
                    panel_open("Matriz de confusión", f"Conjunto de prueba: {n} registros")
                    if cm.shape == (2, 2):
                        tn, fp, fn, tp = cm.ravel()
                        ca, cb = st.columns(2, gap="medium")
                        with ca:
                            confusion_box(tp, "Verdaderos Positivos", tp/n*100, "#dcfce7", "#22c55e", "#166534")
                            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
                            confusion_box(fp, "Falsos Positivos",      fp/n*100, "#ffedd5", "#f59e0b", "#9a3412")
                        with cb:
                            confusion_box(fn, "Falsos Negativos",      fn/n*100, "#fee2e2", "#ef4444", "#991b1b")
                            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
                            confusion_box(tn, "Verdaderos Negativos",  tn/n*100, "#dbeafe", "#3b82f6", "#1d4ed8")
                    else:
                        lbls = sorted(np.unique(payload["y_true"]))
                        cm_df = pd.DataFrame(cm, index=[f"Real_{l}" for l in lbls],
                                             columns=[f"Pred_{l}" for l in lbls])
                        st.dataframe(cm_df, use_container_width=True)
                    panel_close()
                else:
                    c1, c2, c3 = st.columns(3, gap="medium")
                    with c1: metric_card("MAE",  f"{payload['MAE']:.4f}",  "error absoluto medio",   "#4f46e5")
                    with c2: metric_card("RMSE", f"{payload['RMSE']:.4f}", "raíz error cuadrático",  "#7c3aed")
                    with c3: metric_card("R²",   f"{payload['R2']:.4f}",   "coef. de determinación", "#10b981")

                    panel_open("Real vs Predicho")
                    sdf = pd.DataFrame({"Real": payload["y_true"], "Predicho": payload["y_pred"]})
                    fig_sc = px.scatter(sdf, x="Real", y="Predicho", opacity=0.6,
                                        color_discrete_sequence=["#4f46e5"])
                    fig_sc.add_shape(type="line",
                                     x0=sdf["Real"].min(), y0=sdf["Real"].min(),
                                     x1=sdf["Real"].max(), y1=sdf["Real"].max(),
                                     line=dict(color="#ef4444", dash="dash"))
                    st.plotly_chart(_fig_layout(fig_sc), use_container_width=True, config={"displayModeBar": False})
                    panel_close()

    # ── Tab: Redes Neuronales ─────────────────────────────────────────────────
    with tab_nn:
        with st.form("nn_form"):
            nn1, nn2, nn3 = st.columns(3)
            with nn1:
                nn_layers = st.text_input("Capas ocultas", value="64,32",
                                          help="Neuronas por capa separadas por coma. Ej: 128,64,32")
                nn_act    = st.selectbox("Activación", ["relu", "tanh", "logistic"])
            with nn2:
                nn_solver = st.selectbox("Solver", ["adam", "sgd", "lbfgs"])
                nn_lr     = st.number_input("Learning rate", 0.0001, 0.1, 0.001, format="%.4f")
            with nn3:
                nn_iter  = st.slider("Max iteraciones", 100, 2000, 500, 100)
                nn_early = st.toggle("Early stopping", value=True)
            nn_cv  = st.toggle("Evaluar con CV", value=True)
            run_nn = st.form_submit_button("Entrenar red neuronal", use_container_width=True)

        if run_nn:
            try:
                layers = tuple(int(x.strip()) for x in nn_layers.split(",") if x.strip())
                runner_nn = NeuralNetworkRunner(
                    df=df_model, target=target, task=task, features=features,
                    hidden_layer_sizes=layers, activation=nn_act,
                    solver=nn_solver, learning_rate_init=nn_lr,
                    max_iter=nn_iter, early_stopping=nn_early,
                    random_state=rs, pos_label=pos_label,
                )
                with st.spinner("Entrenando red neuronal..."):
                    m_nn  = runner_nn.evaluate()
                    cv_nn = runner_nn.evaluate_cv(n_splits=n_splits) if nn_cv else {}
                st.session_state["nn_result"] = {
                    "arch":    runner_nn.architecture(),
                    "holdout": {k: v for k, v in m_nn.items()  if isinstance(v, (int, float, np.floating))},
                    "cv":      {k: v for k, v in cv_nn.items() if isinstance(v, (int, float, np.floating))},
                }
            except Exception as e:
                st.error(f"Error en red neuronal: {e}")

        nn_res = st.session_state.get("nn_result")
        if nn_res:
            panel_open("Arquitectura")
            st.dataframe(pd.DataFrame([{"Parámetro": k, "Valor": str(v)}
                                        for k, v in nn_res["arch"].items()]),
                         use_container_width=True, hide_index=True)
            panel_close()

            panel_open("Métricas de evaluación", "Holdout y Cross-Validation en una sola tabla")

            holdout_dict = nn_res.get("holdout", {})
            cv_dict = nn_res.get("cv", {})

            metric_names = sorted(
                set(holdout_dict.keys()) |
                {k for k in cv_dict.keys() if not k.endswith("_std")}
            )

            merged_rows = []
            for metric in metric_names:
                holdout_val = holdout_dict.get(metric)
                cv_mean_val = cv_dict.get(metric)
                cv_std_val = cv_dict.get(f"{metric}_std")

                merged_rows.append({
                    "Métrica": metric,
                    "Holdout": round(float(holdout_val), 4) if holdout_val is not None else None,
                    "CV Mean": round(float(cv_mean_val), 4) if cv_mean_val is not None else None,
                    "CV Std": round(float(cv_std_val), 4) if cv_std_val is not None else None,
                })

            st.dataframe(pd.DataFrame(merged_rows), use_container_width=True, hide_index=True)
            panel_close()

    # ── Tab: Tuning ───────────────────────────────────────────────────────────
    with tab_tuning:
        if not st.session_state.get("sup_results"):
            st.info("Ejecuta primero la comparación de modelos para seleccionar el mejor.")
        else:
            res_df_t = pd.DataFrame(st.session_state["sup_results"])
            sort_opts_t = [c for c in res_df_t.columns if c != "Modelo" and pd.api.types.is_numeric_dtype(res_df_t[c])]
            sort_col_t = st.selectbox("Criterio para mejor modelo", sort_opts_t, key="tuning_sort_model")
            if sort_col_t:
                res_df_t = res_df_t.sort_values(sort_col_t, ascending=False).reset_index(drop=True)

            best_name_t = res_df_t.iloc[0]["Modelo"]
            st.markdown(f"**Modelo a optimizar:** {best_name_t}")

            panel_open("Resumen de selección del modelo", "Información usada para elegir el modelo ganador.")
            s1, s2 = st.columns(2, gap="medium")
            with s1:
                metric_card("Modelo ganador", best_name_t, "Seleccionado por ranking actual", "#2563eb")
            with s2:
                metric_card("Criterio", sort_col_t, "Orden actual del ranking", "#7c3aed")
            panel_close()

            best_model_params_df = _get_best_model_params_df(best_name_t, rs, train_size, balance, task)
            configured_best_model = _build_configured_model(best_name_t, rs, train_size, balance, task)

            panel_open("Hiperparámetros reales del modelo ganador", "Parámetros obtenidos directamente desde get_params().")
            st.dataframe(best_model_params_df, use_container_width=True, hide_index=True)
            panel_close()

            panel_open("Búsqueda de hiperparámetros", "Optimiza el mejor modelo actual bajo demanda.")
            genetic_available = _is_genetic_search_available()
            tuning_method_options = ["Exhaustive"] + (["Genetic"] if genetic_available else [])

            with st.form("best_model_tuning_form"):
                tune_col_1, tune_col_2, tune_col_3 = st.columns(3, gap="medium")
                with tune_col_1:
                    tuning_method = st.selectbox("Método de búsqueda", options=tuning_method_options, index=0)
                with tune_col_2:
                    tuning_cv = st.slider("Folds para búsqueda", min_value=3, max_value=10, value=5)
                with tune_col_3:
                    tuning_scoring = st.selectbox(
                        "Scoring",
                        options=["f1", "roc_auc", "accuracy"] if task == "classification" else ["r2", "neg_mean_squared_error", "neg_mean_absolute_error"],
                        index=0,
                    )

                if not genetic_available:
                    st.caption("La búsqueda genética se habilita instalando `sklearn-genetic-opt`.")

                st.caption(
                    "Nota: el tuning hace cross-validation sobre entrenamiento y luego evalúa el mejor modelo en test."
                )

                search_grid, search_grid_errors = _build_best_model_search_grid(
                    best_name_t,
                    configured_best_model.get_params(),
                )

                if search_grid_errors:
                    for error in search_grid_errors:
                        st.error(error)

                st.caption(f"Parámetros incluidos en la búsqueda: {', '.join(search_grid.keys())}")

                run_tuning = st.form_submit_button("Ejecutar búsqueda sobre el mejor modelo", use_container_width=True)

            if run_tuning:
                if search_grid_errors:
                    st.error("Corrige los parámetros inválidos antes de ejecutar la búsqueda.")
                elif not search_grid:
                    st.warning("Define al menos un parámetro en el grid.")
                else:
                    with st.spinner("Buscando mejores hiperparámetros..."):
                        tuning_result = _run_best_model_search(
                            best_name_t,
                            rs,
                            train_size,
                            balance,
                            task,
                            tuning_method,
                            tuning_cv,
                            tuning_scoring,
                            search_grid,
                        )

                    st.session_state["tuning_result"] = tuning_result

            tuning_res = st.session_state.get("tuning_result")
            if tuning_res:
                best_score_value = tuning_res["best_score"]
                tuned_metrics = tuning_res["metrics"]
                tuned_params_df = pd.DataFrame(
                    [{"Parámetro": key, "Valor": str(value)} for key, value in tuning_res["best_params"].items()]
                )

                tcols = st.columns(len(tuned_metrics) + 1, gap="medium")
                with tcols[0]:
                    metric_card("Best CV Score", f"{best_score_value:.4f}", f"Scoring: {tuning_scoring}", "#2563eb")

                colors = ["#7c3aed", "#10b981", "#f59e0b", "#06b6d4", "#ef4444"]
                for i, (mk, mv) in enumerate(tuned_metrics.items()):
                    with tcols[i + 1]:
                        metric_card(mk, f"{mv:.4f}" if mv is not None else "N/A", "Conjunto de prueba", colors[i % len(colors)])

                panel_open("Mejores parámetros encontrados")
                st.dataframe(tuned_params_df, use_container_width=True, hide_index=True)
                panel_close()

            panel_close()

    # ── Tab: Estabilidad ──────────────────────────────────────────────────────
    with tab_stability:
        if not st.session_state.get("sup_results"):
            st.info("Ejecuta primero la comparación de modelos.")
        else:
            best_name_s = pd.DataFrame(st.session_state["sup_results"]).iloc[0]["Modelo"]
            st.markdown(f"**Modelo evaluado:** {best_name_s}")

            seeds_txt = st.text_input("Semillas (separadas por comas)", "42, 0, 7, 13, 21")
            if st.button("Ejecutar análisis de estabilidad", type="primary"):
                try:
                    seeds = [int(s.strip()) for s in seeds_txt.split(",") if s.strip()]
                    bcfg  = _balance_cfg(balance)
                    rows_s = []
                    prog_s = st.progress(0, text="Evaluando semillas...")
                    for i, seed in enumerate(seeds):
                        model_s = (_clf_model(best_name_s, seed) if task == "classification"
                                   else _reg_model(best_name_s, seed))
                        prep_s  = DataPreparer(train_size=train_size, random_state=seed,
                                               scale_X=(best_name_s in _SCALE_MODELS))
                        runner_s = SupervisedRunner(
                            df=df_model, target=target, model=model_s, task=task,
                            features=features, preparer=prep_s, pos_label=pos_label,
                            class_weight=bcfg["class_weight"] if task == "classification" else None,
                            sampling_method=bcfg["sampling_method"] if task == "classification" else None,
                        )
                        m_s  = runner_s.evaluate()
                        cv_s = runner_s.evaluate_cv(n_splits=n_splits) if use_cv else {}
                        row_s = {"Seed": seed}
                        row_s.update({k: v for k, v in m_s.items()  if isinstance(v, (int, float, np.floating))})
                        row_s.update({f"{k}_CV": v for k, v in cv_s.items()
                                       if isinstance(v, (int, float, np.floating)) and not k.endswith("_std")})
                        rows_s.append(row_s)
                        prog_s.progress((i + 1) / len(seeds), text=f"Semilla {seed}")
                    prog_s.empty()
                    st.session_state["stability_df"] = pd.DataFrame(rows_s)
                except Exception as e:
                    st.error(f"Error: {e}")

            stab = st.session_state.get("stability_df")
            if stab is not None and not stab.empty:
                num_s = [c for c in stab.columns if c != "Seed"
                         and pd.api.types.is_numeric_dtype(stab[c])]
                if num_s:
                    primary = num_s[0]
                    s1, s2, s3 = st.columns(3, gap="medium")
                    with s1: metric_card("Semillas", str(len(stab)), "ejecuciones", "#2563eb")
                    with s2: metric_card(f"Media {primary}", f"{stab[primary].mean():.4f}", "promedio", "#7c3aed")
                    with s3: metric_card("Desv. estándar", f"{stab[primary].std():.4f}", primary, "#06b6d4")

                panel_open("Resultados por semilla")
                st.dataframe(stab.round(4), use_container_width=True, hide_index=True)
                panel_close()

                if num_s:
                    vis_m = st.selectbox("Visualizar métrica", num_s, key="stab_vis")
                    fig_stab = px.line(stab, x="Seed", y=vis_m, markers=True,
                                       color_discrete_sequence=["#4f46e5"])
                    st.plotly_chart(_fig_layout(fig_stab, 300), use_container_width=True,
                                    config={"displayModeBar": False})


# ══════════════════════════════════════════════════════════════════════════════
# 4) NO SUPERVISADO
# ══════════════════════════════════════════════════════════════════════════════
elif page == "No Supervisado":
    if not _require_df():
        st.stop()

    df = st.session_state["df"]
    hero("Aprendizaje No Supervisado", "Clustering, reducción dimensional y reglas de asociación.")

    df_encoded = df
    num_cols_ns = df_encoded.select_dtypes(include=np.number).columns.tolist()

    tab_cluster, tab_dim, tab_assoc = st.tabs(
        ["Clustering", "Reducción Dimensional", "Reglas de Asociación"]
    )

    # ── Clustering ────────────────────────────────────────────────────────────
    with tab_cluster:
        feat_cl = st.multiselect("Features numéricas", num_cols_ns,
                                 default=num_cols_ns[:min(5, len(num_cols_ns))],
                                 key="cl_feats")
        if feat_cl:
            cl1, cl2, cl3 = st.columns(3)
            with cl1:
                cl_algo = st.selectbox("Algoritmo", ["KMeans", "HAC (Jerárquico)"], key="cl_algo")
            with cl2:
                k = st.slider("Número de clusters (k)", 2, 15, 3, key="cl_k")
            with cl3:
                linkage = st.selectbox("Linkage (HAC)", ["ward", "complete", "average", "single"])
            _vc1, _vc2, _vc3 = st.columns(3)
            with _vc1:
                color_cl = st.selectbox("Colorear por", ["cluster"] + df.columns.tolist(), key="cl_color")
            with _vc2:
                _vis_proj = st.selectbox("Proyección", ["PCA", "t-SNE"], key="cl_proj",
                                         help="Método para proyectar los clusters al espacio visual.")
            with _vc3:
                _vis_dims = st.radio("Dimensiones", ["2D", "3D"], horizontal=True, key="cl_vis_dims_radio")
            _n_vis = 3 if _vis_dims == "3D" else 2

            if st.button("Ejecutar clustering", type="primary"):
                try:
                    X_cl = df_encoded[feat_cl].dropna()
                    model_cl = (KMeans(n_clusters=k, random_state=42, n_init="auto")
                                if cl_algo == "KMeans"
                                else AgglomerativeClustering(n_clusters=k, linkage=linkage))
                    kind_cl = "kmeans" if cl_algo == "KMeans" else "hac"
                    runner_cl = UnsupervisedRunner(
                        name=cl_algo, X=X_cl, model=model_cl, kind=kind_cl, scale_X=True
                    )
                    runner_cl.fit()
                    _X_scaled = runner_cl.X
                    if _vis_proj == "PCA":
                        emb_cl = PCA(n_components=_n_vis, random_state=42).fit_transform(_X_scaled)
                    elif _vis_proj == "t-SNE":
                        from sklearn.manifold import TSNE
                        emb_cl = TSNE(n_components=_n_vis, random_state=42,
                                      perplexity=min(30, len(_X_scaled)-1)).fit_transform(_X_scaled)
                    else:
                        from sklearn.manifold import TSNE
                        emb_cl = TSNE(n_components=_n_vis, random_state=42,
                                      perplexity=min(30, len(_X_scaled)-1)).fit_transform(_X_scaled)
                    st.session_state["unsup_cluster_result"] = {
                        "embedding": emb_cl, "labels": runner_cl.labels_,
                        "metrics": runner_cl.metrics, "algo": cl_algo,
                        "proj": _vis_proj, "index": X_cl.index,
                    }
                except Exception as e:
                    st.error(f"Error en clustering: {e}")

            cr = st.session_state.get("unsup_cluster_result")
            if cr:
                if cr["metrics"]:
                    mc = st.columns(len(cr["metrics"]), gap="medium")
                    for i, (mk, mv) in enumerate(cr["metrics"].items()):
                        with mc[i]:
                            metric_card(mk.capitalize(), f"{float(mv):.4f}", cr["algo"],
                                        ["#4f46e5","#7c3aed","#10b981","#f59e0b"][i % 4])

                emb         = cr["embedding"]
                labels      = cr["labels"].astype(str)
                _is3d       = emb.shape[1] >= 3
                _cl_palette = px.colors.qualitative.Bold
                _unique_cl  = sorted(set(labels), key=lambda x: int(x))
                _cl_colors  = {c: _cl_palette[i % len(_cl_palette)]
                               for i, c in enumerate(_unique_cl)}

                if color_cl != "cluster" and color_cl in df.columns:
                    _ext_color = df.loc[cr["index"], color_cl].values
                    if _is3d:
                        fig_cl = px.scatter_3d(
                            pd.DataFrame({"PC1": emb[:,0], "PC2": emb[:,1],
                                          "PC3": emb[:,2], "Color": _ext_color,
                                          "Cluster": labels}),
                            x="PC1", y="PC2", z="PC3", color="Color",
                            hover_data=["Cluster"], opacity=0.75,
                        )
                        fig_cl.update_traces(marker=dict(size=4))
                    else:
                        fig_cl = px.scatter(
                            pd.DataFrame({"PC1": emb[:,0], "PC2": emb[:,1],
                                          "Color": _ext_color, "Cluster": labels}),
                            x="PC1", y="PC2", color="Color",
                            hover_data=["Cluster"], opacity=0.75, symbol="Cluster",
                        )
                else:
                    fig_cl = go.Figure()
                    for _cl in _unique_cl:
                        _mask = np.array(labels) == _cl
                        _col  = _cl_colors[_cl]
                        if _is3d:
                            fig_cl.add_trace(go.Scatter3d(
                                x=emb[_mask, 0], y=emb[_mask, 1], z=emb[_mask, 2],
                                mode="markers",
                                marker=dict(size=4, color=_col, opacity=0.75),
                                name=f"Cluster {_cl}",
                                hovertemplate=f"Cluster {_cl}<br>PC1: %{{x:.2f}}<br>PC2: %{{y:.2f}}<br>PC3: %{{z:.2f}}<extra></extra>",
                            ))
                            # Centroide 3D
                            fig_cl.add_trace(go.Scatter3d(
                                x=[emb[_mask, 0].mean()],
                                y=[emb[_mask, 1].mean()],
                                z=[emb[_mask, 2].mean()],
                                mode="markers+text",
                                marker=dict(size=8, color=_col, symbol="cross",
                                            line=dict(width=2, color="black")),
                                text=[f"C{_cl}"], textposition="top center",
                                showlegend=False, hoverinfo="skip",
                            ))
                        else:
                            fig_cl.add_trace(go.Scatter(
                                x=emb[_mask, 0], y=emb[_mask, 1], mode="markers",
                                marker=dict(size=7, color=_col, opacity=0.75,
                                            line=dict(width=0.3, color="white")),
                                name=f"Cluster {_cl}",
                                hovertemplate=f"Cluster {_cl}<br>PC1: %{{x:.2f}}<br>PC2: %{{y:.2f}}<extra></extra>",
                            ))
                            fig_cl.add_trace(go.Scatter(
                                x=[emb[_mask, 0].mean()], y=[emb[_mask, 1].mean()],
                                mode="markers+text",
                                marker=dict(size=14, color=_col, symbol="x",
                                            line=dict(width=2, color="black")),
                                text=[f"C{_cl}"], textposition="top center",
                                textfont=dict(size=11, color="black"),
                                showlegend=False, hoverinfo="skip",
                            ))
                    if not _is3d:
                        fig_cl.update_layout(
                            legend=dict(title="Cluster"),
                            plot_bgcolor="#f9fafb",
                            xaxis=dict(showgrid=True, gridcolor="#e5e7eb", zeroline=False),
                            yaxis=dict(showgrid=True, gridcolor="#e5e7eb", zeroline=False),
                        )

                _dim_label  = "3D" if _is3d else "2D"
                _proj_label = cr.get("proj", "PCA")
                panel_open(f"Clusters {cr['algo']} (proyección {_proj_label} {_dim_label})")
                if _is3d:
                    fig_cl.update_layout(height=520, margin=dict(l=0, r=0, t=40, b=0))
                    st.plotly_chart(fig_cl, use_container_width=True)
                else:
                    st.plotly_chart(_fig_layout(fig_cl, 460), use_container_width=True,
                                    config={"displayModeBar": False})
                panel_close()

            if cl_algo != "KMeans" and feat_cl:
                with st.expander("Dendrograma", expanded=False):
                    _hac_linkage = st.selectbox("Linkage para dendrograma",
                                                ["ward", "complete", "average", "single"],
                                                key="dendro_linkage")
                    _hac_max_leaf = st.slider("Máx. hojas visibles", 10, 100, 30,
                                              key="dendro_leaves")
                    if st.button("Calcular Dendrograma", key="dendro_btn"):
                        from scipy.cluster.hierarchy import linkage as sp_linkage, dendrogram
                        import matplotlib.pyplot as plt
                        X_hac = df[feat_cl].dropna()
                        X_sc  = StandardScaler().fit_transform(X_hac)
                        Z     = sp_linkage(X_sc, method=_hac_linkage)
                        fig_d, ax = plt.subplots(figsize=(12, 4))
                        dendrogram(Z, ax=ax, truncate_mode="lastp",
                                   p=_hac_max_leaf, leaf_rotation=90,
                                   leaf_font_size=9, show_contracted=True,
                                   color_threshold=0.7 * max(Z[:, 2]))
                        ax.set_title(f"Dendrograma HAC — linkage: {_hac_linkage}", fontsize=13)
                        ax.set_xlabel("Observaciones (o tamaño del grupo)")
                        ax.set_ylabel("Distancia")
                        ax.spines[["top", "right"]].set_visible(False)
                        fig_d.tight_layout()
                        st.pyplot(fig_d, use_container_width=True)
                        plt.close(fig_d)
                        st.caption(
                            "Cada unión representa una fusión de clusters. "
                            "Corta el árbol a la altura donde los saltos verticales sean más largos "
                            "para elegir el número óptimo de clusters."
                        )

            if cl_algo == "KMeans" and feat_cl:
                with st.expander("Método del codo (Elbow)", expanded=False):
                    if st.button("Calcular Elbow", key="elbow_btn"):
                        inertias = []
                        X_el = df[feat_cl].dropna()
                        X_sc = pd.DataFrame(StandardScaler().fit_transform(X_el), columns=feat_cl)
                        k_range = range(2, min(12, len(X_sc)))
                        prog_e = st.progress(0)
                        for ki, kv in enumerate(k_range):
                            km = KMeans(n_clusters=kv, random_state=42, n_init="auto")
                            km.fit(X_sc)
                            inertias.append({"k": kv, "Inercia": km.inertia_})
                            prog_e.progress((ki + 1) / len(k_range))
                        prog_e.empty()
                        fig_e = px.line(pd.DataFrame(inertias), x="k", y="Inercia", markers=True,
                                        color_discrete_sequence=["#4f46e5"])
                        st.plotly_chart(_fig_layout(fig_e, 280), use_container_width=True,
                                        config={"displayModeBar": False})

    # ── Reducción Dimensional ─────────────────────────────────────────────────
    with tab_dim:
        feat_dim = st.multiselect("Features", num_cols_ns,
                                  default=num_cols_ns[:min(8, len(num_cols_ns))],
                                  key="dim_feats")
        if feat_dim:
            d1, d2, d3 = st.columns(3)
            with d1: dim_algo = st.selectbox("Algoritmo", ["PCA", "t-SNE"], key="dim_algo")
            with d2:
                _max_comp = min(len(feat_dim), 10) if dim_algo == "PCA" else min(3, len(feat_dim))
                n_comp = st.slider("Componentes", 2, _max_comp, 2, key="dim_comp")
            with d3: color_d  = st.selectbox("Colorear por", ["—"] + df.columns.tolist(), key="dim_color",
                                              help="Usa columnas del dataset original (sin OHE) para colorear.")

            if dim_algo == "t-SNE":
                _t1, _t2 = st.columns(2)
                with _t1:
                    tsne_perplexity = st.slider(
                        "Perplexity", 5, 100, 30, step=5, key="tsne_perp",
                        help="Baja (5-15): estructura local · Media (30): balance · Alta (50-100): estructura global",
                    )
                with _t2:
                    tsne_iterations = st.slider(
                        "Iteraciones", 250, 2000, 1000, step=250, key="tsne_iter",
                        help="Más iteraciones = resultado más estable, pero más lento",
                    )

            if st.button("Ejecutar reducción dimensional", type="primary"):
                try:
                    X_dim = df_encoded[feat_dim].dropna()
                    if dim_algo == "PCA":
                        model_d = PCA(n_components=n_comp, random_state=42)
                        kind_d  = "pca"
                    elif dim_algo == "t-SNE":
                        from sklearn.manifold import TSNE
                        model_d = TSNE(n_components=n_comp, random_state=42,
                                       perplexity=tsne_perplexity,
                                       max_iter=tsne_iterations)
                        kind_d  = "tsne"
                    else:
                        from sklearn.manifold import TSNE
                        model_d = TSNE(n_components=n_comp, random_state=42,
                                       perplexity=tsne_perplexity,
                                       max_iter=tsne_iterations)
                        kind_d  = "tsne"

                    runner_d = UnsupervisedRunner(
                        name=dim_algo, X=X_dim, model=model_d, kind=kind_d, scale_X=True
                    )
                    runner_d.fit()
                    st.session_state["unsup_dim_result"] = {
                        "embedding": runner_d.embedding_,
                        "metrics":   runner_d.metrics,
                        "algo":      dim_algo,
                        "index":     X_dim.index,
                        "model":     runner_d.model,
                    }
                except ImportError as e:
                    st.error(f"Librería no instalada: {e}")
                except Exception as e:
                    st.error(f"Error: {e}")

            dr = st.session_state.get("unsup_dim_result")
            if dr:
                emb_d = dr["embedding"]
                idx_d = dr["index"]
                _algo = dr["algo"]
                _model = dr.get("model")

                if dr["metrics"]:
                    mc = st.columns(len(dr["metrics"]), gap="medium")
                    for i, (mk, mv) in enumerate(dr["metrics"].items()):
                        with mc[i]:
                            metric_card(mk, f"{float(mv):.4f}", _algo, "#4f46e5")

                # Nombres de ejes según algoritmo + varianza explicada para PCA
                _prefixes = {"PCA": "PC", "t-SNE": "t-SNE"}
                _pfx = _prefixes.get(_algo, "Dim")
                _evr = (getattr(_model, "explained_variance_ratio_", None)
                        if _algo == "PCA" else None)
                def _axis_label(i):
                    if _evr is not None and i < len(_evr):
                        return f"{_pfx}{i+1} ({_evr[i]*100:.1f}%)"
                    return f"{_pfx}{i+1}"

                ax1, ax2 = _axis_label(0), _axis_label(1)

                # ── KMeans overlay para t-SNE / UMAP ──────────────────────
                _km_labels = None
                if _algo == "t-SNE":
                    _kov1, _kov2 = st.columns([1, 3])
                    with _kov1:
                        _use_km = st.checkbox("Colorear por KMeans", key="dim_use_km")
                    if _use_km:
                        with _kov2:
                            _km_k = st.slider("k clusters", 2, 12, 3, key="dim_km_k")
                        _km_labels = (KMeans(n_clusters=_km_k, random_state=42, n_init="auto")
                                      .fit_predict(emb_d).astype(str))

                # Color final: KMeans > columna seleccionada > default
                sdf_d = pd.DataFrame({ax1: emb_d[:,0], ax2: emb_d[:,1]})
                if _km_labels is not None:
                    sdf_d["Color"] = [f"Cluster {c}" for c in _km_labels]
                    fig_d = px.scatter(sdf_d, x=ax1, y=ax2, color="Color",
                                       color_discrete_sequence=px.colors.qualitative.Bold,
                                       opacity=0.75)
                elif color_d != "—" and color_d in df.columns:
                    sdf_d["Color"] = df.loc[idx_d, color_d].values
                    fig_d = px.scatter(sdf_d, x=ax1, y=ax2, color="Color", opacity=0.7)
                else:
                    fig_d = px.scatter(sdf_d, x=ax1, y=ax2, opacity=0.6,
                                       color_discrete_sequence=["#4f46e5"])

                panel_open(f"Proyección {_algo} 2D")
                st.plotly_chart(_fig_layout(fig_d, 450), use_container_width=True,
                                config={"displayModeBar": False})
                panel_close()

                if emb_d.shape[1] >= 3:
                    ax3 = _axis_label(2)
                    sdf_3d: dict = {ax1: emb_d[:,0], ax2: emb_d[:,1], ax3: emb_d[:,2]}
                    if _km_labels is not None:
                        sdf_3d["Color"] = [f"Cluster {c}" for c in _km_labels]
                    elif color_d != "—" and color_d in df.columns:
                        sdf_3d["Color"] = df.loc[idx_d, color_d].values
                    _has_color_3d = _km_labels is not None or color_d != "—"
                    fig_3d = px.scatter_3d(
                        pd.DataFrame(sdf_3d), x=ax1, y=ax2, z=ax3,
                        color="Color" if _has_color_3d else None,
                        color_discrete_sequence=(px.colors.qualitative.Bold
                                                 if _km_labels is not None else None),
                        opacity=0.7,
                        title=f"Proyección {_algo} 3D",
                    )
                    fig_3d.update_traces(marker=dict(size=4))
                    fig_3d.update_layout(height=550, margin=dict(l=0, r=0, t=40, b=0))
                    panel_open(f"Proyección {_algo} 3D")
                    st.plotly_chart(fig_3d, use_container_width=True)
                    panel_close()

                # ── Varianza explicada por componente (solo PCA) ───────────
                if _algo == "PCA" and _model is not None and hasattr(_model, "explained_variance_ratio_"):
                    _evr_all   = _model.explained_variance_ratio_
                    _evr_cum   = np.cumsum(_evr_all)
                    _pc_labels = [f"PC{i+1}" for i in range(len(_evr_all))]
                    _fig_var = go.Figure()
                    _fig_var.add_trace(go.Bar(
                        x=_pc_labels, y=_evr_all * 100,
                        name="Varianza individual",
                        marker_color="#4f46e5", opacity=0.85,
                        text=[f"{v*100:.1f}%" for v in _evr_all],
                        textposition="outside",
                    ))
                    _fig_var.add_trace(go.Scatter(
                        x=_pc_labels, y=_evr_cum * 100,
                        name="Varianza acumulada",
                        mode="lines+markers",
                        line=dict(color="#f59e0b", width=2),
                        marker=dict(size=7),
                        yaxis="y2",
                    ))
                    _fig_var.update_layout(
                        title="Varianza Explicada por Componente",
                        yaxis=dict(title="Varianza individual (%)", range=[0, max(_evr_all)*130]),
                        yaxis2=dict(title="Varianza acumulada (%)", overlaying="y",
                                    side="right", range=[0, 110],
                                    showgrid=False, ticksuffix="%"),
                        legend=dict(orientation="h", y=-0.2),
                        height=380, plot_bgcolor="#f9fafb",
                        margin=dict(l=40, r=60, t=50, b=40),
                    )
                    panel_open("Varianza Explicada por Componente")
                    st.plotly_chart(_fig_var, use_container_width=True,
                                    config={"displayModeBar": False})
                    panel_close()

                # ── Círculo de correlación / Biplot (solo PCA) ─────────────
                if _algo == "PCA" and _model is not None:
                    _comps  = _model.components_       # (n_comp, n_features)
                    _feats  = feat_dim
                    _scale  = np.sqrt(_model.explained_variance_)
                    _n_comp = _comps.shape[0]
                    _colors_cc = px.colors.qualitative.Plotly

                    _show_pts = st.checkbox(
                        "Mostrar observaciones en el círculo (Biplot)",
                        value=False, key="pca_biplot",
                    )
                    # Scores normalizados a [-1, 1] para convivir con los loadings
                    _scores = emb_d[:, :min(_n_comp, 3)]
                    _scores_n = _scores / (np.abs(_scores).max(axis=0) + 1e-9)

                    # Color de los puntos — separar en categorías si es string
                    _pt_color_raw = (df.loc[idx_d, color_d].values
                                     if color_d != "—" and color_d in df.columns
                                     else None)
                    _pt_is_cat = (_pt_color_raw is not None and
                                  not np.issubdtype(np.array(_pt_color_raw).dtype, np.number))

                    def _add_biplot_points(fig, xs, ys, zs=None):
                        """Añade puntos al biplot respetando tipo de color."""
                        is3d = zs is not None
                        TraceClass = go.Scatter3d if is3d else go.Scatter
                        if _pt_color_raw is None:
                            kw = dict(mode="markers",
                                      marker=dict(size=3 if is3d else 5,
                                                  color="#94a3b8", opacity=0.35),
                                      showlegend=False, hoverinfo="skip")
                            fig.add_trace(TraceClass(x=xs, y=ys,
                                                     **({} if not is3d else {"z": zs}),
                                                     **kw))
                        elif _pt_is_cat:
                            _cats = pd.Categorical(_pt_color_raw)
                            for _ci, _cat in enumerate(list(_cats.categories)):
                                _mask = _cats == _cat
                                _col  = _colors_cc[_ci % len(_colors_cc)]
                                kw = dict(mode="markers", name=str(_cat),
                                          marker=dict(size=3 if is3d else 5,
                                                      color=_col, opacity=0.35),
                                          showlegend=True,
                                          hovertemplate=f"{color_d}: {_cat}<extra></extra>")
                                fig.add_trace(TraceClass(
                                    x=xs[_mask], y=ys[_mask],
                                    **({} if not is3d else {"z": zs[_mask]}),
                                    **kw))
                        else:
                            kw = dict(mode="markers",
                                      marker=dict(size=3 if is3d else 5,
                                                  color=_pt_color_raw,
                                                  colorscale="Viridis", opacity=0.35,
                                                  showscale=True),
                                      showlegend=False,
                                      hovertemplate=f"{color_d}: %{{marker.color:.2f}}<extra></extra>")
                            fig.add_trace(TraceClass(x=xs, y=ys,
                                                     **({} if not is3d else {"z": zs}),
                                                     **kw))

                    if _n_comp >= 3:
                        # ── Versión 3D ──────────────────────────────────────
                        _load3 = (_comps[:3] * _scale[:3, None]).T  # (n_feat, 3)
                        _fig_cc = go.Figure()
                        # Puntos (biplot 3D)
                        if _show_pts:
                            _add_biplot_points(_fig_cc,
                                               _scores_n[:, 0],
                                               _scores_n[:, 1],
                                               _scores_n[:, 2])
                        # Esferas de referencia (3 círculos en planos XY, XZ, YZ)
                        _th = np.linspace(0, 2 * np.pi, 120)
                        for _xs, _ys, _zs in [
                            (np.cos(_th), np.sin(_th), np.zeros_like(_th)),
                            (np.cos(_th), np.zeros_like(_th), np.sin(_th)),
                            (np.zeros_like(_th), np.cos(_th), np.sin(_th)),
                        ]:
                            _fig_cc.add_trace(go.Scatter3d(
                                x=_xs, y=_ys, z=_zs, mode="lines",
                                line=dict(color="#d1d5db", width=1),
                                showlegend=False, hoverinfo="skip",
                            ))
                        # Flechas como cone + línea por variable
                        for _j, _fname in enumerate(_feats):
                            _cx = float(_load3[_j, 0])
                            _cy = float(_load3[_j, 1])
                            _cz = float(_load3[_j, 2])
                            _col = _colors_cc[_j % len(_colors_cc)]
                            _fig_cc.add_trace(go.Scatter3d(
                                x=[0, _cx], y=[0, _cy], z=[0, _cz],
                                mode="lines",
                                line=dict(color=_col, width=4),
                                showlegend=False,
                                hovertemplate=(
                                    f"<b>{_fname}</b><br>"
                                    f"{ax1}: %{{x:.3f}}<br>"
                                    f"{ax2}: %{{y:.3f}}<br>"
                                    f"{_axis_label(2)}: %{{z:.3f}}<extra></extra>"
                                ),
                            ))
                            _fig_cc.add_trace(go.Cone(
                                x=[_cx], y=[_cy], z=[_cz],
                                u=[_cx * 0.15], v=[_cy * 0.15], w=[_cz * 0.15],
                                colorscale=[[0, _col], [1, _col]],
                                showscale=False, showlegend=False,
                                sizemode="absolute", sizeref=0.08,
                                hoverinfo="skip",
                            ))
                            _fig_cc.add_trace(go.Scatter3d(
                                x=[_cx * 1.15], y=[_cy * 1.15], z=[_cz * 1.15],
                                mode="text", text=[_fname],
                                textfont=dict(size=11, color=_col),
                                showlegend=False, hoverinfo="skip",
                            ))
                        _title_3d = ("Biplot PCA 3D" if _show_pts
                                     else "Círculo de Correlación PCA 3D")
                        _fig_cc.update_layout(
                            title=_title_3d,
                            scene=dict(
                                xaxis=dict(title=ax1, range=[-1.3, 1.3]),
                                yaxis=dict(title=ax2, range=[-1.3, 1.3]),
                                zaxis=dict(title=_axis_label(2), range=[-1.3, 1.3]),
                                bgcolor="#f9fafb",
                            ),
                            height=580, margin=dict(l=0, r=0, t=50, b=0),
                        )
                        _panel_title = ("Biplot PCA 3D" if _show_pts
                                        else "Círculo de Correlación PCA 3D")
                        panel_open(_panel_title)
                        st.plotly_chart(_fig_cc, use_container_width=True)
                        panel_close()

                    else:
                        # ── Versión 2D ──────────────────────────────────────
                        _load2 = (_comps[:2] * _scale[:2, None]).T  # (n_feat, 2)
                        _fig_cc = go.Figure()
                        # Puntos (biplot 2D)
                        if _show_pts:
                            _add_biplot_points(_fig_cc,
                                               _scores_n[:, 0],
                                               _scores_n[:, 1])
                        # Círculo de referencia
                        _theta = np.linspace(0, 2 * np.pi, 200)
                        _fig_cc.add_trace(go.Scatter(
                            x=np.cos(_theta), y=np.sin(_theta), mode="lines",
                            line=dict(color="#d1d5db", width=1, dash="dot"),
                            showlegend=False, hoverinfo="skip",
                        ))
                        for _j, _fname in enumerate(_feats):
                            _cx, _cy = float(_load2[_j, 0]), float(_load2[_j, 1])
                            _col = _colors_cc[_j % len(_colors_cc)]
                            _fig_cc.add_annotation(
                                x=_cx, y=_cy, ax=0, ay=0,
                                xref="x", yref="y", axref="x", ayref="y",
                                showarrow=True,
                                arrowhead=3, arrowsize=1.2, arrowwidth=1.8,
                                arrowcolor=_col,
                            )
                            _fig_cc.add_trace(go.Scatter(
                                x=[_cx * 1.12], y=[_cy * 1.12], mode="text",
                                text=[_fname],
                                textfont=dict(size=11, color=_col),
                                showlegend=False, hoverinfo="skip",
                            ))
                        _title_2d = ("Biplot PCA" if _show_pts
                                     else "Círculo de Correlación (PC1 vs PC2)")
                        _fig_cc.update_layout(
                            title=_title_2d,
                            xaxis=dict(title=ax1, range=[-1.3, 1.3], zeroline=True,
                                       zerolinecolor="#9ca3af", showgrid=False),
                            yaxis=dict(title=ax2, range=[-1.3, 1.3], zeroline=True,
                                       zerolinecolor="#9ca3af", showgrid=False,
                                       scaleanchor="x", scaleratio=1),
                            height=520, plot_bgcolor="#f9fafb",
                            margin=dict(l=40, r=40, t=50, b=40),
                        )
                        _panel_title2d = ("Biplot PCA" if _show_pts
                                          else "Círculo de Correlación PCA")
                        panel_open(_panel_title2d)
                        st.plotly_chart(_fig_cc, use_container_width=True,
                                        config={"displayModeBar": False})
                        panel_close()

    # ── Reglas de Asociación ──────────────────────────────────────────────────
    with tab_assoc:
        fmt = st.radio(
            "¿Cómo están tus datos?",
            [
                "Columnas categóricas  (una fila = una transacción, ítems = col:valor)",
                "Formato largo  (una fila = un ítem, agrupar por transacción)",
                "Matriz binaria  (columnas = ítems, valores 0/1)",
                "Columna con listas  (valores separados por coma)",
            ],
            key="assoc_fmt",
        )

        a1, a2 = st.columns(2)
        with a1:
            min_sup  = st.slider("Min support",    0.01, 0.5,  0.05, 0.01, key="assoc_sup")
            min_conf = st.slider("Min confidence", 0.1,  1.0,  0.5,  0.05, key="assoc_conf")
        with a2:
            min_lift = st.slider("Lift mínimo (filtro visual)", 0.5, 5.0, 1.0, 0.1, key="assoc_lift")

        if "categóricas" in fmt:
            _cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
            _cat_cols += [c for c in df.select_dtypes(include=["int64","int32"]).columns
                          if df[c].nunique() <= 10 and c not in _cat_cols]
            cat_cols_sel = st.multiselect(
                "Columnas categóricas a usar como ítems",
                _cat_cols, default=_cat_cols[:min(5, len(_cat_cols))],
                key="assoc_cat",
                help="Cada fila será una transacción. Los ítems serán 'columna=valor'.",
            )
        elif "largo" in fmt:
            ac1, ac2 = st.columns(2)
            with ac1: tx_col   = st.selectbox("Columna transacción (ID)", df.columns.tolist(), key="assoc_tx")
            with ac2: item_col = st.selectbox("Columna ítem",             df.columns.tolist(), key="assoc_item")
        elif "binaria" in fmt:
            bin_cols = st.multiselect("Columnas que son ítems (0/1)", df.columns.tolist(), key="assoc_bin")
        else:
            lc1, lc2 = st.columns(2)
            with lc1: list_col = st.selectbox("Columna con listas", df.columns.tolist(), key="assoc_lc")
            with lc2: list_sep = st.text_input("Separador", ",", key="assoc_sep")

        if st.button("Ejecutar análisis de asociación", type="primary"):
            try:
                if "categóricas" in fmt:
                    if not cat_cols_sel:
                        st.error("Selecciona al menos 2 columnas.")
                        st.stop()
                    transactions = df[cat_cols_sel].dropna().apply(
                        lambda row: [f"{col}={row[col]}" for col in cat_cols_sel],
                        axis=1,
                    ).tolist()
                    explorer = AssociationRulesExplorer(transactions)
                elif "largo" in fmt:
                    explorer = AssociationRulesExplorer.from_transaction_df(
                        df, transaction_col=tx_col, item_col=item_col
                    )
                elif "binaria" in fmt:
                    if not bin_cols:
                        st.error("Selecciona las columnas de ítems.")
                        st.stop()
                    transactions = df[bin_cols].apply(
                        lambda row: [c for c in bin_cols if row[c]], axis=1
                    ).tolist()
                    explorer = AssociationRulesExplorer(transactions)
                else:
                    transactions = [
                        [v.strip() for v in str(cell).split(list_sep) if v.strip()]
                        for cell in df[list_col].dropna()
                    ]
                    explorer = AssociationRulesExplorer(transactions)

                with st.spinner("Calculando reglas..."):
                    explorer.fit_itemsets(min_support=min_sup)
                    rules = explorer.fit_rules(min_support=min_sup, min_threshold=min_conf)
                    if not rules.empty:
                        rules = rules.copy()
                        rules["antecedents"] = rules["antecedents"].apply(lambda x: ", ".join(sorted(x)))
                        rules["consequents"] = rules["consequents"].apply(lambda x: ", ".join(sorted(x)))

                st.session_state["assoc_rules"]    = rules
                st.session_state["assoc_itemsets"] = explorer.itemsets_
            except Exception as e:
                st.error(f"Error: {e}")

        rules_df   = st.session_state.get("assoc_rules")
        itemsets_d = st.session_state.get("assoc_itemsets")

        if rules_df is not None:
            if rules_df.empty:
                st.warning("Sin reglas con estos umbrales. Intenta reducir min_support o min_confidence.")
            else:
                rules_f = rules_df[rules_df["lift"] >= min_lift]
                r1, r2, r3 = st.columns(3, gap="medium")
                with r1: metric_card("Reglas encontradas", str(len(rules_f)), f"lift ≥ {min_lift}", "#4f46e5")
                with r2: metric_card("Itemsets frecuentes", str(len(itemsets_d)) if itemsets_d is not None else "—",
                                     f"support ≥ {min_sup}", "#7c3aed")
                with r3: metric_card("Max lift", f"{rules_f['lift'].max():.3f}" if not rules_f.empty else "—",
                                     "reglas filtradas", "#10b981")

                panel_open("Reglas de asociación", f"{len(rules_f)} reglas — ordenadas por lift ↓")
                disp = [c for c in ["antecedents","consequents","support","confidence","lift"]
                        if c in rules_f.columns]
                st.dataframe(rules_f[disp].sort_values("lift", ascending=False).round(4),
                             use_container_width=True, hide_index=True)
                st.download_button("Descargar CSV", rules_f.to_csv(index=False).encode(),
                                   "reglas.csv", "text/csv")
                panel_close()

                if not rules_f.empty:
                    panel_open("Support vs Confidence  (tamaño = lift)")
                    fig_r = px.scatter(rules_f, x="support", y="confidence", size="lift",
                                       hover_data=["antecedents","consequents","lift"],
                                       color="lift", color_continuous_scale=["#60a5fa","#7c3aed"])
                    st.plotly_chart(_fig_layout(fig_r, 380), use_container_width=True,
                                    config={"displayModeBar": False})
                    panel_close()

                if itemsets_d is not None and not itemsets_d.empty:
                    with st.expander("Top ítems frecuentes", expanded=False):
                        try:
                            singles = itemsets_d[itemsets_d["n_items"] == 1]
                            items_list = [next(iter(iset)) for iset in singles["itemsets"]]
                            items_vc = (pd.Series(items_list).value_counts().head(15)
                                        .reset_index())
                            items_vc.columns = ["Ítem", "Frecuencia"]
                            fig_it = px.bar(items_vc.sort_values("Frecuencia"),
                                            x="Frecuencia", y="Ítem", orientation="h",
                                            color="Frecuencia",
                                            color_continuous_scale=["#60a5fa","#7c3aed"])
                            fig_it.update_layout(coloraxis_showscale=False)
                            st.plotly_chart(_fig_layout(fig_it, 380), use_container_width=True,
                                            config={"displayModeBar": False})
                        except Exception:
                            st.dataframe(itemsets_d.head(20), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# 5) WEB SCRAPING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Series de Tiempo":
    if not _require_df():
        st.stop()

    df_raw_ts = st.session_state["df"]
    hero("Series de Tiempo", f"Análisis y pronóstico temporal — {st.session_state['df_name']}")

    # ── Detección de columnas de fecha ──
    def _ts_datetime_candidates(df: pd.DataFrame):
        candidates = []
        for col in df.columns:
            if "datetime" in str(df[col].dtype):
                candidates.append(col)
            elif df[col].dtype == object:
                try:
                    pd.to_datetime(df[col].dropna().head(30))
                    candidates.append(col)
                except Exception:
                    pass
        return candidates or df.columns.tolist()[:1]

    def _ts_prepare(df: pd.DataFrame, date_col: str, target_col: str) -> pd.DataFrame:
        out = df[[date_col, target_col]].copy()
        out[date_col] = pd.to_datetime(out[date_col])
        out = out.dropna(subset=[date_col, target_col]).sort_values(date_col).reset_index(drop=True)
        return out

    def _ts_frequency_hint(date_series: pd.Series) -> str:
        diffs = date_series.sort_values().diff().dropna()
        if diffs.empty:
            return "No determinada"
        mode = diffs.mode().iloc[0]
        if mode <= pd.Timedelta(hours=1): return "Horaria"
        if mode <= pd.Timedelta(days=1): return "Diaria"
        if mode <= pd.Timedelta(days=7): return "Semanal"
        if mode <= pd.Timedelta(days=31): return "Mensual"
        return str(mode)

    def _ts_run_analysis(df, date_col, target_col, model_name, ranking_metric, lags, n_splits, train_pct, forecast_steps):
        warnings.filterwarnings("ignore")
        data = _ts_prepare(df, date_col, target_col)
        if len(data) <= lags + max(5, n_splits):
            raise ValueError("No hay suficientes registros para esa cantidad de lags y folds.")

        model_df = data[[target_col]].copy()
        train_size = train_pct / 100.0

        model_map = {
            "DeepLearning (Red Neuronal)": MLPRegressor(
                hidden_layer_sizes=(64, 32), activation="relu", solver="adam", max_iter=500, random_state=42
            ),
            "HoltWinters": HoltWintersForecaster(seasonal_periods=7),
            "HoltWinters-Calibrado": HoltWintersForecaster(seasonal_periods=12),
            "Arima": ARIMAForecaster(order=(1, 1, 1)),
            "Arima-Calibrado": ARIMAForecaster(order=(2, 1, 2)),
        }
        candidates = list(model_map.keys()) if model_name == "Mejor modelo automático" else [model_name]
        sort_col = {"Holdout RMSE": "RMSE", "CV RMSE": "CV_RMSE"}.get(ranking_metric, "RMSE")

        test_size = max(1, len(model_df) - int(round(len(model_df) * train_size)))
        cv_test_size = max(1, min(test_size, (len(model_df) - lags) // (n_splits + 1)))

        rows, detailed = [], {}
        for name in candidates:
            model = model_map[name]
            preparer = DataPreparer(train_size=train_size, random_state=42)
            runner = TimeSeriesRunner(
                df=model_df, target=target_col, model=model,
                lags=lags, preparer=preparer, test_size=test_size,
            )
            holdout = runner.evaluate()
            cv = runner.evaluate_cv(n_splits=n_splits, test_size=cv_test_size)
            runner.fit_full()
            future = runner.forecast(steps=forecast_steps)
            pred = runner.fit_predict()
            y_test = runner.y_test
            pred_index = runner.test_index

            rows.append({
                "Algoritmo": name,
                "MAE": float(holdout.get("MAE", np.nan)),
                "RMSE": float(holdout.get("RMSE", np.nan)),
                "MAPE_%": float(holdout.get("MAPE_%", np.nan)),
                "R2": float(holdout.get("R2", np.nan)),
                "CV_MAE": float(cv.get("MAE", np.nan)),
                "CV_RMSE": float(cv.get("RMSE", np.nan)),
                "CV_MAPE_%": float(cv.get("MAPE_%", np.nan)),
                "CV_R2": float(cv.get("R2", np.nan)),
            })

            pred_df_item = pd.DataFrame({
                "date": data.loc[pred_index, date_col].values,
                "actual": np.asarray(y_test),
                "predicted": np.asarray(pred),
            })
            future_dates = pd.date_range(
                data[date_col].max() + pd.Timedelta(days=1), periods=forecast_steps, freq="D"
            )
            future_df_item = pd.DataFrame({"date": future_dates, "forecast": np.asarray(future)})
            detailed[name] = {"holdout": holdout, "cv": cv, "pred_df": pred_df_item, "future_df": future_df_item}

        results_df_ts = pd.DataFrame(rows).sort_values(sort_col, ascending=True).reset_index(drop=True)
        best_name_ts = results_df_ts.iloc[0]["Algoritmo"]
        return {
            "data": data, "results_df": results_df_ts, "best_name": best_name_ts,
            "best_detail": detailed[best_name_ts], "detailed": detailed,
            "ranking_metric_label": ranking_metric, "ranking_metric_column": sort_col,
            "train_pct": train_pct, "test_pct": 100 - train_pct,
            "lags": lags, "n_splits": n_splits, "forecast_steps": forecast_steps,
            "date_col": date_col, "target_col": target_col,
        }

    # ── Validación de columnas disponibles ──
    num_cols_ts = df_raw_ts.select_dtypes(include=np.number).columns.tolist()
    date_candidates_ts = _ts_datetime_candidates(df_raw_ts)

    if not date_candidates_ts or not num_cols_ts:
        st.warning("El dataset necesita al menos una columna de fecha y una columna numérica.")
        st.stop()

    # ── Configuración ──
    with st.expander("⚙️ Configuración del análisis", expanded=True):
        cfg1, cfg2, cfg3 = st.columns(3)
        with cfg1:
            ts_date_col = st.selectbox("Columna de fecha", date_candidates_ts, key="ts_date_col")
            ts_target_col = st.selectbox("Variable objetivo", num_cols_ts, key="ts_target_col")
        with cfg2:
            ts_model_name = st.selectbox("Algoritmo", [
                "Mejor modelo automático", "DeepLearning (Red Neuronal)",
                "HoltWinters", "HoltWinters-Calibrado", "Arima", "Arima-Calibrado",
            ], key="ts_model")
            ts_ranking = st.selectbox("Criterio de selección", ["Holdout RMSE", "CV RMSE"], key="ts_ranking")
        with cfg3:
            ts_lags = st.slider("Lags", 3, 30, 7, key="ts_lags")
            ts_n_splits = st.slider("K-Folds temporales", 2, 10, 5, key="ts_splits")
            ts_train_pct = st.slider("% Entrenamiento", 60, 90, 80, step=5, key="ts_train_pct")
            ts_forecast = st.slider("Horizonte de predicción", 3, 30, 7, key="ts_forecast")
        run_ts = st.button("▶ Ejecutar análisis", type="primary", key="ts_run")

    if run_ts:
        with st.spinner("Ejecutando análisis de series de tiempo..."):
            try:
                st.session_state["ts_analysis"] = _ts_run_analysis(
                    df_raw_ts, ts_date_col, ts_target_col, ts_model_name,
                    ts_ranking, ts_lags, ts_n_splits, ts_train_pct, ts_forecast,
                )
            except Exception as e:
                st.error(f"Error en el análisis: {e}")

    analysis_ts = st.session_state.get("ts_analysis")
    if analysis_ts is None:
        st.info("Configura los parámetros y presiona **▶ Ejecutar análisis** para comenzar.")
        st.stop()

    ts_data = analysis_ts["data"]
    ts_results_df = analysis_ts["results_df"]
    ts_best_name = analysis_ts["best_name"]
    ts_best_detail = analysis_ts["best_detail"]
    ts_date_out = analysis_ts["date_col"]
    ts_target_out = analysis_ts["target_col"]

    tab_eda_ts, tab_models_ts, tab_pred_ts = st.tabs(["EDA", "Modelos", "Predicciones"])

    # ── Tab EDA ──
    with tab_eda_ts:
        null_vals_ts = int(ts_data[ts_target_out].isna().sum())
        full_range_ts = pd.date_range(ts_data[ts_date_out].min(), ts_data[ts_date_out].max(), freq="D")
        missing_dates_ts = int(len(full_range_ts.difference(pd.DatetimeIndex(ts_data[ts_date_out]))))

        c1, c2, c3, c4 = st.columns(4, gap="medium")
        with c1: metric_card("Registros", f"{len(ts_data):,}", "filas válidas", "#4f46e5")
        with c2: metric_card("Valores nulos", str(null_vals_ts), f"{null_vals_ts/max(len(ts_data),1)*100:.2f}%", "#f59e0b")
        with c3: metric_card("Fechas faltantes", str(missing_dates_ts), "vs frecuencia diaria", "#7c3aed")
        with c4: metric_card("Frecuencia", _ts_frequency_hint(ts_data[ts_date_out]), "estimada", "#2563eb")

        left, right = st.columns(2)
        with left:
            panel_open("Serie temporal completa")
            fig_line_ts = viz.line_chart(ts_data, x=ts_date_out, y=ts_target_out, height=360)
            st.plotly_chart(fig_line_ts, use_container_width=True)
            panel_close()
        with right:
            panel_open("Distribución de la variable objetivo")
            hist_vals_ts, edges_ts = np.histogram(ts_data[ts_target_out].dropna(), bins=8)
            labels_ts = [f"{edges_ts[i]:.0f}-{edges_ts[i+1]:.0f}" for i in range(len(edges_ts) - 1)]
            dist_df_ts = pd.DataFrame({"Rango": labels_ts, "Frecuencia": hist_vals_ts})
            fig_hist_ts = viz.grouped_bar_chart(
                dist_df_ts, x="Rango", y="Frecuencia", color="Rango", height=360, showlegend=False
            )
            st.plotly_chart(fig_hist_ts, use_container_width=True)
            panel_close()

        desc_ts = ts_data[ts_target_out].describe()
        panel_open("Estadísticas descriptivas")
        s1, s2, s3, s4, s5 = st.columns(5)
        for col, (lbl, val) in zip([s1, s2, s3, s4, s5], [
            ("Media", f"{desc_ts['mean']:.2f}"),
            ("Mediana", f"{ts_data[ts_target_out].median():.2f}"),
            ("Desv. Est.", f"{desc_ts['std']:.2f}"),
            ("Mínimo", f"{desc_ts['min']:.2f}"),
            ("Máximo", f"{desc_ts['max']:.2f}"),
        ]):
            with col: metric_card(lbl, val, "")
        panel_close()

    # ── Tab Modelos ──
    with tab_models_ts:
        best_row_ts = ts_results_df.iloc[0]
        sort_col_ts = analysis_ts["ranking_metric_column"]

        c1, c2, c3, c4 = st.columns(4, gap="medium")
        with c1: metric_card("Mejor modelo", ts_best_name, f"por {analysis_ts['ranking_metric_label']}", "#4f46e5")
        with c2: metric_card(analysis_ts["ranking_metric_label"], f"{best_row_ts[sort_col_ts]:.3f}", f"K-Folds: {analysis_ts['n_splits']}", "#2563eb")
        with c3:
            mae_v = best_row_ts["CV_MAE"] if sort_col_ts == "CV_RMSE" else best_row_ts["MAE"]
            metric_card("MAE", f"{mae_v:.3f}", f"Lags: {analysis_ts['lags']}", "#10b981")
        with c4:
            mape_v = best_row_ts["CV_MAPE_%"] if sort_col_ts == "CV_RMSE" else best_row_ts["MAPE_%"]
            metric_card("MAPE", f"{mape_v:.2f}%", f"Train/Test: {analysis_ts['train_pct']}/{analysis_ts['test_pct']}", "#7c3aed")

        left, right = st.columns(2)
        with left:
            panel_open("Comparación de métricas por modelo")
            metric_cols_ts = ["CV_RMSE", "CV_MAE", "CV_MAPE_%"] if sort_col_ts == "CV_RMSE" else ["RMSE", "MAE", "MAPE_%"]
            metrics_long_ts = ts_results_df.melt(id_vars="Algoritmo", value_vars=metric_cols_ts, var_name="Métrica", value_name="Valor")
            fig_bar_ts = viz.grouped_bar_chart(
                metrics_long_ts, x="Algoritmo", y="Valor", color="Métrica", barmode="group", height=380
            )
            st.plotly_chart(fig_bar_ts, use_container_width=True)
            panel_close()
        with right:
            panel_open("Real vs predicción (holdout)")
            pred_df_ts = ts_best_detail["pred_df"]
            fig_pred_ts = viz.multi_line_chart([
                {"x": pred_df_ts["date"], "y": pred_df_ts["actual"], "mode": "lines+markers", "name": "Actual"},
                {"x": pred_df_ts["date"], "y": pred_df_ts["predicted"], "mode": "lines+markers",
                 "name": "Predicho", "line": dict(dash="dash")},
            ], height=380)
            st.plotly_chart(fig_pred_ts, use_container_width=True)
            panel_close()

        panel_open("Ranking de modelos")
        disp_ts = ts_results_df.copy()
        for c in ["MAE", "RMSE", "MAPE_%", "R2", "CV_MAE", "CV_RMSE", "CV_MAPE_%", "CV_R2"]:
            disp_ts[c] = disp_ts[c].map(lambda x: round(float(x), 4) if pd.notna(x) else x)
        st.dataframe(disp_ts, use_container_width=True, hide_index=True)
        panel_close()

    # ── Tab Predicciones ──
    with tab_pred_ts:
        future_df_ts = ts_best_detail["future_df"]
        trend_ts = future_df_ts["forecast"].iloc[-1] - future_df_ts["forecast"].iloc[0]
        trend_color = "#10b981" if trend_ts >= 0 else "#ef4444"

        c1, c2, c3, c4 = st.columns(4, gap="medium")
        with c1: metric_card("Modelo", ts_best_name, "seleccionado", "#4f46e5")
        with c2: metric_card("Horizonte", f"{analysis_ts['forecast_steps']} pasos", "días proyectados", "#2563eb")
        with c3: metric_card("Promedio proyectado", f"{future_df_ts['forecast'].mean():.2f}", "media del forecast", "#10b981")
        with c4: metric_card("Tendencia", f"{trend_ts:+.2f}", "último - primer valor", trend_color)

        hist_tail_ts = ts_data.tail(30).rename(columns={ts_date_out: "date", ts_target_out: "actual"})[["date", "actual"]]
        hist_tail_ts["forecast"] = np.nan
        fut_plot_ts = future_df_ts.copy()
        fut_plot_ts["actual"] = np.nan
        chart_df_ts = pd.concat([hist_tail_ts, fut_plot_ts], ignore_index=True)

        panel_open("Serie histórica y pronóstico")
        fig_fc_ts = viz.multi_line_chart([
            {"x": chart_df_ts["date"], "y": chart_df_ts["actual"], "mode": "lines+markers", "name": "Histórico"},
            {"x": chart_df_ts["date"], "y": chart_df_ts["forecast"], "mode": "lines+markers",
             "name": "Forecast", "line": dict(dash="dash")},
        ], height=420)
        st.plotly_chart(fig_fc_ts, use_container_width=True)
        panel_close()

        panel_open("Predicciones detalladas")
        pred_tbl_ts = future_df_ts.copy()
        pred_tbl_ts["date"] = pred_tbl_ts["date"].dt.strftime("%Y-%m-%d")
        pred_tbl_ts["forecast"] = pred_tbl_ts["forecast"].round(3)
        st.dataframe(pred_tbl_ts, use_container_width=True, hide_index=True)
        st.download_button(
            "Descargar forecast CSV",
            pred_tbl_ts.to_csv(index=False).encode(),
            "forecast.csv", "text/csv", key="ts_dl_fc",
        )
        panel_close()
