import sys
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
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve,
    mean_absolute_error, mean_squared_error, r2_score,
)

from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

from ml_toolkit import (
    DataPreparer, SupervisedRunner, NeuralNetworkRunner,
    UnsupervisedRunner, AssociationRulesExplorer, WebMiningToolkit,
    get_positive_score,
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
    "sup_results": None, "sup_best_payload": None,
    "sup_target": None, "sup_task": "classification",
    "sup_features": [], "sup_balance": "none",
    "sup_train_size": 0.75, "sup_rs": 42,
    "sup_use_cv": True, "sup_n_splits": 5, "sup_pos_label": "1",
    "nn_result": None, "tuning_result": None, "stability_df": None,
    "unsup_cluster_result": None, "unsup_dim_result": None,
    "assoc_rules": None, "assoc_itemsets": None,
    "ws_texts": None, "ws_links": None, "ws_table": None,
    "ws_records": None, "ws_regex_texts": [],
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

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

def _coerce_pos_label(s: str):
    try:
        return int(s)
    except Exception:
        try:
            return float(s)
        except Exception:
            return s

def _avg_strategy(y_true) -> str:
    return "binary" if len(np.unique(y_true)) == 2 else "macro"

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔍 Explorador ML")
    st.caption("Herramienta genérica de análisis de datos")
    page = st.radio(
        "Sección",
        ["Datos", "EDA", "Supervisado", "No Supervisado", "Web Scraping"],
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
                "df": df, "df_name": uploaded.name,
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
        with c1: metric_card("Filas", f"{df.shape[0]:,}", "registros", "#4f46e5")
        with c2: metric_card("Columnas", str(df.shape[1]), "variables", "#2563eb")
        with c3: metric_card("Valores nulos", str(int(df.isnull().sum().sum())), "total en el dataset", "#f59e0b")
        with c4: metric_card("Duplicados", str(int(df.duplicated().sum())), "filas repetidas", "#ef4444")

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
    with c1: metric_card("Numéricas", str(len(num_cols)), "columnas numéricas", "#4f46e5")
    with c2: metric_card("Categóricas", str(len(cat_cols)), "columnas no numéricas", "#7c3aed")
    with c3: metric_card("Valores nulos", str(int(df.isnull().sum().sum())), "total", "#f59e0b")
    with c4: metric_card("Duplicados", str(int(df.duplicated().sum())), "filas repetidas", "#ef4444")

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
        features_sel = st.multiselect("Features (vacío = todas)", feat_opts)

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
                        df=df, target=target, model=model, task=task,
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
                        df=df, target=target, model=model, task=task,
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
                    df=df, target=target, task=task, features=features,
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

            panel_open("Métricas holdout")
            st.dataframe(pd.DataFrame([{"Métrica": k, "Valor": round(float(v), 4)}
                                        for k, v in nn_res["holdout"].items()]),
                         use_container_width=True, hide_index=True)
            panel_close()

            if nn_res["cv"]:
                panel_open("Métricas CV  (mean ± std)")
                main_keys = [k for k in nn_res["cv"] if not k.endswith("_std")]
                cv_rows = [{"Métrica": k,
                             "Mean":    round(float(nn_res["cv"][k]), 4),
                             "Std":     round(float(nn_res["cv"].get(k + "_std", 0)), 4)}
                           for k in main_keys]
                st.dataframe(pd.DataFrame(cv_rows), use_container_width=True, hide_index=True)
                panel_close()

    # ── Tab: Tuning ───────────────────────────────────────────────────────────
    with tab_tuning:
        if not st.session_state.get("sup_results"):
            st.info("Ejecuta primero la comparación de modelos para seleccionar el mejor.")
        else:
            best_name_t = pd.DataFrame(st.session_state["sup_results"]).iloc[0]["Modelo"]
            st.markdown(f"**Modelo a optimizar:** {best_name_t}")

            with st.form("tuning_form"):
                t1, t2 = st.columns(2)
                with t1:
                    t_cv_n = st.slider("Folds CV búsqueda", 3, 10, 5)
                    score_opts = (["f1", "roc_auc", "accuracy"] if task == "classification"
                                  else ["neg_mean_squared_error", "r2", "neg_mean_absolute_error"])
                    t_scoring = st.selectbox("Scoring", score_opts)
                with t2:
                    st.caption("Grid: un parámetro por línea  →  nombre: val1, val2, val3")
                    grid_text = st.text_area("Parámetros", height=130,
                                             placeholder="n_estimators: 50, 100, 200\nmax_depth: 3, 5, 7")
                run_tuning = st.form_submit_button("Ejecutar Grid Search", use_container_width=True)

            if run_tuning:
                param_grid: dict = {}
                try:
                    for line in grid_text.strip().splitlines():
                        if ":" not in line:
                            continue
                        pname, vstr = line.split(":", 1)
                        parsed = []
                        for v in vstr.split(","):
                            v = v.strip()
                            if not v:
                                continue
                            try:
                                parsed.append(int(v))
                            except ValueError:
                                try:
                                    parsed.append(float(v))
                                except ValueError:
                                    parsed.append(None if v == "None" else v)
                        if parsed:
                            param_grid[pname.strip()] = parsed
                except Exception as e:
                    st.error(f"Error al parsear el grid: {e}")

                if param_grid:
                    try:
                        bcfg = _balance_cfg(balance)
                        model_t = (_clf_model(best_name_t, rs) if task == "classification"
                                   else _reg_model(best_name_t, rs))
                        prep_t  = DataPreparer(train_size=train_size, random_state=rs,
                                               scale_X=(best_name_t in _SCALE_MODELS))
                        runner_t = SupervisedRunner(
                            df=df, target=target, model=model_t, task=task,
                            features=features, preparer=prep_t, pos_label=pos_label,
                            class_weight=bcfg["class_weight"] if task == "classification" else None,
                            sampling_method=bcfg["sampling_method"] if task == "classification" else None,
                        )
                        with st.spinner("Buscando hiperparámetros..."):
                            evaluator = runner_t.build_evaluator(scoring=t_scoring, cv=t_cv_n)
                            search_input = {best_name_t: {
                                "estimator": runner_t.get_model_for_current_split(),
                                "param_grid": param_grid,
                            }}
                            results_t = evaluator.exhaustive_search(search_input)

                        res_t  = results_t[best_name_t]
                        y_pt   = res_t["estimator"].predict(evaluator.X_test)
                        tuned_metrics: dict = {}
                        if task == "classification":
                            avg_t = _avg_strategy(evaluator.y_test)
                            tuned_metrics = {
                                "Accuracy":  float(accuracy_score(evaluator.y_test, y_pt)),
                                "F1":        float(f1_score(evaluator.y_test, y_pt, pos_label=pos_label,
                                                            zero_division=0, average=avg_t)),
                            }
                            y_sc_t = get_positive_score(res_t["estimator"], evaluator.X_test)
                            if y_sc_t is not None and len(np.unique(evaluator.y_test)) == 2:
                                try:
                                    tuned_metrics["ROC_AUC"] = float(roc_auc_score(evaluator.y_test, y_sc_t))
                                except Exception:
                                    pass
                        else:
                            tuned_metrics = {
                                "MAE":  float(mean_absolute_error(evaluator.y_test, y_pt)),
                                "RMSE": float(np.sqrt(mean_squared_error(evaluator.y_test, y_pt))),
                                "R2":   float(r2_score(evaluator.y_test, y_pt)),
                            }
                        st.session_state["tuning_result"] = {
                            "best_params": res_t["best_params"],
                            "best_score":  res_t["best_score"],
                            "metrics":     tuned_metrics,
                            "scoring":     t_scoring,
                        }
                    except Exception as e:
                        st.error(f"Error en tuning: {e}")
                else:
                    st.warning("Define al menos un parámetro en el grid.")

            tuning_res = st.session_state.get("tuning_result")
            if tuning_res:
                _colors = ["#2563eb", "#7c3aed", "#10b981", "#f59e0b", "#06b6d4"]
                tcols = st.columns(len(tuning_res["metrics"]) + 1, gap="medium")
                with tcols[0]:
                    metric_card("Best CV Score", f"{tuning_res['best_score']:.4f}",
                                tuning_res["scoring"], "#4f46e5")
                for i, (mk, mv) in enumerate(tuning_res["metrics"].items()):
                    with tcols[i + 1]:
                        metric_card(mk, f"{mv:.4f}", "test", _colors[(i + 1) % len(_colors)])

                panel_open("Mejores parámetros")
                st.dataframe(pd.DataFrame([{"Parámetro": k, "Valor": str(v)}
                                            for k, v in tuning_res["best_params"].items()]),
                             use_container_width=True, hide_index=True)
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
                            df=df, target=target, model=model_s, task=task,
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

    num_cols_ns = df.select_dtypes(include=np.number).columns.tolist()

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
            color_cl = st.selectbox("Colorear scatter por", ["cluster"] + df.columns.tolist(), key="cl_color")

            if st.button("Ejecutar clustering", type="primary"):
                try:
                    X_cl = df[feat_cl].dropna()
                    model_cl = (KMeans(n_clusters=k, random_state=42, n_init="auto")
                                if cl_algo == "KMeans"
                                else AgglomerativeClustering(n_clusters=k, linkage=linkage))
                    kind_cl = "kmeans" if cl_algo == "KMeans" else "hac"
                    runner_cl = UnsupervisedRunner(
                        name=cl_algo, X=X_cl, model=model_cl, kind=kind_cl, scale_X=True
                    )
                    runner_cl.fit()
                    emb_cl = runner_cl.ensure_2d_embedding()
                    st.session_state["unsup_cluster_result"] = {
                        "embedding": emb_cl, "labels": runner_cl.labels_,
                        "metrics": runner_cl.metrics, "algo": cl_algo,
                        "index": X_cl.index,
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

                emb = cr["embedding"]
                sdf = pd.DataFrame({"PC1": emb[:,0], "PC2": emb[:,1],
                                    "Cluster": cr["labels"].astype(str)})
                if color_cl != "cluster" and color_cl in df.columns:
                    sdf["Color"] = df.loc[cr["index"], color_cl].values
                    fig_cl = px.scatter(sdf, x="PC1", y="PC2", color="Color",
                                        hover_data=["Cluster"], opacity=0.7)
                else:
                    fig_cl = px.scatter(sdf, x="PC1", y="PC2", color="Cluster", opacity=0.7)
                panel_open("Scatter 2D (PCA sobre espacio escalado)")
                st.plotly_chart(_fig_layout(fig_cl, 420), use_container_width=True,
                                config={"displayModeBar": False})
                panel_close()

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
            with d1: dim_algo = st.selectbox("Algoritmo", ["PCA", "t-SNE", "UMAP"], key="dim_algo")
            with d2: n_comp   = st.slider("Componentes", 2, min(3, len(feat_dim)), 2, key="dim_comp")
            with d3: color_d  = st.selectbox("Colorear por", ["—"] + df.columns.tolist(), key="dim_color")

            if st.button("Ejecutar reducción dimensional", type="primary"):
                try:
                    X_dim = df[feat_dim].dropna()
                    if dim_algo == "PCA":
                        model_d = PCA(n_components=n_comp, random_state=42)
                        kind_d  = "pca"
                    elif dim_algo == "t-SNE":
                        from sklearn.manifold import TSNE
                        model_d = TSNE(n_components=n_comp, random_state=42)
                        kind_d  = "tsne"
                    else:
                        import umap
                        model_d = umap.UMAP(n_components=n_comp, random_state=42)
                        kind_d  = "umap"

                    runner_d = UnsupervisedRunner(
                        name=dim_algo, X=X_dim, model=model_d, kind=kind_d, scale_X=True
                    )
                    runner_d.fit()
                    st.session_state["unsup_dim_result"] = {
                        "embedding": runner_d.embedding_,
                        "metrics":   runner_d.metrics,
                        "algo":      dim_algo,
                        "index":     X_dim.index,
                    }
                except ImportError as e:
                    st.error(f"Librería no instalada: {e}")
                except Exception as e:
                    st.error(f"Error: {e}")

            dr = st.session_state.get("unsup_dim_result")
            if dr:
                emb_d = dr["embedding"]
                idx_d = dr["index"]

                if dr["metrics"]:
                    mc = st.columns(len(dr["metrics"]), gap="medium")
                    for i, (mk, mv) in enumerate(dr["metrics"].items()):
                        with mc[i]:
                            metric_card(mk, f"{float(mv):.4f}", dr["algo"], "#4f46e5")

                sdf_d = pd.DataFrame({"Dim1": emb_d[:,0], "Dim2": emb_d[:,1]})
                if color_d != "—" and color_d in df.columns:
                    sdf_d["Color"] = df.loc[idx_d, color_d].values
                    fig_d = px.scatter(sdf_d, x="Dim1", y="Dim2", color="Color", opacity=0.7)
                else:
                    fig_d = px.scatter(sdf_d, x="Dim1", y="Dim2", opacity=0.6,
                                       color_discrete_sequence=["#4f46e5"])

                panel_open(f"Proyección {dr['algo']} 2D")
                st.plotly_chart(_fig_layout(fig_d, 450), use_container_width=True,
                                config={"displayModeBar": False})
                panel_close()

                if emb_d.shape[1] >= 3:
                    sdf_3d: dict = {"D1": emb_d[:,0], "D2": emb_d[:,1], "D3": emb_d[:,2]}
                    if color_d != "—" and color_d in df.columns:
                        sdf_3d["Color"] = df.loc[idx_d, color_d].values
                    fig_3d = px.scatter_3d(pd.DataFrame(sdf_3d), x="D1", y="D2", z="D3",
                                           color="Color" if color_d != "—" else None, opacity=0.6)
                    fig_3d.update_layout(height=500, margin=dict(l=0, r=0, t=20, b=0))
                    st.plotly_chart(fig_3d, use_container_width=True)

    # ── Reglas de Asociación ──────────────────────────────────────────────────
    with tab_assoc:
        fmt = st.radio(
            "¿Cómo están tus datos?",
            [
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

        if "largo" in fmt:
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
                if "largo" in fmt:
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
elif page == "Web Scraping":
    hero("Web Scraping", "Extrae datos de páginas web con WebMiningToolkit.")

    url_input = st.text_input("URL a analizar", placeholder="https://ejemplo.com", key="ws_url")

    tab_txt, tab_links, tab_table, tab_rec, tab_regex = st.tabs(
        ["Texto", "Links", "Tabla HTML", "Registros", "Regex"]
    )

    # ── Texto ─────────────────────────────────────────────────────────────────
    with tab_txt:
        tw1, tw2 = st.columns([3, 1])
        with tw1: css_sel   = st.text_input("Selector CSS", placeholder="p, h1, .title", key="ws_css")
        with tw2: lim_txt   = st.number_input("Límite", 1, 500, 50, key="ws_lim")

        if st.button("Extraer texto", type="primary", key="ws_fetch_txt"):
            if not url_input:
                st.error("Ingresa una URL.")
            else:
                try:
                    wmt = WebMiningToolkit()
                    with st.spinner("Descargando página..."):
                        wmt.fetch(url_input)
                        texts = wmt.extract_text(
                            css_selector=css_sel.strip() or None,
                            tag="p" if not css_sel.strip() else None,
                            limit=int(lim_txt),
                        )
                    st.session_state["ws_texts"] = texts
                    st.success(f"{len(texts)} elementos extraídos.")
                except Exception as e:
                    st.error(f"Error: {e}")

        ws_texts = st.session_state.get("ws_texts")
        if ws_texts is not None:
            for i, t in enumerate(ws_texts[:40]):
                if t.strip():
                    st.markdown(f"**[{i+1}]** {t[:300]}")
            txt_df = pd.DataFrame({"texto": ws_texts})
            st.download_button("Descargar CSV", txt_df.to_csv(index=False).encode(),
                               "textos.csv", "text/csv", key="ws_txt_dl")
            if st.button("Usar como dataset activo", key="ws_txt_use"):
                st.session_state["df"]      = txt_df
                st.session_state["df_name"] = "texto_scrapeado.csv"
                st.success("Dataset actualizado con el texto extraído.")

    # ── Links ─────────────────────────────────────────────────────────────────
    with tab_links:
        href_filt = st.text_input("Filtrar hrefs que contengan", key="ws_href_filt")
        if st.button("Extraer links", type="primary", key="ws_fetch_links"):
            if not url_input:
                st.error("Ingresa una URL.")
            else:
                try:
                    wmt = WebMiningToolkit()
                    with st.spinner("Descargando..."):
                        wmt.fetch(url_input)
                        links = wmt.extract_links(href_contains=href_filt.strip() or None)
                    st.session_state["ws_links"] = links
                    st.success(f"{len(links)} links encontrados.")
                except Exception as e:
                    st.error(f"Error: {e}")

        ws_links = st.session_state.get("ws_links")
        if ws_links:
            ldf = pd.DataFrame({"href": ws_links})
            st.dataframe(ldf, use_container_width=True)
            st.download_button("Descargar CSV", ldf.to_csv(index=False).encode(),
                               "links.csv", "text/csv", key="ws_lnk_dl")

    # ── Tabla HTML ────────────────────────────────────────────────────────────
    with tab_table:
        t_idx = st.number_input("Índice de tabla (0 = primera)", 0, 20, 0, key="ws_tidx")
        if st.button("Extraer tabla HTML", type="primary", key="ws_fetch_tbl"):
            if not url_input:
                st.error("Ingresa una URL.")
            else:
                try:
                    wmt = WebMiningToolkit()
                    with st.spinner("Descargando tabla..."):
                        wmt.fetch(url_input)
                        tbl = wmt.extract_table(index=int(t_idx))
                    st.session_state["ws_table"] = tbl
                    st.success(f"Tabla: {tbl.shape[0]} filas × {tbl.shape[1]} cols")
                except Exception as e:
                    st.error(f"Error: {e}")

        ws_tbl = st.session_state.get("ws_table")
        if ws_tbl is not None:
            st.dataframe(ws_tbl, use_container_width=True)
            st.download_button("Descargar CSV", ws_tbl.to_csv(index=False).encode(),
                               "tabla.csv", "text/csv", key="ws_tbl_dl")
            if st.button("Usar como dataset activo", key="ws_tbl_use"):
                st.session_state["df"]      = ws_tbl
                st.session_state["df_name"] = "tabla_scrapeada.csv"
                st.success("Dataset actualizado con la tabla extraída.")

    # ── Registros ─────────────────────────────────────────────────────────────
    with tab_rec:
        st.caption("Define el selector del bloque repetido y los campos a extraer. Útil para catálogos y listados.")
        item_sel = st.text_input("Selector del bloque repetido",
                                 placeholder=".product, .item, li.card", key="ws_item_sel")
        n_fields = int(st.number_input("Número de campos", 1, 10, 3, key="ws_nfields"))

        fields: dict = {}
        for i in range(n_fields):
            rc1, rc2, rc3 = st.columns(3)
            with rc1: fname = st.text_input(f"Campo {i+1}: nombre",    key=f"ws_fn_{i}", placeholder=f"campo_{i+1}")
            with rc2: fsel  = st.text_input(f"Campo {i+1}: selector",  key=f"ws_fs_{i}", placeholder=".price, h2 a")
            with rc3: fattr = st.text_input(f"Campo {i+1}: atributo",  key=f"ws_fa_{i}", value="text",
                                            placeholder="text / href / src")
            if fname.strip() and fsel.strip():
                fields[fname.strip()] = {"selector": fsel.strip(), "attr": fattr.strip() or "text"}

        if st.button("Extraer registros", type="primary", key="ws_fetch_rec"):
            if not url_input:
                st.error("Ingresa una URL.")
            elif not item_sel or not fields:
                st.error("Define el selector y al menos un campo.")
            else:
                try:
                    wmt = WebMiningToolkit()
                    with st.spinner("Extrayendo registros..."):
                        wmt.fetch(url_input)
                        rec_df = wmt.extract_records(item_selector=item_sel, fields=fields)
                    st.session_state["ws_records"] = rec_df
                    st.success(f"{len(rec_df)} registros extraídos.")
                except Exception as e:
                    st.error(f"Error: {e}")

        ws_rec = st.session_state.get("ws_records")
        if ws_rec is not None:
            st.dataframe(ws_rec, use_container_width=True)
            st.download_button("Descargar CSV", ws_rec.to_csv(index=False).encode(),
                               "registros.csv", "text/csv", key="ws_rec_dl")
            if st.button("Usar como dataset activo", key="ws_rec_use"):
                st.session_state["df"]      = ws_rec
                st.session_state["df_name"] = "registros_scrapeados.csv"
                st.success("Dataset actualizado con los registros extraídos.")

    # ── Regex ─────────────────────────────────────────────────────────────────
    with tab_regex:
        st.caption("Aplica operaciones regex sobre texto de una URL o sobre una columna del dataset activo.")

        rg_src = st.radio("Fuente de texto",
                          ["Extraer de URL", "Columna del dataset activo"], key="ws_rg_src")

        if rg_src == "Columna del dataset activo":
            if st.session_state["df"] is None:
                st.warning("No hay dataset cargado.")
                source_texts: list = []
            else:
                rg_col = st.selectbox("Columna", st.session_state["df"].columns.tolist(), key="ws_rg_col")
                source_texts = st.session_state["df"][rg_col].dropna().astype(str).tolist()
                st.caption(f"{len(source_texts)} textos listos.")
        else:
            rg_css = st.text_input("Selector CSS para extraer textos", placeholder="p, span", key="ws_rg_css")
            if st.button("Cargar textos de URL", key="ws_rg_load"):
                if url_input:
                    try:
                        wmt = WebMiningToolkit()
                        wmt.fetch(url_input)
                        loaded = wmt.extract_text(css_selector=rg_css.strip() or None,
                                                  tag="p" if not rg_css.strip() else None)
                        st.session_state["ws_regex_texts"] = loaded
                        st.success(f"{len(loaded)} textos cargados.")
                    except Exception as e:
                        st.error(f"Error: {e}")
            source_texts = st.session_state.get("ws_regex_texts", [])

        rp1, rp2 = st.columns(2)
        with rp1: rg_pattern = st.text_input("Patrón regex", placeholder=r"\d+\.\d+", key="ws_rg_pat")
        with rp2: rg_mode    = st.radio("Modo", ["Filtrar (search)", "Extraer grupos (extract)"], key="ws_rg_mode")

        if st.button("Aplicar regex", type="primary", key="ws_apply_rg"):
            if not rg_pattern:
                st.error("Define un patrón.")
            elif not source_texts:
                st.warning("No hay textos para procesar.")
            else:
                try:
                    wmt = WebMiningToolkit()
                    if "Filtrar" in rg_mode:
                        filtered = wmt.regex_filter(source_texts, rg_pattern)
                        st.success(f"{len(filtered)} textos coinciden.")
                        for t in filtered[:30]:
                            st.markdown(f"- {t[:250]}")
                    else:
                        rg_df = wmt.regex_extract(source_texts, rg_pattern)
                        if rg_df.empty:
                            st.warning("Sin coincidencias.")
                        else:
                            st.success(f"{len(rg_df)} coincidencias.")
                            st.dataframe(rg_df, use_container_width=True)
                            st.download_button("Descargar CSV", rg_df.to_csv(index=False).encode(),
                                               "regex_result.csv", "text/csv", key="ws_rg_dl")
                except Exception as e:
                    st.error(f"Error: {e}")
