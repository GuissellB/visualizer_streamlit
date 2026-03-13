import math
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

TARGET_COL = "Result"
DATA_PATH = "Phishing_Websites_Data.csv"


@st.cache_data
def load_eda_data(path: str):
    df = pd.read_csv(path)

    # Limpieza básica para EDA
    df = df.copy()

    total_rows = len(df)
    total_cols = df.shape[1]
    null_count = int(df.isna().sum().sum())
    dup_count = int(df.duplicated().sum())

    if TARGET_COL not in df.columns:
        raise ValueError(f"No se encontró la columna target '{TARGET_COL}' en el dataset.")

    class_counts = df[TARGET_COL].value_counts().sort_index()

    phishing_count = int(class_counts.get(-1, 0))
    legit_count = int(class_counts.get(1, 0))

    # Ratio simple entre clases
    ratio_text = f"1:{round(legit_count / phishing_count)}" if phishing_count > 0 else "N/A"

    # Correlación absoluta con target para Top Features
    numeric_df = df.select_dtypes(include=["number"]).copy()
    corr_with_target = (
        numeric_df.corr(numeric_only=True)[TARGET_COL]
        .drop(TARGET_COL)
        .abs()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )
    corr_with_target.columns = ["feature", "importance"]

    pie_df = pd.DataFrame(
        {
            "tipo": ["Legítimo", "Phishing"],
            "cantidad": [legit_count, phishing_count],
        }
    )

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

    return df, summary, corr_with_target, pie_df

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
# Helpers
# =========================
def hero(title: str, subtitle: str) -> None:
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


def panel_open(title: str, subtitle: str = "", extra_class: str = "") -> None:
    st.markdown(
        f'<div class="panel {extra_class}"><h3>{title}</h3><div class="sub">{subtitle}</div>',
        unsafe_allow_html=True,
    )


def panel_close() -> None:
    st.markdown("</div>", unsafe_allow_html=True)

df_eda, eda_summary, feature_importance_real, pie_df_real = load_eda_data(DATA_PATH)
# =========================
# Sidebar
# =========================
with st.sidebar:
    st.markdown("## 🛡️ Phishing Detection")
    st.caption("Dashboard Analytics")
    page = st.radio(
        "Navegación",
        [
            "EDA - Análisis Exploratorio",
            "Rendimiento del Algoritmo",
            "Parametrización del Modelo",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
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

# =========================
# Página 1: EDA
# =========================
if page == "EDA - Análisis Exploratorio":
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

        fig = px.bar(
            feature_importance_real.sort_values("importance"),
            x="importance",
            y="feature",
            orientation="h",
            text=feature_importance_real.sort_values("importance")["importance"].map(lambda x: f"{x:.2f}"),
            color="importance",
            color_continuous_scale=["#60a5fa", "#7c3aed"],
        )

        fig.update_traces(
            textposition="inside",
            marker_line_width=0,
            cliponaxis=False,
            insidetextanchor="end",
        )

        fig.update_yaxes(automargin=True)
        fig.update_xaxes(range=[0, float(feature_importance_real["importance"].max()) * 1.05])

        fig.update_layout(
            height=390,
            margin=dict(l=20, r=20, t=10, b=20),
            coloraxis_showscale=False,
            xaxis=dict(
                title="Correlación absoluta",
                showgrid=True,
                gridcolor="rgba(148,163,184,0.18)",
                zeroline=False,
            ),
            yaxis=dict(
                title="",
                showgrid=False,
                automargin=True,
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(248,250,252,0.92)",
            font=dict(color="#1e293b"),
        )

        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        panel_close()

    with right:
        panel_open("Distribución general", "Balance real entre clases del dataset.")

        fig_pie = px.pie(
            pie_df_real,
            names="tipo",
            values="cantidad",
            hole=0.68,
            color="tipo",
            color_discrete_map={"Legítimo": "#2563eb", "Phishing": "#7c3aed"},
        )

        fig_pie.update_traces(textinfo="percent+label", textposition="inside")

        fig_pie.update_layout(
            height=390,
            margin=dict(l=15, r=15, t=10, b=15),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(248,250,252,0.92)",
            font=dict(color="#1e293b"),
            legend=dict(
                orientation="h",
                y=-0.08,
                x=0.5,
                xanchor="center",
            ),
            uniformtext_minsize=12,
            uniformtext_mode="hide",
        )

        st.plotly_chart(fig_pie, width="stretch", config={"displayModeBar": False})
        panel_close()

    panel_open("Resumen de calidad de datos", "Estado real del dataset cargado.")
    q1, q2, q3 = st.columns(3, gap="medium")

    with q1:
        st.markdown(
            f"""
            <div class="panel metric-panel">
                <h3>Valores nulos</h3>
                <div class="sub">Conteo total de valores faltantes.</div>
                <div class="metric-panel-value" style="color:#10b981;">{eda_summary['null_count']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with q2:
        st.markdown(
            f"""
            <div class="panel metric-panel">
                <h3>Duplicados</h3>
                <div class="sub">Filas repetidas detectadas en el dataset.</div>
                <div class="metric-panel-value" style="color:#f59e0b;">{eda_summary['dup_count']:,}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with q3:
        st.markdown(
            f"""
            <div class="panel metric-panel">
                <h3>Variable objetivo</h3>
                <div class="sub">Columna usada como clase a predecir.</div>
                <div class="metric-panel-value" style="color:#2563eb;">{TARGET_COL}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    panel_close()

# =========================
# Página 2: Performance
# =========================
elif page == "Rendimiento del Algoritmo":
    hero(
        "Rendimiento del Algoritmo",
        "Métricas detalladas del desempeño del modelo de detección.",
    )

    cols = st.columns(5)
    metrics = [
        ("Exactitud (Accuracy)", f"{ACCURACY:.2f}%", "Desempeño global", "#4f46e5"),
        ("Precisión (Precision)", f"{PRECISION:.2f}%", "Control de falsos positivos", "#10b981"),
        ("Sensibilidad (Recall)", f"{RECALL:.2f}%", "Detección de phishing", "#2563eb"),
        ("F1-Score", f"{F1:.2f}%", "Balance precisión-recall", "#7c3aed"),
        ("Especificidad", f"{SPECIFICITY:.2f}%", "Detección de legítimos", "#f59e0b"),
    ]
    for col, m in zip(cols, metrics):
        with col:
            metric_card(*m)

    roc_col, pr_col = st.columns(2)

    with roc_col:
        panel_open("Curva ROC (Receiver Operating Characteristic)", "AUC (Área Bajo la Curva): 0.987")
        roc_fig = go.Figure()
        roc_fig.add_trace(
            go.Scatter(
                x=roc_df["fpr"],
                y=roc_df["tpr"],
                mode="lines+markers",
                name="ROC Curve",
                line=dict(width=4, color="#2563eb"),
                marker=dict(size=7, color="#7c3aed"),
            )
        )
        roc_fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Random Classifier",
                line=dict(width=2, dash="dash", color="rgba(100,116,139,0.8)"),
            )
        )
        roc_fig.update_layout(
            height=380,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Tasa de Falsos Positivos (FPR)",
            yaxis_title="Tasa de Verdaderos Positivos (TPR)",
            xaxis=dict(gridcolor="rgba(148,163,184,0.18)"),
            yaxis=dict(gridcolor="rgba(148,163,184,0.18)"),
        )
        st.plotly_chart(roc_fig, width="stretch")
        panel_close()

    with pr_col:
        panel_open("Curva Precision-Recall", "AP (Average Precision): 0.934")
        pr_fig = go.Figure()
        pr_fig.add_trace(
            go.Scatter(
                x=pr_df["recall"],
                y=pr_df["precision"],
                mode="lines+markers",
                name="PR Curve",
                line=dict(width=4, color="#10b981"),
                marker=dict(size=7, color="#06b6d4"),
                fill="tozeroy",
                fillcolor="rgba(16,185,129,0.12)",
            )
        )
        pr_fig.update_layout(
            height=380,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Recall (Sensibilidad)",
            yaxis_title="Precision",
            xaxis=dict(gridcolor="rgba(148,163,184,0.18)"),
            yaxis=dict(gridcolor="rgba(148,163,184,0.18)"),
        )
        st.plotly_chart(pr_fig, width="stretch")
        panel_close()

    panel_open("Matriz de confusión", f"Basada en {TOTAL:,} websites analizados.")

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
            st.markdown(
                "<div style='height:20px;'></div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<div style='height:168px;display:flex;align-items:center;justify-content:flex-end;font-weight:800;color:#334155;font-size:1rem;'>Real: Legítimo</div>",
                unsafe_allow_html=True,
            )

        with matrix_cols:
            a, b = st.columns(2, gap="medium")

            with a:
                st.markdown(
                    f"<div class='cm-box' style='background:#dcfce7;border:2px solid #22c55e;color:#166534;'><div style='font-size:2rem'>{confusion['TP']}</div><div>Verdaderos Positivos</div><div class='small-muted'>{confusion['TP']/TOTAL*100:.2f}%</div></div>",
                    unsafe_allow_html=True,
                )
                st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
                st.markdown(
                    f"<div class='cm-box' style='background:#ffedd5;border:2px solid #f59e0b;color:#9a3412;'><div style='font-size:2rem'>{confusion['FP']}</div><div>Falsos Positivos</div><div class='small-muted'>{confusion['FP']/TOTAL*100:.2f}%</div></div>",
                    unsafe_allow_html=True,
                )

            with b:
                st.markdown(
                    f"<div class='cm-box' style='background:#fee2e2;border:2px solid #ef4444;color:#991b1b;'><div style='font-size:2rem'>{confusion['FN']}</div><div>Falsos Negativos</div><div class='small-muted'>{confusion['FN']/TOTAL*100:.2f}%</div></div>",
                    unsafe_allow_html=True,
                )
                st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
                st.markdown(
                    f"<div class='cm-box' style='background:#dbeafe;border:2px solid #3b82f6;color:#1d4ed8;'><div style='font-size:2rem'>{confusion['TN']}</div><div>Verdaderos Negativos</div><div class='small-muted'>{confusion['TN']/TOTAL*100:.2f}%</div></div>",
                    unsafe_allow_html=True,
                )

    panel_close()

    err1, err2, err3 = st.columns(3, gap="medium")

    with err1:
        st.markdown(
            f"""
            <div class="panel metric-panel">
                <h3>Falsos Negativos</h3>
                <div class="sub">Phishing no detectado (crítico).</div>
                <div class="metric-panel-value" style="color:#ef4444;">{confusion['FN']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with err2:
        st.markdown(
            f"""
            <div class="panel metric-panel">
                <h3>Falsos Positivos</h3>
                <div class="sub">Website legítimo marcado como phishing.</div>
                <div class="metric-panel-value" style="color:#f59e0b;">{confusion['FP']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with err3:
        st.markdown(
            f"""
            <div class="panel metric-panel">
                <h3>Tasa de Error Total</h3>
                <div class="sub">Porcentaje total de clasificaciones incorrectas.</div>
                <div class="metric-panel-value" style="color:#2563eb;">{ERROR_RATE:.3f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# =========================
# Página 3: Parámetros
# =========================
else:
    hero(
        "Parametrización del Modelo",
        "Configuración e hiperparámetros del modelo.",
    )

    panel_open("Información del modelo", model_config["algorithm"])
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Versión", model_config["version"], "Modelo desplegado", "#2563eb")
    with c2:
        metric_card("Fecha de entrenamiento", model_config["training_date"], "Último training completo", "#7c3aed")
    with c3:
        metric_card("Última actualización", model_config["last_update"], "Refresh de configuración", "#06b6d4")
    with c4:
        metric_card("Estado", model_config["status"], "Modelo activo en producción", "#10b981")
    panel_close()

    panel_open("Hiperparámetros del Random Forest", "Tabla base del modelo activo.")
    st.dataframe(hyperparameters, width="stretch", hide_index=True)
    panel_close()

    panel_open("Feature Engineering (47 Features)", "Variables extraídas y utilizadas por el modelo.")
    fg_cols = st.columns(2)
    items = list(feature_groups.items())
    half = math.ceil(len(items) / 2)
    chunks = [items[:half], items[half:]]
    for col, chunk in zip(fg_cols, chunks):
        with col:
            for group_name, feats in chunk:
                pills = "".join([f"<span class='pill'>{feat}</span>" for feat in feats])
                st.markdown(
                    f"""
                    <div class="panel" style="padding:1rem; margin-bottom:0.8rem;">
                        <h3 style="font-size:1rem; margin-bottom:0.55rem;">{group_name}</h3>
                        <div>{pills}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    panel_close()

    panel_open("Métricas de entrenamiento", "Resumen del proceso de entrenamiento y validación.")
    tm_cols = st.columns(3)
    for idx, (label, value) in enumerate(training_metrics):
        with tm_cols[idx % 3]:
            st.markdown(
                f"""
                <div class="metric-card" style="min-height:110px; margin-bottom:0.8rem;">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value" style="font-size:1.4rem;">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    panel_close()

    panel_open("Configuración de umbral de decisión", "Comparación rápida de distintos thresholds de clasificación.")
    st.markdown(
        """
        <div style="background:linear-gradient(135deg, rgba(245,158,11,0.12), rgba(251,191,36,0.16)); border:1px solid rgba(245,158,11,0.22); border-radius:18px; padding:1rem; margin-bottom:1rem;">
            <div style="display:flex; justify-content:space-between; gap:12px; align-items:center;">
                <div>
                    <div style="font-weight:800; color:#854d0e;">Umbral Actual de Clasificación</div>
                    <div style="color:#a16207; font-size:0.92rem; margin-top:0.18rem;">Websites con probabilidad &gt; 0.65 son clasificados como phishing.</div>
                </div>
                <div style="background:#f59e0b; color:white; padding:0.35rem 0.75rem; border-radius:999px; font-weight:800;">0.65</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    t1, t2, t3 = st.columns(3)
    with t1:
        st.markdown(
            """
            <div class="threshold-card">
                <div class="metric-label">Umbral Conservador</div>
                <div class="metric-value" style="color:#ef4444;">0.50</div>
                <div class="metric-sub">Más detecciones, más falsos positivos</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with t2:
        st.markdown(
            """
            <div class="threshold-card active">
                <div class="metric-label" style="color:#1e3a8a;">Umbral Balanceado</div>
                <div class="metric-value" style="color:#2563eb;">0.65</div>
                <div class="metric-sub" style="color:#1d4ed8;">Balance óptimo actual</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with t3:
        st.markdown(
            """
            <div class="threshold-card">
                <div class="metric-label">Umbral Estricto</div>
                <div class="metric-value" style="color:#10b981;">0.80</div>
                <div class="metric-sub">Alta precisión, menos sensibilidad</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    panel_close()