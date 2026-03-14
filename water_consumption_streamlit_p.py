import warnings
from pathlib import Path
from typing import Dict, List
from sklearn.neural_network import MLPRegressor

import numpy as np
import pandas as pd
import streamlit as st

from ml_toolkit import ARIMAForecaster, EDAExplorer, HoltWintersForecaster, TimeSeriesRunner, DataPreparer
from visualizer import Visualizer

warnings.filterwarnings("ignore")
st.set_page_config(
    page_title="Water Consumption Prediction Dashboard",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------
# Styles
# ---------------------------
st.markdown(
    """
    <style>
    .main { background-color: #f8fafc; }
    .block-container { padding-top: 1.4rem; padding-bottom: 2rem; }
    section[data-testid="stSidebar"] {
        background: white;
        border-right: 1px solid #e5e7eb;
    }
    .section-title {
        font-size: 2rem;
        font-weight: 800;
        color: #111827;
        margin-bottom: 0.2rem;
        white-space: normal;
        word-break: break-word;
        line-height: 1.2;
    }
    .section-subtitle {
        color: #6b7280;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 18px;
        padding: 18px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
        min-height: 116px;
    }
    .metric-title { font-size: .85rem; color: #6b7280; font-weight: 600; margin-bottom: 6px; }
    .metric-value { font-size: 1.8rem; font-weight: 800; color: #111827; }
    .metric-caption { font-size: .8rem; color: #6b7280; margin-top: 8px; }
    .panel {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 18px;
        padding: 18px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }
    .panel-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #111827;
    margin-top: 18px;
    margin-bottom: 0.5rem;
    }
    .panel-subtitle { font-size: .85rem; color: #6b7280; margin-bottom: .8rem; }
    .status-box {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        border-radius: 14px;
        padding: 14px;
        margin-top: 1rem;
    }
    .small-muted { font-size: .82rem; color: #6b7280; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------
# Renderiza encabezados, métricas y prepara datos para la app
# ---------------------------
def render_header(title: str, subtitle: str) -> None:
    # Dibuja el título y subtítulo principal de cada página.
    st.markdown(
        f"""
        <div style="width: 100%; overflow: visible; margin-bottom: 0.8rem;">
            <h1 style="
                font-size: 2.1rem;
                font-weight: 800;
                color: #111827;
                margin: 0 0 0.25rem 0;
                line-height: 1.2;
                white-space: normal;
                word-break: break-word;
                overflow-wrap: anywhere;
            ">
                {title}
            </h1>
            <p style="
                color: #6b7280;
                margin: 0;
                font-size: 1rem;
            ">
                {subtitle}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(title: str, value: str, caption: str = "") -> None:
    # Dibuja una tarjeta numérica con el estilo visual definido en la app.
    caption_html = f'<div class="metric-caption">{caption}</div>' if caption else ""
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
            {caption_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_data(csv_path: str) -> pd.DataFrame:
    # Carga el CSV usando EDAExplorer del toolkit para mantener una lectura consistente.
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {csv_path}")

    # Reutiliza EDAExplorer para unificar la lectura del CSV con el toolkit.
    eda = EDAExplorer(str(path), modo_csv=2)
    df = eda.df.copy()
    df.columns = df.columns.str.replace('"', '', regex=False).str.strip()
    return df


def find_datetime_candidates(df: pd.DataFrame) -> List[str]:
    # Usa EDAExplorer del toolkit para inferir qué columnas parecen fechas.
    eda = EDAExplorer.from_df(df)
    return eda.detectar_columnas_fecha()


def prepare_timeseries_df(df: pd.DataFrame, date_col: str, target_col: str) -> pd.DataFrame:
    # Usa EDAExplorer del toolkit para normalizar la serie antes del modelado.
    eda = EDAExplorer.from_df(df)
    return eda.preparar_serie_temporal(date_col=date_col, target_col=target_col).df


def get_available_models() -> List[str]:
    # Define las opciones visibles en el selector de algoritmos.
    return [
        "Mejor modelo automático",
        "DeepLearning (Red Neuronal)",
        "HoltWinters",
        "HoltWinters-Calibrado",
        "Arima",
        "Arima-Calibrado",
    ]


def build_model_map(random_state: int = 42) -> Dict[str, object]:
    # Construye el catálogo de modelos que usará la app.
    # Incluye un modelo tabular de sklearn y adaptadores del toolkit para ARIMA/Holt-Winters.
    return {
        "DeepLearning (Red Neuronal)": MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            max_iter=500,
            random_state=random_state,
        ),
        "HoltWinters": HoltWintersForecaster(seasonal_periods=7),
        "HoltWinters-Calibrado": HoltWintersForecaster(seasonal_periods=12),
        "Arima": ARIMAForecaster(order=(1, 1, 1)),
        "Arima-Calibrado": ARIMAForecaster(order=(2, 1, 2)),
    }

@st.cache_data(show_spinner=False)
def analyze_time_series(
    csv_path: str,
    date_col: str,
    target_col: str,
    model_name: str,
    ranking_metric: str,
    lags: int,
    n_splits: int,
    train_pct: int,
    forecast_steps: int,
) -> Dict[str, object]:
    # Coordina el análisis principal:
    # prepara datos, ejecuta TimeSeriesRunner del toolkit y arma resultados para la UI.
    raw = load_data(csv_path)
    data = prepare_timeseries_df(raw, date_col, target_col)

    if len(data) <= lags + max(5, n_splits):
        raise ValueError("No hay suficientes registros para trabajar con esa cantidad de lags y folds.")

    data_model = data[[date_col, target_col]].copy()
    model_df = data_model[[target_col]].copy()
    train_size = train_pct / 100.0

    model_map = build_model_map()
    candidate_names = get_available_models()[1:] if model_name == "Mejor modelo automático" else [model_name]

    test_size = max(1, len(model_df) - int(round(len(model_df) * train_size)))
    cv_test_size = max(1, min(test_size, (len(model_df) - lags) // (n_splits + 1)))

    ranking_options = {
        "Holdout RMSE": "RMSE",
        "CV RMSE": "CV_RMSE",
    }
    sort_column = ranking_options.get(ranking_metric, "RMSE")

    evaluation_rows: List[Dict[str, float]] = []
    detailed: Dict[str, Dict[str, object]] = {}

    for name in candidate_names:
        if name not in model_map:
            raise ValueError(f"Modelo no soportado: {name}")

        # TimeSeriesRunner del toolkit ejecuta entrenamiento, validación y forecast
        # tanto para modelos por lags como para modelos clásicos de serie directa.
        model = model_map[name]
        preparer = DataPreparer(train_size=train_size, random_state=42)
        runner = TimeSeriesRunner(
            df=model_df,
            target=target_col,
            model=model,
            lags=lags,
            preparer=preparer,
            test_size=test_size,
        )

        holdout = runner.evaluate()
        cv = runner.evaluate_cv(n_splits=n_splits, test_size=cv_test_size)
        runner.fit_full()
        future = runner.forecast(steps=forecast_steps)

        # Reutiliza las métricas holdout que ya calcula TimeSeriesRunner en el toolkit.
        pred = runner.fit_predict()
        y_test = runner.y_test
        pred_index = runner.test_index

        evaluation_rows.append(
            {
                "Algoritmo": name,
                "MAE": float(holdout.get("MAE", np.nan)),
                "RMSE": float(holdout.get("RMSE", np.nan)),
                "MAPE_%": float(holdout.get("MAPE_%", np.nan)),
                "R2": float(holdout.get("R2", np.nan)),
                "CV_MAE": float(cv.get("MAE", np.nan)),
                "CV_RMSE": float(cv.get("RMSE", np.nan)),
                "CV_MAPE_%": float(cv.get("MAPE_%", np.nan)),
                "CV_R2": float(cv.get("R2", np.nan)),
            }
        )

        pred_df = pd.DataFrame(
            {
                # Reconstruye la fecha del holdout para que Visualizer grafique la comparación temporal.
                "date": data_model.loc[pred_index, date_col].values,
                "actual": np.asarray(y_test),
                "predicted": np.asarray(pred),
            }
        )

        # El forecast siempre se proyecta hacia adelante desde la última fecha observada.
        future_dates = pd.date_range(
            data_model[date_col].max() + pd.Timedelta(days=1),
            periods=forecast_steps,
            freq="D"
        )
        future_df = pd.DataFrame({"date": future_dates, "forecast": np.asarray(future)})

        detailed[name] = {
            "holdout": holdout,
            "cv": cv,
            "pred_df": pred_df,
            "future_df": future_df,
        }

    results_df = pd.DataFrame(evaluation_rows).sort_values(sort_column, ascending=True).reset_index(drop=True)
    best_name = results_df.iloc[0]["Algoritmo"]

    return {
        "data": data_model,
        "results_df": results_df,
        "best_name": best_name,
        "best_detail": detailed[best_name],
        "detailed": detailed,
        "ranking_metric_label": ranking_metric,
        "ranking_metric_column": sort_column,
        "train_pct": train_pct,
        "test_pct": 100 - train_pct,
        "lags": lags,
        "n_splits": n_splits,
        "forecast_steps": forecast_steps,
    }


def make_distribution_bins(series: pd.Series, bins: int = 8) -> pd.DataFrame:
    # Convierte una serie numérica en una tabla de frecuencias para graficarla con Visualizer.
    hist, edges = np.histogram(series.dropna(), bins=bins)
    labels = [f"{edges[i]:.0f}-{edges[i+1]:.0f}" for i in range(len(edges) - 1)]
    return pd.DataFrame({"range": labels, "count": hist})


def series_frequency_hint(date_series: pd.Series) -> str:
    # Resume la frecuencia dominante de la serie para mostrarla como texto en la UI.
    diffs = date_series.sort_values().diff().dropna()
    if diffs.empty:
        return "No determinada"
    mode = diffs.mode().iloc[0]
    if mode <= pd.Timedelta(hours=1):
        return "Horaria"
    if mode <= pd.Timedelta(days=1):
        return "Diaria"
    if mode <= pd.Timedelta(days=7):
        return "Semanal"
    if mode <= pd.Timedelta(days=31):
        return "Mensual"
    return str(mode)


def render_eda_page(data: pd.DataFrame, df_raw: pd.DataFrame, date_col: str, target_col: str, viz: Visualizer) -> None:
    # Página EDA: usa Visualizer para las gráficas y dataframes simples para los resúmenes.
    render_header(
        "Análisis Exploratorio de Datos (EDA)",
        "Resumen estadístico y visual del dataset de consumo de agua.",
    )

    null_values = int(data[target_col].isna().sum())
    full_date_range = pd.date_range(data[date_col].min(), data[date_col].max(), freq="D")
    missing_dates = int(len(full_date_range.difference(pd.DatetimeIndex(data[date_col]))))

    c1, c2, c3 = st.columns(3)
    with c1:
        render_metric_card("Total de Registros", f"{len(data):,}", "Filas válidas del dataset")
    with c2:
        render_metric_card("Valores Nulos", f"{null_values}", f"{(null_values/max(len(data),1))*100:.2f}% del total")
    with c3:
        render_metric_card("Fechas Faltantes", f"{missing_dates}", "Comparado contra frecuencia diaria")

    desc = data[target_col].describe()
    st.markdown("<div class='panel-title'>Estadísticas Descriptivas del Consumo</div>", unsafe_allow_html=True)
    s1, s2, s3, s4, s5 = st.columns(5)
    stats = [
        ("Media", f"{desc['mean']:.2f}"),
        ("Mediana", f"{data[target_col].median():.2f}"),
        ("Desv. Est.", f"{desc['std']:.2f}"),
        ("Mínimo", f"{desc['min']:.2f}"),
        ("Máximo", f"{desc['max']:.2f}"),
    ]
    for col, (label, value) in zip([s1, s2, s3, s4, s5], stats):
        with col:
            render_metric_card(label, value)

    left, right = st.columns(2)
    with left:
        st.markdown("<div class='panel-title'>Serie temporal completa</div>", unsafe_allow_html=True)
        fig = viz.line_chart(
            data,
            x=date_col,
            y=target_col,
            height=360,
        )
        st.plotly_chart(fig, width="stretch")

    with right:
        st.markdown("<div class='panel-title'>Distribución del consumo</div>", unsafe_allow_html=True)
        dist_df = make_distribution_bins(data[target_col], bins=8)
        fig_hist = viz.grouped_bar_chart(
            dist_df,
            x="range",
            y="count",
            color="range",
            height=360,
            showlegend=False,
        )
        st.plotly_chart(fig_hist, width="stretch")

    st.markdown("<div class='panel-title'>Información del dataset</div>", unsafe_allow_html=True)
    info_df = pd.DataFrame(
        {
            "Métrica": [
                "Rango de fechas",
                "Frecuencia estimada",
                "Registros duplicados",
                "Completitud",
                "IQR",
            ],
            "Valor": [
                f"{data[date_col].min().date()} - {data[date_col].max().date()}",
                series_frequency_hint(data[date_col]),
                int(df_raw.duplicated().sum()),
                f"{((len(data) - null_values) / max(len(data),1))*100:.2f}%",
                f"{data[target_col].quantile(0.75) - data[target_col].quantile(0.25):.2f}",
            ],
        }
    )
    st.dataframe(info_df, width="stretch", hide_index=True)
    with st.expander("Vista previa del dataset"):
        st.dataframe(data.head(30), width="stretch")


def render_model_page(analysis: Dict[str, object], results_df: pd.DataFrame, best_name: str, best_detail: Dict[str, object], viz: Visualizer) -> None:
    # Página de rendimiento: toma resultados ya calculados y los presenta con gráficas de Visualizer.
    render_header(
        "Rendimiento del Modelo",
        "Comparación de modelos de series temporales y métricas con validación temporal.",
    )

    best_row = results_df.iloc[0]
    ranking_metric_label = analysis["ranking_metric_label"]
    ranking_metric_column = analysis["ranking_metric_column"]
    ranking_value = best_row[ranking_metric_column]
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card("Mejor modelo", best_name, f"Ordenado por menor {ranking_metric_label}")
    with c2:
        render_metric_card(ranking_metric_label, f"{ranking_value:.3f}", f"K-Folds: {analysis['n_splits']}")
    with c3:
        mae_label = "CV MAE" if ranking_metric_column == "CV_RMSE" else "Holdout MAE"
        mae_value = best_row["CV_MAE"] if ranking_metric_column == "CV_RMSE" else best_row["MAE"]
        render_metric_card(mae_label, f"{mae_value:.3f}", f"Lags: {analysis['lags']}")
    with c4:
        mape_label = "CV MAPE" if ranking_metric_column == "CV_RMSE" else "Holdout MAPE"
        mape_value = best_row["CV_MAPE_%"] if ranking_metric_column == "CV_RMSE" else best_row["MAPE_%"]
        render_metric_card(mape_label, f"{mape_value:.2f}%", f"Train/Test: {analysis['train_pct']}/{analysis['test_pct']}")

    left, right = st.columns(2)
    with left:
        st.markdown("<div class='panel-title'>Comparación de métricas por modelo</div>", unsafe_allow_html=True)
        metric_columns = ["CV_RMSE", "CV_MAE", "CV_MAPE_%"] if ranking_metric_column == "CV_RMSE" else ["RMSE", "MAE", "MAPE_%"]
        metrics_long = results_df.melt(id_vars="Algoritmo", value_vars=metric_columns, var_name="Métrica", value_name="Valor")
        fig_bar = viz.grouped_bar_chart(
            metrics_long,
            x="Algoritmo",
            y="Valor",
            color="Métrica",
            barmode="group",
            height=380,
        )
        st.plotly_chart(fig_bar, width="stretch")

    with right:
        st.markdown("<div class='panel-title'>Real vs predicción en holdout</div>", unsafe_allow_html=True)
        pred_df = best_detail["pred_df"]
        fig_pred = viz.multi_line_chart(
            [
                {"x": pred_df["date"], "y": pred_df["actual"], "mode": "lines+markers", "name": "Actual"},
                {
                    "x": pred_df["date"],
                    "y": pred_df["predicted"],
                    "mode": "lines+markers",
                    "name": "Predicho",
                    "line": dict(dash="dash"),
                },
            ],
            height=380,
        )
        st.plotly_chart(fig_pred, width="stretch")

    st.markdown("<div class='panel-title'>Ranking de modelos</div>", unsafe_allow_html=True)
    display_df = results_df.copy()
    for col in ["MAE", "RMSE", "MAPE_%", "R2", "CV_MAE", "CV_RMSE", "CV_MAPE_%", "CV_R2"]:
        display_df[col] = display_df[col].map(lambda x: round(float(x), 4) if pd.notna(x) else x)
    st.dataframe(display_df, width="stretch", hide_index=True)


def render_predictions_page(
    analysis: Dict[str, object],
    data: pd.DataFrame,
    date_col: str,
    target_col: str,
    best_name: str,
    best_detail: Dict[str, object],
    selected_model: str,
    viz: Visualizer,
) -> None:
    # Página de predicciones: muestra el forecast generado por TimeSeriesRunner con apoyo de Visualizer.
    render_header(
        "Predicciones",
        "Pronóstico de consumo de agua basado en el mejor modelo o en el algoritmo seleccionado.",
    )

    c1, c2, c3, c4 = st.columns(4)
    forecast_df = best_detail["future_df"]
    with c1:
        render_metric_card("Modelo usado", best_name)
    with c2:
        render_metric_card("Horizonte", f"{analysis['forecast_steps']} días")
    with c3:
        render_metric_card("Promedio proyectado", f"{forecast_df['forecast'].mean():.2f}")
    with c4:
        trend = forecast_df["forecast"].iloc[-1] - forecast_df["forecast"].iloc[0]
        render_metric_card("Tendencia proyectada", f"{trend:+.2f}")

    combined_hist = data.tail(30).rename(columns={date_col: "date", target_col: "actual"})[["date", "actual"]]
    combined = combined_hist.copy()
    combined["forecast"] = np.nan
    future_plot = forecast_df.copy()
    future_plot["actual"] = np.nan
    chart_df = pd.concat([combined, future_plot], ignore_index=True)

    st.markdown("<div class='panel-title'>Serie histórica y pronóstico</div>", unsafe_allow_html=True)
    fig_forecast = viz.multi_line_chart(
        [
            {"x": chart_df["date"], "y": chart_df["actual"], "mode": "lines+markers", "name": "Histórico"},
            {
                "x": chart_df["date"],
                "y": chart_df["forecast"],
                "mode": "lines+markers",
                "name": "Forecast",
                "line": dict(dash="dash"),
            },
        ],
        height=420,
    )
    st.plotly_chart(fig_forecast, width="stretch")

    st.markdown("<div class='panel-title'>Predicciones detalladas</div>", unsafe_allow_html=True)
    pred_table = forecast_df.copy()
    pred_table["date"] = pred_table["date"].dt.strftime("%Y-%m-%d")
    pred_table["forecast"] = pred_table["forecast"].round(3)
    st.dataframe(pred_table, width="stretch", hide_index=True)

    st.markdown("<div class='panel-title'>Configuración activa</div>", unsafe_allow_html=True)
    config_df = pd.DataFrame(
        {
            "Parámetro": ["Algoritmo", "Lags", "K-Folds", "Train %", "Test %", "Horizonte"],
            "Valor": [
                best_name if selected_model == "Mejor modelo automático" else selected_model,
                analysis["lags"],
                analysis["n_splits"],
                analysis["train_pct"],
                analysis["test_pct"],
                analysis["forecast_steps"],
            ],
        }
    )
    config_df["Valor"] = config_df["Valor"].astype(str)
    st.dataframe(config_df, width="stretch", hide_index=True)


# ---------------------------
# Sidebar
# ---------------------------
with st.sidebar:
    # La barra lateral reúne los parámetros que controlan el análisis.
    st.markdown("## 💧 Water Consumption")
    st.caption("Dashboard de series temporales")
    st.divider()

    dataset_path = st.text_input("Ruta del CSV", value="consumo_agua.csv")
    page = st.radio(
        "Navegación",
        [
            "EDA - Análisis Exploratorio",
            "Rendimiento del Modelo",
            "Predicciones",
        ],
        label_visibility="collapsed",
    )

    st.markdown("<div style='height: 200px;'></div>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="status-box">
            <div style="font-weight:700; color:#1d4ed8;">● Sistema Activo</div>
            <div class="small-muted">Última actualización: {pd.Timestamp.now().strftime('%H:%M:%S')}</div>
            <div class="small-muted">Dataset esperado: consumo_agua.csv</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------
# Main logic
# ---------------------------
try:
    # Primera carga del dataset usando la función que delega en EDAExplorer.
    df_raw = load_data(dataset_path)
except Exception as exc:
    st.error(f"No pude abrir el dataset. Coloca `consumo_agua.csv` en la misma carpeta del app o ajusta la ruta.\n\nDetalle: {exc}")
    st.stop()

# Estas listas alimentan los controles de la barra lateral.
candidates_date = find_datetime_candidates(df_raw)
num_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()

if not candidates_date:
    st.error("No encontré una columna de fecha interpretable en el CSV.")
    st.stop()
if not num_cols:
    st.error("No encontré columnas numéricas para usar como variable objetivo.")
    st.stop()

with st.sidebar:
    # Estos controles alimentan analyze_time_series y cambian el contenido de las páginas.
    date_col = st.selectbox("Columna de fecha", candidates_date, index=0)
    default_target_idx = 0
    target_col = st.selectbox("Variable objetivo", num_cols, index=default_target_idx)

    model_names = get_available_models()
    selected_model = st.selectbox("Algoritmo", model_names, index=0)
    ranking_metric = st.selectbox(
        "Criterio de selección",
        ["Holdout RMSE", "CV RMSE"],
        index=0,
    )
    lags = st.slider("Cantidad de lags", min_value=3, max_value=30, value=7, step=1)
    n_splits = st.slider("K-Folds temporales", min_value=2, max_value=10, value=5, step=1)
    train_pct = st.slider("Porcentaje entrenamiento", min_value=60, max_value=90, value=80, step=5)
    forecast_steps = st.slider("Horizonte de predicción", min_value=3, max_value=30, value=7, step=1)

analysis = analyze_time_series(
    dataset_path,
    date_col,
    target_col,
    selected_model,
    ranking_metric,
    lags,
    n_splits,
    train_pct,
    forecast_steps,
)

# Objetos finales que consumen las funciones render_* para construir la interfaz.
data = analysis["data"]
results_df = analysis["results_df"]
best_name = analysis["best_name"]
best_detail = analysis["best_detail"]
viz = Visualizer()

if page == "EDA - Análisis Exploratorio":
    render_eda_page(data, df_raw, date_col, target_col, viz)
elif page == "Rendimiento del Modelo":
    render_model_page(analysis, results_df, best_name, best_detail, viz)
elif page == "Predicciones":
    render_predictions_page(analysis, data, date_col, target_col, best_name, best_detail, selected_model, viz)
