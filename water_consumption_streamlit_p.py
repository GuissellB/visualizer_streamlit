import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from sklearn.neural_network import MLPRegressor

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# optional models

from ml_toolkit import TimeSeriesRunner, DataPreparer

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
# Helpers
# ---------------------------
def render_header(title: str, subtitle: str) -> None:
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
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {csv_path}")

    df = pd.read_csv(path, sep=";")
    df.columns = df.columns.str.replace('"', '', regex=False).str.strip()
    return df


def find_datetime_candidates(df: pd.DataFrame) -> List[str]:
    candidates = []
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            candidates.append(col)
            continue

        sample = df[col].dropna().astype(str).head(20)
        if sample.empty:
            continue

        parsed = pd.to_datetime(sample, errors="coerce")
        if parsed.notna().mean() >= 0.7:
            candidates.append(col)

    return candidates


def prepare_timeseries_df(df: pd.DataFrame, date_col: str, target_col: str) -> pd.DataFrame:
    data = df.copy()
    data.columns = data.columns.str.replace('"', '', regex=False).str.strip()
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
    data[target_col] = pd.to_numeric(data[target_col], errors="coerce")
    data = data.dropna(subset=[date_col, target_col]).sort_values(date_col).reset_index(drop=True)
    return data


def get_available_models() -> List[str]:
    return [
        "Mejor modelo automático",
        "DeepLearning (Red Neuronal)",
        "HoltWinters",
        "HoltWinters-Calibrado",
        "Arima",
        "Arima-Calibrado",
    ]


def build_model_map(random_state: int = 42) -> Dict[str, object]:
    return {
        "DeepLearning (Red Neuronal)": MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            max_iter=500,
            random_state=random_state,
        )
    }

@st.cache_data(show_spinner=False)
def analyze_time_series(
    csv_path: str,
    date_col: str,
    target_col: str,
    model_name: str,
    lags: int,
    n_splits: int,
    train_pct: int,
    forecast_steps: int,
) -> Dict[str, object]:
    raw = load_data(csv_path)
    data = prepare_timeseries_df(raw, date_col, target_col)

    if len(data) <= lags + max(5, n_splits):
        raise ValueError("No hay suficientes registros para trabajar con esa cantidad de lags y folds.")

    data_model = data[[date_col, target_col]].copy()
    model_df = data_model[[target_col]].copy()
    train_size = train_pct / 100.0

    sklearn_models = build_model_map()
    candidate_names = get_available_models()[1:] if model_name == "Mejor modelo automático" else [model_name]

    test_size = max(1, len(model_df) - int(round(len(model_df) * train_size)))
    cv_test_size = max(1, min(test_size, (len(model_df) - lags) // (n_splits + 1)))

    evaluation_rows: List[Dict[str, float]] = []
    detailed: Dict[str, Dict[str, object]] = {}

    for name in candidate_names:
        if name == "DeepLearning (Red Neuronal)":
            model = sklearn_models[name]
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

            pred = runner.fit_predict()
            y_test = runner.y_test
            pred_index = runner.X_test.index

        elif name in ["HoltWinters", "HoltWinters-Calibrado"]:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing

            series = model_df[target_col].astype(float).reset_index(drop=True)
            split_idx = int(round(len(series) * train_size))
            train_series = series.iloc[:split_idx]
            test_series = series.iloc[split_idx:]

            seasonal_periods = 7 if name == "HoltWinters" else 12

            hw_model = ExponentialSmoothing(
                train_series,
                trend="add",
                seasonal="add",
                seasonal_periods=seasonal_periods
            ).fit()

            pred = hw_model.forecast(len(test_series))
            y_test = test_series.values
            pred_index = test_series.index

            rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
            mae = float(mean_absolute_error(y_test, pred))
            r2 = float(r2_score(y_test, pred)) if len(y_test) > 1 else np.nan
            mape = float(
                np.nanmean(
                    np.abs(
                        (np.asarray(y_test) - np.asarray(pred)) /
                        np.where(np.asarray(y_test) == 0, np.nan, np.asarray(y_test))
                    )
                ) * 100
            )

            cv = {"RMSE": rmse, "MAE": mae, "MAPE_%": mape, "R2": r2}
            holdout = cv

            hw_full = ExponentialSmoothing(
                series,
                trend="add",
                seasonal="add",
                seasonal_periods=seasonal_periods
            ).fit()
            future = hw_full.forecast(forecast_steps)

        elif name in ["Arima", "Arima-Calibrado"]:
            from statsmodels.tsa.arima.model import ARIMA

            series = model_df[target_col].astype(float).reset_index(drop=True)
            split_idx = int(round(len(series) * train_size))
            train_series = series.iloc[:split_idx]
            test_series = series.iloc[split_idx:]

            order = (1, 1, 1) if name == "Arima" else (2, 1, 2)

            arima_model = ARIMA(train_series, order=order).fit()
            pred = arima_model.forecast(steps=len(test_series))
            y_test = test_series.values
            pred_index = test_series.index

            rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
            mae = float(mean_absolute_error(y_test, pred))
            r2 = float(r2_score(y_test, pred)) if len(y_test) > 1 else np.nan
            mape = float(
                np.nanmean(
                    np.abs(
                        (np.asarray(y_test) - np.asarray(pred)) /
                        np.where(np.asarray(y_test) == 0, np.nan, np.asarray(y_test))
                    )
                ) * 100
            )

            cv = {"RMSE": rmse, "MAE": mae, "MAPE_%": mape, "R2": r2}
            holdout = cv

            arima_full = ARIMA(series, order=order).fit()
            future = arima_full.forecast(steps=forecast_steps)

        else:
            raise ValueError(f"Modelo no soportado: {name}")

        rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
        mae = float(mean_absolute_error(y_test, pred))
        r2 = float(r2_score(y_test, pred)) if len(y_test) > 1 else np.nan
        mape = float(
            np.nanmean(
                np.abs(
                    (np.asarray(y_test) - np.asarray(pred)) /
                    np.where(np.asarray(y_test) == 0, np.nan, np.asarray(y_test))
                )
            ) * 100
        )

        evaluation_rows.append(
            {
                "Algoritmo": name,
                "MAE": mae,
                "RMSE": rmse,
                "MAPE_%": mape,
                "R2": r2,
                "CV_MAE": float(cv.get("MAE", np.nan)),
                "CV_RMSE": float(cv.get("RMSE", np.nan)),
                "CV_MAPE_%": float(cv.get("MAPE_%", np.nan)),
                "CV_R2": float(cv.get("R2", np.nan)),
            }
        )

        pred_df = pd.DataFrame(
            {
                "date": data_model.loc[pred_index, date_col].values,
                "actual": np.asarray(y_test),
                "predicted": np.asarray(pred),
            }
        )

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

    results_df = pd.DataFrame(evaluation_rows).sort_values("CV_RMSE", ascending=True).reset_index(drop=True)
    best_name = results_df.iloc[0]["Algoritmo"]

    return {
        "data": data_model,
        "results_df": results_df,
        "best_name": best_name,
        "best_detail": detailed[best_name],
        "detailed": detailed,
        "train_pct": train_pct,
        "test_pct": 100 - train_pct,
        "lags": lags,
        "n_splits": n_splits,
        "forecast_steps": forecast_steps,
    }


def make_distribution_bins(series: pd.Series, bins: int = 8) -> pd.DataFrame:
    hist, edges = np.histogram(series.dropna(), bins=bins)
    labels = [f"{edges[i]:.0f}-{edges[i+1]:.0f}" for i in range(len(edges) - 1)]
    return pd.DataFrame({"range": labels, "count": hist})


def series_frequency_hint(date_series: pd.Series) -> str:
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


# ---------------------------
# Sidebar
# ---------------------------
with st.sidebar:
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
    df_raw = load_data(dataset_path)
except Exception as exc:
    st.error(f"No pude abrir el dataset. Coloca `consumo_agua.csv` en la misma carpeta del app o ajusta la ruta.\n\nDetalle: {exc}")
    st.stop()

candidates_date = find_datetime_candidates(df_raw)
num_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()

if not candidates_date:
    st.error("No encontré una columna de fecha interpretable en el CSV.")
    st.stop()
if not num_cols:
    st.error("No encontré columnas numéricas para usar como variable objetivo.")
    st.stop()

with st.sidebar:
    date_col = st.selectbox("Columna de fecha", candidates_date, index=0)
    default_target_idx = 0
    target_col = st.selectbox("Variable objetivo", num_cols, index=default_target_idx)

    model_names = get_available_models()
    selected_model = st.selectbox("Algoritmo", model_names, index=0)
    lags = st.slider("Cantidad de lags", min_value=3, max_value=30, value=7, step=1)
    n_splits = st.slider("K-Folds temporales", min_value=2, max_value=10, value=5, step=1)
    train_pct = st.slider("Porcentaje entrenamiento", min_value=60, max_value=90, value=80, step=5)
    forecast_steps = st.slider("Horizonte de predicción", min_value=3, max_value=30, value=7, step=1)

analysis = analyze_time_series(
    dataset_path,
    date_col,
    target_col,
    selected_model,
    lags,
    n_splits,
    train_pct,
    forecast_steps,
)

data = analysis["data"]
results_df = analysis["results_df"]
best_name = analysis["best_name"]
best_detail = analysis["best_detail"]

# ---------------------------
# Pages
# ---------------------------
if page == "EDA - Análisis Exploratorio":
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
        fig = px.line(data, x=date_col, y=target_col)
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown("<div class='panel-title'>Distribución del consumo</div>", unsafe_allow_html=True)
        dist_df = make_distribution_bins(data[target_col], bins=8)
        fig_hist = px.bar(dist_df, x="range", y="count")
        fig_hist.update_layout(height=360, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig_hist, use_container_width=True)

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
    st.dataframe(info_df, use_container_width=True, hide_index=True)
    with st.expander("Vista previa del dataset"):
        st.dataframe(data.head(30), use_container_width=True)

elif page == "Rendimiento del Modelo":
    render_header(
        "Rendimiento del Modelo",
        "Comparación de modelos de series temporales y métricas con validación temporal.",
    )

    best_row = results_df.iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card("Mejor modelo", best_name, "Ordenado por menor CV RMSE")
    with c2:
        render_metric_card("CV RMSE", f"{best_row['CV_RMSE']:.3f}", f"K-Folds: {analysis['n_splits']}")
    with c3:
        render_metric_card("CV MAE", f"{best_row['CV_MAE']:.3f}", f"Lags: {analysis['lags']}")
    with c4:
        render_metric_card("CV MAPE", f"{best_row['CV_MAPE_%']:.2f}%", f"Train/Test: {analysis['train_pct']}/{analysis['test_pct']}")

    left, right = st.columns(2)
    with left:
        st.markdown("<div class='panel-title'>Comparación de métricas por modelo</div>", unsafe_allow_html=True)
        metrics_long = results_df.melt(id_vars="Algoritmo", value_vars=["CV_RMSE", "CV_MAE", "CV_MAPE_%"], var_name="Métrica", value_name="Valor")
        fig_bar = px.bar(metrics_long, x="Algoritmo", y="Valor", color="Métrica", barmode="group")
        fig_bar.update_layout(height=380, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig_bar, use_container_width=True)

    with right:
        st.markdown("<div class='panel-title'>Real vs predicción en holdout</div>", unsafe_allow_html=True)
        pred_df = best_detail["pred_df"]
        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(x=pred_df["date"], y=pred_df["actual"], mode="lines+markers", name="Actual"))
        fig_pred.add_trace(go.Scatter(x=pred_df["date"], y=pred_df["predicted"], mode="lines+markers", name="Predicho", line=dict(dash="dash")))
        fig_pred.update_layout(height=380, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig_pred, use_container_width=True)

    st.markdown("<div class='panel-title'>Ranking de modelos</div>", unsafe_allow_html=True)
    display_df = results_df.copy()
    for col in ["MAE", "RMSE", "MAPE_%", "R2", "CV_MAE", "CV_RMSE", "CV_MAPE_%", "CV_R2"]:
        display_df[col] = display_df[col].map(lambda x: round(float(x), 4) if pd.notna(x) else x)
    st.dataframe(display_df, use_container_width=True, hide_index=True)

elif page == "Predicciones":
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
    fig_forecast = go.Figure()
    fig_forecast.add_trace(go.Scatter(x=chart_df["date"], y=chart_df["actual"], mode="lines+markers", name="Histórico"))
    fig_forecast.add_trace(go.Scatter(x=chart_df["date"], y=chart_df["forecast"], mode="lines+markers", name="Forecast", line=dict(dash="dash")))
    fig_forecast.update_layout(height=420, margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig_forecast, use_container_width=True)

    st.markdown("<div class='panel-title'>Predicciones detalladas</div>", unsafe_allow_html=True)
    pred_table = forecast_df.copy()
    pred_table["date"] = pred_table["date"].dt.strftime("%Y-%m-%d")
    pred_table["forecast"] = pred_table["forecast"].round(3)
    st.dataframe(pred_table, use_container_width=True, hide_index=True)

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
    st.dataframe(config_df, use_container_width=True, hide_index=True)
