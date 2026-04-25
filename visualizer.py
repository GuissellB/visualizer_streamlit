# Desarrollado por: Guissell Betancur Oviedo y Anyelin Arias Camacho
# Curso: Minería de Datos Avanzada

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_auc_score, roc_curve


class Visualizer:
    """Crea figuras Plotly reutilizables para las apps de Streamlit."""

    def _base_layout(
        self,
        fig: go.Figure,
        *,
        height: int,
        margin: Optional[Dict[str, int]] = None,
        paper_bgcolor: str = "rgba(0,0,0,0)",
        plot_bgcolor: str = "rgba(248,250,252,0.92)",
        font_color: str = "#1e293b",
        showlegend: Optional[bool] = None,
    ) -> go.Figure:
        fig.update_layout(
            height=height,
            margin=margin or dict(l=10, r=10, t=10, b=10),
            paper_bgcolor=paper_bgcolor,
            plot_bgcolor=plot_bgcolor,
            font=dict(color=font_color),
        )
        if showlegend is not None:
            fig.update_layout(showlegend=showlegend)
        return fig

    def horizontal_bar(
        self,
        df: pd.DataFrame,
        *,
        x: str,
        y: str,
        color: Optional[str] = None,
        text: Optional[Sequence[str]] = None,
        color_scale: Optional[Sequence[str]] = None,
        height: int = 390,
        margin: Optional[Dict[str, int]] = None,
        x_title: str = "",
        y_title: str = "",
        x_range: Optional[Sequence[float]] = None,
        show_color_scale: bool = False,
        x_gridcolor: str = "rgba(148,163,184,0.18)",
    ) -> go.Figure:
        fig = px.bar(
            df,
            x=x,
            y=y,
            orientation="h",
            text=text,
            color=color,
            color_continuous_scale=color_scale,
        )
        fig.update_traces(
            textposition="inside",
            marker_line_width=0,
            cliponaxis=False,
            insidetextanchor="end",
        )
        fig.update_xaxes(
            title=x_title,
            range=x_range,
            showgrid=True,
            gridcolor=x_gridcolor,
            zeroline=False,
        )
        fig.update_yaxes(title=y_title, showgrid=False, automargin=True)
        fig.update_layout(coloraxis_showscale=show_color_scale)
        return self._base_layout(fig, height=height, margin=margin or dict(l=20, r=20, t=10, b=20))

    def donut_chart(
        self,
        df: pd.DataFrame,
        *,
        names: str,
        values: str,
        color: Optional[str] = None,
        color_map: Optional[Dict[str, str]] = None,
        hole: float = 0.68,
        height: int = 390,
        margin: Optional[Dict[str, int]] = None,
    ) -> go.Figure:
        fig = px.pie(
            df,
            names=names,
            values=values,
            hole=hole,
            color=color,
            color_discrete_map=color_map,
        )
        fig.update_traces(textinfo="percent+label", textposition="inside")
        fig.update_layout(
            legend=dict(orientation="h", y=-0.08, x=0.5, xanchor="center"),
            uniformtext_minsize=12,
            uniformtext_mode="hide",
        )
        return self._base_layout(fig, height=height, margin=margin or dict(l=15, r=15, t=10, b=15))

    def target_distribution_donut(
        self,
        df: pd.DataFrame,
        *,
        target_col: str,
        label_map: Optional[Dict[Any, str]] = None,
        color_map: Optional[Dict[str, str]] = None,
        hole: float = 0.68,
        height: int = 390,
        margin: Optional[Dict[str, int]] = None,
    ) -> Optional[go.Figure]:
        if target_col not in df.columns:
            return None

        pie_df = (
            df[target_col]
            .value_counts()
            .rename_axis("target")
            .reset_index(name="cantidad")
        )
        label_map = label_map or {1: "Legítimo", -1: "Phishing"}
        pie_df["tipo"] = pie_df["target"].map(label_map).fillna(pie_df["target"].astype(str))
        return self.donut_chart(
            pie_df,
            names="tipo",
            values="cantidad",
            color="tipo",
            color_map=color_map or {"Legítimo": "#2563eb", "Phishing": "#7c3aed"},
            hole=hole,
            height=height,
            margin=margin or dict(l=15, r=15, t=10, b=15),
        )

    def roc_curve_plot(
        self,
        roc_df: pd.DataFrame,
        *,
        auc_value: Optional[float] = None,
        curve_name: str = "ROC Curve",
        line_color: str = "#2563eb",
        random_name: str = "Random Classifier",
        height: int = 380,
        margin: Optional[Dict[str, int]] = None,
        x_title: str = "Tasa de Falsos Positivos (FPR)",
        y_title: str = "Tasa de Verdaderos Positivos (TPR)",
    ) -> go.Figure:
        fig = go.Figure()
        trace_name = curve_name if auc_value is None else f"{curve_name} (AUC = {auc_value:.4f})"
        fig.add_trace(
            go.Scatter(
                x=roc_df["fpr"],
                y=roc_df["tpr"],
                mode="lines",
                name=trace_name,
                line=dict(width=4, color=line_color),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name=random_name,
                line=dict(width=2, dash="dash", color="rgba(100,116,139,0.8)"),
            )
        )
        fig.update_xaxes(title=x_title, gridcolor="rgba(148,163,184,0.18)")
        fig.update_yaxes(title=y_title, gridcolor="rgba(148,163,184,0.18)")
        return self._base_layout(fig, height=height, margin=margin or dict(l=10, r=10, t=10, b=10))

    def precision_recall_plot(
        self,
        pr_df: pd.DataFrame,
        *,
        curve_name: str = "PR Curve",
        line_color: str = "#10b981",
        fillcolor: str = "rgba(16,185,129,0.12)",
        height: int = 380,
        margin: Optional[Dict[str, int]] = None,
        x_title: str = "Recall",
        y_title: str = "Precision",
    ) -> go.Figure:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=pr_df["recall"],
                y=pr_df["precision"],
                mode="lines",
                name=curve_name,
                line=dict(width=4, color=line_color),
                fill="tozeroy",
                fillcolor=fillcolor,
            )
        )
        fig.update_xaxes(title=x_title, gridcolor="rgba(148,163,184,0.18)")
        fig.update_yaxes(title=y_title, gridcolor="rgba(148,163,184,0.18)")
        return self._base_layout(fig, height=height, margin=margin or dict(l=10, r=10, t=10, b=10))

    def line_chart(
        self,
        df: pd.DataFrame,
        *,
        x: str,
        y: str,
        height: int = 360,
        margin: Optional[Dict[str, int]] = None,
    ) -> go.Figure:
        fig = px.line(df, x=x, y=y)
        return self._base_layout(fig, height=height, margin=margin or dict(l=10, r=10, t=20, b=10))

    def grouped_bar_chart(
        self,
        df: pd.DataFrame,
        *,
        x: str,
        y: str,
        color: str,
        barmode: str = "group",
        height: int = 380,
        margin: Optional[Dict[str, int]] = None,
        showlegend: Optional[bool] = None,
    ) -> go.Figure:
        fig = px.bar(df, x=x, y=y, color=color, barmode=barmode)
        return self._base_layout(
            fig,
            height=height,
            margin=margin or dict(l=10, r=10, t=20, b=10),
            showlegend=showlegend,
        )

    def multi_line_chart(
        self,
        series: Sequence[Dict[str, Any]],
        *,
        height: int = 380,
        margin: Optional[Dict[str, int]] = None,
    ) -> go.Figure:
        fig = go.Figure()
        for item in series:
            fig.add_trace(
                go.Scatter(
                    x=item["x"],
                    y=item["y"],
                    mode=item.get("mode", "lines"),
                    name=item["name"],
                    line=item.get("line"),
                )
            )
        return self._base_layout(fig, height=height, margin=margin or dict(l=10, r=10, t=20, b=10))

    def correlation_heatmap(
        self,
        df: pd.DataFrame,
        *,
        height: int = 720,
    ) -> Optional[go.Figure]:
        corr = df.corr(numeric_only=True)
        if corr.empty:
            return None

        fig = go.Figure(
            data=go.Heatmap(
                z=corr.values,
                x=corr.columns,
                y=corr.index,
                zmin=-1,
                zmax=1,
                colorscale="RdBu",
                reversescale=True,
                text=np.round(corr.values, 2),
                texttemplate="%{text}",
                hovertemplate="X: %{x}<br>Y: %{y}<br>Correlación: %{z:.2f}<extra></extra>",
            )
        )
        fig.update_xaxes(side="bottom")
        fig.update_yaxes(autorange="reversed")
        return self._base_layout(
            fig,
            height=height,
            margin=dict(l=10, r=10, t=40, b=10),
            plot_bgcolor="white",
        )

    def eda_histogramaClase(self, df: pd.DataFrame, target_col: str) -> Optional[go.Figure]:
        if target_col not in df.columns:
            return None

        counts = df[target_col].value_counts().sort_index().reset_index()
        counts.columns = [target_col, "Frecuencia"]
        fig = px.bar(counts, x=target_col, y="Frecuencia", color=target_col, text="Frecuencia")
        fig.update_traces(marker_line_width=0)
        fig.update_xaxes(title=target_col)
        fig.update_yaxes(title="Frecuencia", gridcolor="rgba(148,163,184,0.18)")
        return self._base_layout(fig, height=360, margin=dict(l=10, r=10, t=20, b=10), showlegend=False)

    def eda_graficoCorrelacionTarget(self, df: pd.DataFrame, target_col: str) -> Optional[go.Figure]:
        if target_col not in df.columns:
            return None

        corr = (
            df.corr(numeric_only=True)[target_col]
            .drop(target_col)
            .sort_values(ascending=False)
            .reset_index()
        )
        corr.columns = ["feature", "correlation"]
        sorted_corr = corr.sort_values("correlation")
        return self.horizontal_bar(
            sorted_corr,
            x="correlation",
            y="feature",
            color="correlation",
            text=sorted_corr["correlation"].map(lambda value: f"{value:.2f}"),
            color_scale=["#60a5fa", "#7c3aed"],
            height=max(360, len(corr) * 28),
            x_title="Correlation Coefficient",
            y_title="Predictor Variable",
            x_range=[
                float(corr["correlation"].min()) * 1.1 if len(corr) else -1,
                float(corr["correlation"].max()) * 1.1 if len(corr) else 1,
            ],
        )

    def top_target_correlation_bar(
        self,
        df: pd.DataFrame,
        *,
        target_col: str,
        top_n: int = 10,
        absolute: bool = True,
        height: int = 390,
        margin: Optional[Dict[str, int]] = None,
        x_title: str = "Correlación absoluta",
        y_title: str = "",
        color_scale: Optional[Sequence[str]] = None,
    ) -> Optional[go.Figure]:
        if target_col not in df.columns:
            return None

        corr = (
            df.corr(numeric_only=True)[target_col]
            .drop(target_col, errors="ignore")
        )
        if absolute:
            corr = corr.abs()

        corr_df = (
            corr.sort_values(ascending=False)
            .head(top_n)
            .reset_index()
        )
        corr_df.columns = ["feature", "importance"]
        sorted_corr_df = corr_df.sort_values("importance")
        max_value = float(corr_df["importance"].max()) if len(corr_df) else 1.0

        return self.horizontal_bar(
            sorted_corr_df,
            x="importance",
            y="feature",
            color="importance",
            text=sorted_corr_df["importance"].map(lambda value: f"{value:.2f}"),
            color_scale=color_scale or ["#60a5fa", "#7c3aed"],
            height=height,
            margin=margin or dict(l=20, r=20, t=10, b=20),
            x_title=x_title,
            y_title=y_title,
            x_range=[0, max_value * 1.05],
        )

    def eda_graficoCorrelacion(self, df: pd.DataFrame) -> Optional[go.Figure]:
        return self.correlation_heatmap(df)

    def sup_plot_roc(self, y_true: Sequence, y_score: Sequence, label: str = "Modelo", title: str = "Curva ROC") -> Optional[go.Figure]:
        y_true = np.asarray(y_true)
        y_score = np.asarray(y_score)
        if y_true.shape[0] != y_score.shape[0] or y_true.shape[0] == 0:
            return None

        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc_score = roc_auc_score(y_true, y_score)
        fig = self.roc_curve_plot(
            pd.DataFrame({"fpr": fpr, "tpr": tpr}),
            auc_value=auc_score,
            curve_name=label,
            x_title="False Positive Rate",
            y_title="True Positive Rate",
        )
        fig.update_layout(title=title)
        return fig

    def sup_plot_roc_compare(
        self,
        curves: Dict[str, Tuple[Sequence, Sequence]],
        title: str = "Comparación Curva ROC",
    ) -> Optional[go.Figure]:
        if not curves:
            return None

        fig = go.Figure()
        plotted = 0
        for name, pair in curves.items():
            if pair is None or len(pair) != 2:
                continue
            y_true, y_score = pair
            y_true = np.asarray(y_true)
            y_score = np.asarray(y_score)
            if y_true.shape[0] != y_score.shape[0] or y_true.shape[0] == 0:
                continue

            fpr, tpr, _ = roc_curve(y_true, y_score)
            auc_score = roc_auc_score(y_true, y_score)
            fig.add_trace(
                go.Scatter(
                    x=fpr,
                    y=tpr,
                    mode="lines",
                    name=f"{name} (AUC = {auc_score:.4f})",
                    line=dict(width=3),
                )
            )
            plotted += 1

        if plotted == 0:
            return None

        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Azar",
                line=dict(width=2, dash="dash", color="rgba(100,116,139,0.8)"),
            )
        )
        fig.update_layout(title=title)
        fig.update_xaxes(title="False Positive Rate", gridcolor="rgba(148,163,184,0.18)")
        fig.update_yaxes(title="True Positive Rate", gridcolor="rgba(148,163,184,0.18)")
        return self._base_layout(fig, height=420, margin=dict(l=10, r=10, t=40, b=10))

    def results_to_df(
        self,
        resultados: Union[List[Dict[str, Any]], Dict[str, Any]],
        sort_by: Optional[str] = None,
        ascending: bool = False,
    ) -> pd.DataFrame:
        if isinstance(resultados, dict):
            return self.metrics_dict_to_df(resultados)

        df = pd.DataFrame(resultados)
        if sort_by and sort_by in df.columns:
            df = df.sort_values(sort_by, ascending=ascending).reset_index(drop=True)
        return df

    def metrics_dict_to_df(self, metrics: Dict[str, Any]) -> pd.DataFrame:
        rows = []
        for key, value in metrics.items():
            if isinstance(value, (np.ndarray, list)) and np.asarray(value).ndim == 2:
                continue
            rows.append({"Metrica": key, "Valor": value})
        return pd.DataFrame(rows)

    def confusion_matrix_df(self, metrics: Dict[str, Any], labels: Optional[Sequence[Any]] = None) -> Optional[pd.DataFrame]:
        cm = metrics.get("ConfusionMatrix")
        if cm is None:
            return None
        cm = np.asarray(cm)
        if cm.ndim != 2:
            return None

        if labels is None:
            labels = list(range(cm.shape[0]))
        return pd.DataFrame(cm, index=[f"Real_{label}" for label in labels], columns=[f"Pred_{label}" for label in labels])
