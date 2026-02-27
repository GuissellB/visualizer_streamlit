import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram


class Visualizer:
    """Solo visualización + tablas para notebooks/Streamlit.

    - NO entrena modelos
    - NO hace splits
    - NO modifica dataframes originales
    - Retorna figuras (matplotlib) y DataFrames listos para mostrar
    """

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------
    @staticmethod
    def _grid(n_plots: int, n_cols: int = 3, w: float = 5, h: float = 4, dpi: int = 150):
        n_rows = int(math.ceil(n_plots / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(w * n_cols, h * n_rows), dpi=dpi)
        return fig, np.array(axes).flatten()

    @staticmethod
    def _numeric_cols(df: pd.DataFrame) -> List[str]:
        return df.select_dtypes(include="number").columns.tolist()

    # ---------------------------------------------------------------------
    # EDA (desde EDAExplorer)
    # ---------------------------------------------------------------------
    @staticmethod
    def eda_boxplots(df: pd.DataFrame):
        cols = Visualizer._numeric_cols(df)
        if not cols:
            return None

        fig, axes = Visualizer._grid(len(cols))
        colores = sns.color_palette("Set3", len(cols))
        for i, col in enumerate(cols):
            sns.boxplot(y=df[col], ax=axes[i], color=colores[i])
            axes[i].set_title(f"Boxplot: {col}", fontsize=10)
            axes[i].grid(True, linestyle="--", alpha=0.5)

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        fig.tight_layout()
        return fig

    @staticmethod
    def eda_histogramas(df: pd.DataFrame):
        cols = Visualizer._numeric_cols(df)
        if not cols:
            return None

        fig, axes = Visualizer._grid(len(cols))
        colores = sns.color_palette("Set2", len(cols))
        for i, col in enumerate(cols):
            axes[i].hist(df[col].dropna(), bins=30, edgecolor="black", alpha=0.7, color=colores[i])
            axes[i].set_title(f"Hist: {col}", fontsize=10)
            axes[i].grid(True, linestyle="--", alpha=0.5)

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        fig.tight_layout()
        return fig

    @staticmethod
    def eda_distribuciones(df: pd.DataFrame):
        cols = Visualizer._numeric_cols(df)
        if not cols:
            return None

        fig, axes = Visualizer._grid(len(cols))
        colores = sns.color_palette("coolwarm", len(cols))
        for i, col in enumerate(cols):
            sns.histplot(df[col].dropna(), kde=True, ax=axes[i], bins=30, color=colores[i])
            axes[i].set_title(f"Dist: {col}", fontsize=10)
            axes[i].grid(True, linestyle="--", alpha=0.5)

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        fig.tight_layout()
        return fig

    @staticmethod
    def eda_histogramaClase(df: pd.DataFrame, target_col: str):
        if target_col not in df.columns:
            return None

        fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
        colores = sns.color_palette("pastel")
        df[target_col].value_counts().plot(kind="bar", color=colores, ax=ax)
        ax.set_title(f"Distribución de la Clase: {target_col}")
        ax.set_xlabel(target_col)
        ax.set_ylabel("Frecuencia")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        fig.tight_layout()
        return fig

    @staticmethod
    def eda_densidades(df: pd.DataFrame):
        cols = Visualizer._numeric_cols(df)
        if not cols:
            return None

        fig, axes = Visualizer._grid(len(cols))
        colores = sns.color_palette("husl", len(cols))
        for i, col in enumerate(cols):
            sns.kdeplot(data=df, x=col, fill=True, ax=axes[i], linewidth=2, color=colores[i])
            axes[i].set_title(f"Densidad: {col}", fontsize=10)
            axes[i].grid(True, linestyle="--", alpha=0.5)

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        fig.tight_layout()
        return fig

    @staticmethod
    def eda_graficoCorrelacion(df: pd.DataFrame, figsize: Tuple[int, int] = (20, 20)):
        corr = df.corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=figsize, dpi=150)
        cmap = sns.diverging_palette(240, 10, as_cmap=True).reversed()
        sns.heatmap(
            corr,
            vmin=-1,
            vmax=1,
            cmap=cmap,
            annot=True,
            fmt=".2f",
            linewidths=0.5,
            linecolor="white",
            square=True,
            cbar_kws={"shrink": 0.8, "label": "Correlación"},
            annot_kws={"size": 10, "color": "black"},
            ax=ax,
        )
        ax.set_title("Mapa de Calor de Correlaciones", fontsize=16)
        ax.tick_params(axis="x", rotation=45)
        ax.tick_params(axis="y", rotation=0)
        fig.tight_layout()
        return fig

    @staticmethod
    def eda_graficoCorrelacionTarget(df: pd.DataFrame, target_col: str):
        if target_col not in df.columns:
            return None

        corr = df.corr(numeric_only=True)[target_col].drop(target_col).sort_values(ascending=False)
        fig_h = max(4, corr.shape[0] * 0.35)
        fig, ax = plt.subplots(figsize=(10, fig_h))
        sns.barplot(x=corr.values, y=corr.index, palette="vlag", orient="h", ax=ax)

        ax.axvline(0, color="gray", linestyle="--", linewidth=1)
        offset = corr.abs().max() * 0.02 if len(corr) else 0.0
        for i, v in enumerate(corr.values):
            ax.text(
                v + offset if v >= 0 else v - offset,
                i,
                f"{v:.2f}",
                va="center",
                ha="left" if v >= 0 else "right",
                fontsize=9,
            )

        ax.set_title(f"Correlation of Features with {target_col}", fontsize=14)
        ax.set_xlabel("Correlation Coefficient", fontsize=12)
        ax.set_ylabel("Predictor Variable", fontsize=12)
        sns.despine(left=False, bottom=False)
        fig.tight_layout()
        return fig

    @staticmethod
    def eda_pairplot(df: pd.DataFrame):
        cols = Visualizer._numeric_cols(df)
        if len(cols) < 2:
            return None
        g = sns.pairplot(df[cols])
        return g.fig

    # ---------------------------------------------------------------------
    # UnsupervisedRunner plots (mismos nombres, pero retornando fig)
    # ---------------------------------------------------------------------
    @staticmethod
    def unsup_plot_clusters(runner, title: Optional[str] = None, use_pca: bool = True):
        if runner.labels_ is None:
            return None
        coords = PCA(n_components=2, random_state=42).fit_transform(runner.X) if use_pca else runner.embedding_
        if coords is None:
            return None

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(coords[:, 0], coords[:, 1], c=runner.labels_, s=20, alpha=0.7)
        ax.set_title(title or f"Clusters - {runner.name}")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        return fig

    @staticmethod
    def unsup_graficar_2d(runner, color=None, titulo: Optional[str] = None):
        if runner.embedding_ is None:
            runner.embedding_ = PCA(n_components=2, random_state=0).fit_transform(runner.X)

        c = runner.labels_ if (color is None and runner.labels_ is not None) else color
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(runner.embedding_[:, 0], runner.embedding_[:, 1], c=c, s=15, alpha=0.7)
        ax.set_title(titulo or runner.name)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        return fig

    @staticmethod
    def unsup_plot_scree(runner):
        if runner.kind != "pca" or not hasattr(runner.model, "explained_variance_ratio_"):
            return None

        var = runner.model.explained_variance_ratio_ * 100
        comps = np.arange(1, len(var) + 1)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(comps, var, alpha=0.6)
        ax.plot(comps, var, marker="o")
        ax.set_xlabel("Componente principal")
        ax.set_ylabel("% Varianza explicada")
        ax.set_title(f"Scree plot - {runner.name}")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        return fig

    @staticmethod
    def unsup_plot_correlation_circle(runner, feature_names: Optional[Sequence[str]] = None, scale: float = 1.0):
        if runner.kind != "pca" or not hasattr(runner.model, "components_"):
            return None

        comps = runner.model.components_
        if comps.shape[0] < 2:
            return None

        if feature_names is None:
            feature_names = list(runner.X.columns)

        pc1, pc2 = comps[0], comps[1]
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.add_artist(plt.Circle((0, 0), 1, color="lightgray", fill=False))

        for x, y, lab in zip(pc1, pc2, feature_names):
            ax.arrow(0, 0, x * scale, y * scale, head_width=0.03, head_length=0.03, alpha=0.6)
            ax.text(x * 1.1 * scale, y * 1.1 * scale, lab, ha="center", va="center", fontsize=9)

        ax.axhline(0, color="grey", lw=1)
        ax.axvline(0, color="grey", lw=1)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(f"Círculo de correlación - {runner.name}")
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_aspect("equal", "box")
        fig.tight_layout()
        return fig

    @staticmethod
    def unsup_plot_embedding(runner, color=None, titulo: Optional[str] = None):
        if runner.embedding_ is None:
            return None
        if color is None and runner.labels_ is not None:
            color = runner.labels_

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(runner.embedding_[:, 0], runner.embedding_[:, 1], c=color, s=20, alpha=0.8)
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")
        ax.set_title(titulo if titulo is not None else f"{runner.kind.upper()} - {runner.name}")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        return fig

    @staticmethod
    def unsup_plot_cluster_scatter(runner, usar_pca: bool = True, titulo: Optional[str] = None):
        if runner.labels_ is None:
            return None

        if usar_pca:
            coords = PCA(n_components=2, random_state=42).fit_transform(runner.X)
        else:
            if runner.embedding_ is None:
                return None
            coords = runner.embedding_

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(coords[:, 0], coords[:, 1], c=runner.labels_, s=20, alpha=0.8)
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")
        ax.set_title(titulo if titulo else f"Clusters - {runner.name}")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        return fig

    @staticmethod
    def unsup_plot_embedding_by_clusters(runner, labels=None, titulo: Optional[str] = None):
        if runner.embedding_ is None:
            return None
        if labels is None:
            labels = runner.labels_
        if labels is None:
            return None
        if titulo is None:
            titulo = f"{runner.kind.upper()} - coloreado por clusters"
        return Visualizer.unsup_plot_embedding(runner, color=labels, titulo=titulo)

    @staticmethod
    def unsup_plot_embedding_by_variable(runner, serie, titulo: Optional[str] = None):
        if runner.embedding_ is None:
            return None
        if isinstance(serie, pd.DataFrame):
            serie = serie.iloc[:, 0]
        if len(serie) != runner.embedding_.shape[0]:
            return None
        if titulo is None:
            titulo = f"{runner.kind.upper()} - coloreado por variable"
        return Visualizer.unsup_plot_embedding(runner, color=serie, titulo=titulo)

    @staticmethod
    def unsup_plot_centroids(runner, feature_names: Optional[Sequence[str]] = None, scale: bool = False):
        if runner.kind != "kmeans" or not hasattr(runner.model, "cluster_centers_"):
            return None

        centers = runner.model.cluster_centers_.copy()
        n_clusters, n_features = centers.shape

        if scale:
            max_abs = np.max(np.abs(centers), axis=0)
            max_abs[max_abs == 0] = 1
            centers = centers / max_abs

        if feature_names is None:
            feature_names = list(runner.X.columns)

        fig, axes = plt.subplots(1, n_clusters, figsize=(4 * n_clusters, 4), sharey=True)
        if n_clusters == 1:
            axes = [axes]

        for k, ax in enumerate(axes):
            ax.barh(np.arange(n_features), centers[k])
            ax.set_yticks(np.arange(n_features))
            ax.set_yticklabels(feature_names)
            ax.set_title(f"Cluster {k}")
            ax.grid(axis="x", alpha=0.3)

        fig.tight_layout()
        return fig

    @staticmethod
    def unsup_plot_dendrogram(runner, method: str = "ward", p: int = 30):
        Z = linkage(runner.X, method=method, metric="euclidean")
        fig, ax = plt.subplots(figsize=(10, 5))
        dendrogram(
            Z,
            truncate_mode="lastp",
            p=p,
            show_leaf_counts=True,
            leaf_rotation=45,
            leaf_font_size=10,
            ax=ax,
        )
        ax.set_title(f"Dendrograma ({method}) - {runner.name} | lastp={p}")
        ax.set_xlabel("Clusters (truncado)")
        ax.set_ylabel("Distancia")
        fig.tight_layout()
        return fig

    # ---------------------------------------------------------------------
    # Tablas de resultados (para reemplazar prints del notebook)
    # ---------------------------------------------------------------------
    @staticmethod
    def results_to_df(resultados: Union[List[Dict[str, Any]], Dict[str, Any]], sort_by: Optional[str] = None, ascending: bool = False) -> pd.DataFrame:
        """Convierte resultados a DataFrame.

        - Si recibes una lista de dicts (como tu notebook), te arma la tabla completa.
        - Si recibes un dict (un solo modelo), te arma una tabla 2 columnas: Métrica / Valor.
        """
        if isinstance(resultados, dict):
            return Visualizer.metrics_dict_to_df(resultados)

        df = pd.DataFrame(resultados)
        if sort_by and sort_by in df.columns:
            df = df.sort_values(sort_by, ascending=ascending).reset_index(drop=True)
        return df

    @staticmethod
    def metrics_dict_to_df(metrics: Dict[str, Any]) -> pd.DataFrame:
        rows = []
        for k, v in metrics.items():
            if isinstance(v, (np.ndarray, list)) and np.asarray(v).ndim == 2:
                # matrices (ej: ConfusionMatrix) se dejan fuera para otra visualización
                continue
            rows.append({"Metrica": k, "Valor": v})
        return pd.DataFrame(rows)

    @staticmethod
    def confusion_matrix_df(metrics: Dict[str, Any], labels: Optional[Sequence[Any]] = None) -> Optional[pd.DataFrame]:
        cm = metrics.get("ConfusionMatrix", None)
        if cm is None:
            return None
        cm = np.asarray(cm)
        if cm.ndim != 2:
            return None

        if labels is None:
            labels = list(range(cm.shape[0]))
        return pd.DataFrame(cm, index=[f"Real_{l}" for l in labels], columns=[f"Pred_{l}" for l in labels])
