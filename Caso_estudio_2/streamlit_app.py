from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


# =========================================================
# CONFIGURACIÓN GENERAL
# =========================================================
st.set_page_config(
    page_title="Caso de Estudio | Redes Neuronales y Reglas de Asociación",
    page_icon="🧠",
    layout="wide",
)


# =========================================================
# ESTILOS
# =========================================================
CUSTOM_CSS = """
<style>
    .main > div {
        padding-top: 1.2rem;
    }
    .hero-card {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: white;
        padding: 1.4rem 1.6rem;
        border-radius: 22px;
        margin-bottom: 1rem;
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.18);
    }
    .hero-title {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .hero-subtitle {
        font-size: 1rem;
        opacity: 0.9;
    }
    .soft-card {
        background: #ffffff;
        border: 1px solid rgba(15, 23, 42, 0.08);
        border-radius: 18px;
        padding: 1rem 1rem;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
        margin-bottom: 1rem;
    }
    .section-title {
        font-size: 1.15rem;
        font-weight: 700;
        margin-bottom: 0.75rem;
    }
    .small-note {
        color: #475569;
        font-size: 0.92rem;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# =========================================================
# HELPERS
# =========================================================
def normalize_path(base_dir: Path, value: str | Path) -> Path:
    p = Path(value)

    if p.is_absolute():
        return p

    # intento 1: relativo al manifest
    candidate_1 = (base_dir / p).resolve()
    if candidate_1.exists():
        return candidate_1

    # intento 2: relativo a la carpeta padre del manifest (raíz del proyecto)
    candidate_2 = (base_dir.parent / p).resolve()
    if candidate_2.exists():
        return candidate_2

    # intento 3: relativo al directorio actual
    candidate_3 = p.resolve()
    if candidate_3.exists():
        return candidate_3

    # si no existe aún, devolver el más razonable
    return candidate_2


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_manifest(manifest_path_str: str) -> Dict[str, Any]:
    manifest_path = Path(manifest_path_str).resolve()
    manifest = load_json(manifest_path)
    manifest["_manifest_path"] = str(manifest_path)
    manifest["_base_dir"] = str(manifest_path.parent)
    return manifest


@st.cache_data(show_spinner=False)
def load_csv(path_str: str) -> pd.DataFrame:
    if not path_str:
        return pd.DataFrame()

    path = Path(path_str)
    if not path.exists():
        st.warning(f"No se encontró el archivo: {path}")
        return pd.DataFrame()

    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_optional_json(path_str: str) -> Dict[str, Any]:
    if not path_str:
        return {}

    path = Path(path_str)
    if not path.exists():
        st.warning(f"No se encontró el JSON: {path}")
        return {}

    return load_json(path)


@st.cache_data(show_spinner=False)
def prepare_rule_length_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    work = df.copy()
    if "antecedents" in work.columns:
        work["antecedent_len"] = work["antecedents"].fillna("").apply(
            lambda x: 0 if str(x).strip() == "" else len([i for i in str(x).split(",") if i.strip()])
        )
    if "consequents" in work.columns:
        work["consequent_len"] = work["consequents"].fillna("").apply(
            lambda x: 0 if str(x).strip() == "" else len([i for i in str(x).split(",") if i.strip()])
        )
    return work


@st.cache_data(show_spinner=False)
def prepare_nn_long_format(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    metric_cols = [c for c in df.columns if c != "model_name"]
    long_df = df.melt(id_vars="model_name", value_vars=metric_cols, var_name="metric", value_name="value")
    long_df = long_df[pd.to_numeric(long_df["value"], errors="coerce").notna()].copy()
    long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce")
    return long_df


def infer_best_metric(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        "F1_Pos",
        "Accuracy",
        "ROC_AUC_Pos",
        "Recall_Pos",
        "Precision_Pos",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    numeric_cols = [c for c in df.columns if c != "model_name" and pd.api.types.is_numeric_dtype(df[c])]
    return numeric_cols[0] if numeric_cols else None


def section_card(title: str):
    container = st.container(border=True)
    container.markdown(f"### {title}")
    return container


# =========================================================
# CARGA DE DATOS
# =========================================================
def load_project_bundle(manifest_path_str: str) -> Dict[str, Any]:
    manifest = load_manifest(manifest_path_str)
    base_dir = Path(manifest["_base_dir"])
    files = manifest.get("files", {})

    resolved_files = {k: str(normalize_path(base_dir, v)) for k, v in files.items()}

    bundle = {
        "manifest": manifest,
        "files": resolved_files,
        "products": load_csv(resolved_files.get("products_csv", "")),
        "rules": load_csv(resolved_files.get("association_rules_csv", "")),
        "itemsets": load_csv(resolved_files.get("association_itemsets_csv", "")),
        "top_items": load_csv(resolved_files.get("association_top_items_csv", "")),
        "transactions": load_csv(resolved_files.get("association_transactions_csv", "")),
        "nn_holdout": load_csv(resolved_files.get("nn_holdout_results_csv", "")),
        "nn_cv": load_csv(resolved_files.get("nn_cv_results_csv", "")),
        "nn_dataset": load_csv(resolved_files.get("nn_dataset_csv", "")),
        "nn_confusion": load_optional_json(resolved_files.get("nn_confusion_matrices_json", "")),
        "nn_architectures": load_optional_json(resolved_files.get("nn_architectures_json", "")),
    }
    return bundle


# =========================================================
# UI SIDEBAR
# =========================================================
st.sidebar.title("Configuración")
manifest_default = "outputs/00_manifest.json"
manifest_path = st.sidebar.text_input(
    "Ruta del manifest.json",
    value=manifest_default,
    help="Ruta al archivo 00_manifest.json generado por tu pipeline.",
)

show_raw_tables = st.sidebar.toggle("Mostrar tablas completas", value=False)
show_download_buttons = st.sidebar.toggle("Mostrar botones de descarga", value=True)
page = st.sidebar.radio(
    "Sección",
    ["Resumen", "Redes neuronales", "Reglas de asociación", "Productos"],
)

try:
    bundle = load_project_bundle(manifest_path)
    manifest = bundle["manifest"]
    products_df = bundle["products"]
    rules_df = prepare_rule_length_columns(bundle["rules"])
    itemsets_df = bundle["itemsets"]
    top_items_df = bundle["top_items"]
    nn_holdout_df = bundle["nn_holdout"]
    nn_cv_df = bundle["nn_cv"]
    nn_dataset_df = bundle["nn_dataset"]
    confusion_payload = bundle["nn_confusion"]
    architecture_payload = bundle["nn_architectures"]
except Exception as exc:
    st.error(f"No se pudieron cargar los archivos del proyecto: {exc}")
    st.stop()


# =========================================================
# HEADER
# =========================================================
st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">Caso de estudio integrado</div>
        <div class="hero-subtitle">
            Visualización de resultados para redes neuronales y reglas de asociación,
            construida para consumir directamente los archivos exportados por tu pipeline.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

row_counts = manifest.get("row_counts", {})
metric_cols = st.columns(5)
metric_cols[0].metric("Productos scrapeados", row_counts.get("raw_products", 0))
metric_cols[1].metric("Productos limpios", row_counts.get("products_clean", 0))
metric_cols[2].metric("Transacciones", row_counts.get("transactions", 0))
metric_cols[3].metric("Reglas", row_counts.get("association_rules", 0))
metric_cols[4].metric("Dataset NN", row_counts.get("nn_dataset", 0))

with st.expander("Ver metadata del experimento"):
    st.json(
        {
            "scenario": manifest.get("scenario"),
            "sources": manifest.get("sources", []),
            "min_support": manifest.get("min_support"),
            "min_confidence": manifest.get("min_confidence"),
            "random_state": manifest.get("random_state"),
            "keywords": manifest.get("keywords", []),
            "price_thresholds": manifest.get("price_thresholds", {}),
        }
    )


# =========================================================
# RESUMEN
# =========================================================
if page == "Resumen":
    col1, col2 = st.columns([1.2, 1])

    with col1:
        card = section_card("Distribución de segmentos de precio")
        if not products_df.empty and "price_segment" in products_df.columns:
            seg_counts = products_df["price_segment"].value_counts(dropna=False).reset_index()
            seg_counts.columns = ["segmento", "cantidad"]
            fig_seg = px.bar(seg_counts, x="segmento", y="cantidad", text="cantidad")
            fig_seg.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10))
            card.plotly_chart(fig_seg, use_container_width=True)
        else:
            card.info("No hay datos de productos para mostrar la distribución de segmentos.")

    with col2:
        card = section_card("Top items frecuentes")
        if not top_items_df.empty:
            top_n = card.slider("Cantidad de items a mostrar", 5, 20, 10, key="summary_top_n")
            subset = top_items_df.head(top_n)
            fig_top_items = px.bar(
                subset.sort_values("support_proxy", ascending=True),
                x="support_proxy",
                y="item",
                orientation="h",
                text="support_proxy",
            )
            fig_top_items.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10))
            card.plotly_chart(fig_top_items, use_container_width=True)
        else:
            card.info("No hay items frecuentes disponibles.")

    card = section_card("Comparación rápida de modelos")
    if not nn_holdout_df.empty:
        best_metric = infer_best_metric(nn_holdout_df)
        if best_metric:
            fig_models = px.bar(
                nn_holdout_df.sort_values(best_metric, ascending=False),
                x="model_name",
                y=best_metric,
                text=best_metric,
            )
            fig_models.update_layout(height=420, xaxis_title="Modelo", yaxis_title=best_metric)
            card.plotly_chart(fig_models, use_container_width=True)
        else:
            card.info("No se encontró una métrica numérica para comparar modelos.")
    else:
        card.info("No hay resultados holdout de redes neuronales.")


# =========================================================
# REDES NEURONALES
# =========================================================
elif page == "Redes neuronales":
    card = section_card("Resultados de holdout y validación cruzada")

    tab_holdout, tab_cv, tab_conf, tab_arch = card.tabs([
        "Holdout",
        "Validación cruzada",
        "Matrices de confusión",
        "Arquitecturas",
    ])

    with tab_holdout:
        if nn_holdout_df.empty:
            st.info("No hay resultados holdout disponibles.")
        else:
            long_df = prepare_nn_long_format(nn_holdout_df)
            available_metrics = sorted(long_df["metric"].unique().tolist()) if not long_df.empty else []
            default_metric = infer_best_metric(nn_holdout_df)
            metric_selected = st.selectbox(
                "Métrica a visualizar",
                options=available_metrics,
                index=available_metrics.index(default_metric) if default_metric in available_metrics else 0,
                key="holdout_metric_selector",
            ) if available_metrics else None

            if metric_selected:
                chart_df = long_df[long_df["metric"] == metric_selected].sort_values("value", ascending=False)
                fig = px.bar(chart_df, x="model_name", y="value", text="value")
                fig.update_layout(height=420, xaxis_title="Modelo", yaxis_title=metric_selected)
                st.plotly_chart(fig, use_container_width=True)

            st.dataframe(nn_holdout_df, use_container_width=True, height=320)
            if show_download_buttons:
                st.download_button(
                    "Descargar resultados holdout",
                    data=nn_holdout_df.to_csv(index=False).encode("utf-8-sig"),
                    file_name="nn_holdout_results.csv",
                    mime="text/csv",
                )

    with tab_cv:
        if nn_cv_df.empty:
            st.info("No hay resultados de validación cruzada disponibles.")
        else:
            long_df_cv = prepare_nn_long_format(nn_cv_df)
            available_metrics_cv = sorted(long_df_cv["metric"].unique().tolist()) if not long_df_cv.empty else []
            default_metric_cv = infer_best_metric(nn_cv_df)
            metric_selected_cv = st.selectbox(
                "Métrica de CV a visualizar",
                options=available_metrics_cv,
                index=available_metrics_cv.index(default_metric_cv) if default_metric_cv in available_metrics_cv else 0,
                key="cv_metric_selector",
            ) if available_metrics_cv else None

            if metric_selected_cv:
                chart_df_cv = long_df_cv[long_df_cv["metric"] == metric_selected_cv].sort_values("value", ascending=False)
                fig_cv = px.bar(chart_df_cv, x="model_name", y="value", text="value")
                fig_cv.update_layout(height=420, xaxis_title="Modelo", yaxis_title=metric_selected_cv)
                st.plotly_chart(fig_cv, use_container_width=True)

            st.dataframe(nn_cv_df, use_container_width=True, height=320)
            if show_download_buttons:
                st.download_button(
                    "Descargar resultados CV",
                    data=nn_cv_df.to_csv(index=False).encode("utf-8-sig"),
                    file_name="nn_cv_results.csv",
                    mime="text/csv",
                )

    with tab_conf:
        if not confusion_payload:
            st.info("No se encontraron matrices de confusión exportadas.")
        else:
            model_options = list(confusion_payload.keys())
            selected_model = st.selectbox("Modelo", model_options, key="conf_model_selector")
            matrix = np.array(confusion_payload[selected_model])
            fig_conf = go.Figure(
                data=go.Heatmap(
                    z=matrix,
                    text=matrix,
                    texttemplate="%{text}",
                    hovertemplate="Valor: %{z}<extra></extra>",
                )
            )
            fig_conf.update_layout(
                height=500,
                xaxis_title="Predicción",
                yaxis_title="Real",
                title=f"Matriz de confusión | {selected_model}",
            )
            st.plotly_chart(fig_conf, use_container_width=True)
            st.dataframe(pd.DataFrame(matrix), use_container_width=False)

    with tab_arch:
        if not architecture_payload:
            st.info("No se encontraron arquitecturas exportadas.")
        else:
            model_options = list(architecture_payload.keys())
            selected_model = st.selectbox("Arquitectura a inspeccionar", model_options, key="arch_model_selector")
            st.json(architecture_payload[selected_model])


# =========================================================
# REGLAS DE ASOCIACIÓN
# =========================================================
elif page == "Reglas de asociación":
    card = section_card("Explorador de reglas de asociación")

    with card:
        if rules_df.empty:
            st.info("No hay reglas de asociación disponibles.")
        else:
            numeric_rule_metrics = [
                c for c in rules_df.columns
                if pd.api.types.is_numeric_dtype(rules_df[c]) and c not in ["antecedent_len", "consequent_len"]
            ]

            c1, c2, c3, c4 = st.columns(4)
            min_conf = c1.slider(
                "Confianza mínima",
                min_value=float(rules_df["confidence"].min()) if "confidence" in rules_df.columns else 0.0,
                max_value=float(rules_df["confidence"].max()) if "confidence" in rules_df.columns else 1.0,
                value=float(rules_df["confidence"].quantile(0.50)) if "confidence" in rules_df.columns else 0.5,
                step=0.01,
            )
            min_support = c2.slider(
                "Soporte mínimo",
                min_value=float(rules_df["support"].min()) if "support" in rules_df.columns else 0.0,
                max_value=float(rules_df["support"].max()) if "support" in rules_df.columns else 1.0,
                value=float(rules_df["support"].quantile(0.30)) if "support" in rules_df.columns else 0.1,
                step=0.01,
            )
            max_antecedents = c3.slider("Máximo de items en antecedente", 1, int(max(rules_df.get("antecedent_len", pd.Series([1])).max(), 1)), 3)
            sort_metric = c4.selectbox(
                "Ordenar por",
                options=numeric_rule_metrics if numeric_rule_metrics else ["confidence"],
                index=0,
            )

            contains_text = st.text_input("Filtrar por texto contenido en antecedente o consecuente")
            top_n_rules = st.slider("Cantidad de reglas a mostrar", 5, 100, 25)

            filtered_rules = rules_df.copy()
            if "confidence" in filtered_rules.columns:
                filtered_rules = filtered_rules[filtered_rules["confidence"] >= min_conf]
            if "support" in filtered_rules.columns:
                filtered_rules = filtered_rules[filtered_rules["support"] >= min_support]
            if "antecedent_len" in filtered_rules.columns:
                filtered_rules = filtered_rules[filtered_rules["antecedent_len"] <= max_antecedents]
            if contains_text.strip():
                txt = contains_text.strip().lower()
                filtered_rules = filtered_rules[
                    filtered_rules["antecedents"].fillna("").str.lower().str.contains(txt)
                    | filtered_rules["consequents"].fillna("").str.lower().str.contains(txt)
                ]

            if sort_metric in filtered_rules.columns:
                filtered_rules = filtered_rules.sort_values(sort_metric, ascending=False)
            filtered_rules_show = filtered_rules.head(top_n_rules)

            kpi_cols = st.columns(4)
            kpi_cols[0].metric("Reglas filtradas", len(filtered_rules))
            if "confidence" in filtered_rules.columns and not filtered_rules.empty:
                kpi_cols[1].metric("Confianza promedio", f"{filtered_rules['confidence'].mean():.3f}")
            if "lift" in filtered_rules.columns and not filtered_rules.empty:
                kpi_cols[2].metric("Lift promedio", f"{filtered_rules['lift'].mean():.3f}")
            if "support" in filtered_rules.columns and not filtered_rules.empty:
                kpi_cols[3].metric("Soporte promedio", f"{filtered_rules['support'].mean():.3f}")

            vis_col1, vis_col2 = st.columns([1.1, 1])
            with vis_col1:
                if all(col in filtered_rules.columns for col in ["support", "confidence"]):
                    size_col = "lift" if "lift" in filtered_rules.columns else None
                    color_col = sort_metric if sort_metric in filtered_rules.columns else "confidence"
                    fig_rules = px.scatter(
                        filtered_rules_show,
                        x="support",
                        y="confidence",
                        size=size_col,
                        color=color_col,
                        hover_data=[c for c in ["antecedents", "consequents", "lift"] if c in filtered_rules_show.columns],
                    )
                    fig_rules.update_layout(height=430, title="Soporte vs confianza")
                    st.plotly_chart(fig_rules, use_container_width=True)
                else:
                    st.info("No hay columnas suficientes para graficar soporte y confianza.")

            with vis_col2:
                if not top_items_df.empty:
                    top_n = st.slider("Top items frecuentes", 5, 20, 10, key="assoc_top_items")
                    subset = top_items_df.head(top_n)
                    fig_freq = px.bar(
                        subset.sort_values("support_proxy", ascending=True),
                        x="support_proxy",
                        y="item",
                        orientation="h",
                        text="support_proxy",
                    )
                    fig_freq.update_layout(height=430, title="Items más frecuentes")
                    st.plotly_chart(fig_freq, use_container_width=True)
                else:
                    st.info("No hay top items disponibles.")

            st.dataframe(filtered_rules_show, use_container_width=True, height=340)

            if show_download_buttons:
                st.download_button(
                    "Descargar reglas filtradas",
                    data=filtered_rules.to_csv(index=False).encode("utf-8-sig"),
                    file_name="association_rules_filtered.csv",
                    mime="text/csv",
                )

    if not itemsets_df.empty and show_raw_tables:
        card = section_card("Itemsets frecuentes")
        card.dataframe(itemsets_df, use_container_width=True, height=320)


# =========================================================
# PRODUCTOS
# =========================================================
elif page == "Productos":
    card = section_card("Explorador de productos scrapeados")

    with card:
        if products_df.empty:
            st.info("No hay productos disponibles.")
        else:
            c1, c2, c3 = st.columns(3)
            brands = sorted(products_df["brand"].dropna().astype(str).unique().tolist()) if "brand" in products_df.columns else []
            segments = sorted(products_df["price_segment"].dropna().astype(str).unique().tolist()) if "price_segment" in products_df.columns else []

            selected_brands = c1.multiselect("Marca", options=brands, default=[])
            selected_segments = c2.multiselect("Segmento", options=segments, default=[])
            search_text = c3.text_input("Buscar por título")

            filtered_products = products_df.copy()
            if selected_brands:
                filtered_products = filtered_products[filtered_products["brand"].astype(str).isin(selected_brands)]
            if selected_segments:
                filtered_products = filtered_products[filtered_products["price_segment"].astype(str).isin(selected_segments)]
            if search_text.strip() and "title" in filtered_products.columns:
                filtered_products = filtered_products[
                    filtered_products["title"].fillna("").str.lower().str.contains(search_text.strip().lower())
                ]

            top_preview = st.slider("Filas a mostrar", 5, 100, 20)

            chart_col1, chart_col2 = st.columns(2)
            with chart_col1:
                if "price" in filtered_products.columns and not filtered_products.empty:
                    fig_price = px.histogram(filtered_products, x="price", nbins=20)
                    fig_price.update_layout(height=360, title="Distribución de precios")
                    st.plotly_chart(fig_price, use_container_width=True)
            with chart_col2:
                if "brand" in filtered_products.columns and not filtered_products.empty:
                    brand_counts = filtered_products["brand"].value_counts().reset_index()
                    brand_counts.columns = ["brand", "count"]
                    fig_brand = px.bar(brand_counts.head(12), x="brand", y="count", text="count")
                    fig_brand.update_layout(height=360, title="Top marcas")
                    st.plotly_chart(fig_brand, use_container_width=True)

            st.dataframe(filtered_products.head(top_preview), use_container_width=True, height=360)

            if show_download_buttons:
                st.download_button(
                    "Descargar productos filtrados",
                    data=filtered_products.to_csv(index=False).encode("utf-8-sig"),
                    file_name="products_filtered.csv",
                    mime="text/csv",
                )

    if not nn_dataset_df.empty and show_raw_tables:
        card = section_card("Dataset procesado para redes neuronales")
        card.dataframe(nn_dataset_df.head(50), use_container_width=True, height=320)