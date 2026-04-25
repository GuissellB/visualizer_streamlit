# Desarrollado por: Guissell Betancur Oviedo y Anyelin Arias Camacho
# Curso: Minería de Datos Avanzada

"""
Caso de estudio integrado: Web Mining + Reglas de Asociación + Redes Neuronales.

Diseñado para trabajar SOBRE el toolkit del curso (ml_toolkit.py) y para que luego
sea fácil construir un dashboard en Streamlit a partir de los CSV/JSON generados.

Escenario propuesto:
- Se hace web mining sobre un catálogo de laptops.
- Se limpia y estructura la información del producto.
- Se generan transacciones sintéticas/analíticas por producto para reglas de asociación.
- Se construyen 5 configuraciones de redes neuronales para clasificar el segmento de precio.

Salidas pensadas para Streamlit:
- CSV limpios para tablas
- JSON livianos para metadata y matrices de confusión
- Un manifest con las rutas relevantes del experimento

Uso sugerido:
    python caso_estudio_integrado.py --output-dir outputs

Requisitos esperados:
- Tener disponible ml_toolkit.py en la misma carpeta, o indicar --toolkit-path.
- Instalar dependencias usuales del toolkit: pandas, numpy, scikit-learn,
  imbalanced-learn, beautifulsoup4, requests, mlxtend.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urljoin

import numpy as np
import pandas as pd


# =============================================================================
# CARGA DINÁMICA DEL TOOLKIT
# =============================================================================

def load_toolkit_module(toolkit_path: Path):
    script_dir = Path(__file__).resolve().parent

    if toolkit_path.is_absolute():
        resolved_path = toolkit_path
    else:
        resolved_path = (script_dir / toolkit_path).resolve()

    if not resolved_path.exists():
        raise FileNotFoundError(f"No se encontró el toolkit en: {resolved_path}")

    spec = importlib.util.spec_from_file_location("course_ml_toolkit", resolved_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"No se pudo cargar el toolkit desde: {resolved_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# =============================================================================
# CONFIGURACIÓN
# =============================================================================
DEFAULT_START_URLS = [
    "https://webscraper.io/test-sites/e-commerce/static/computers/laptops",
]


@dataclass
class PipelineConfig:
    output_dir: str = "outputs"
    toolkit_path: str = "../ml_toolkit.py"
    random_state: int = 42
    min_support: float = 0.10
    min_confidence: float = 0.55
    max_pages: int = 50
    top_k_words: int = 15
    price_quantiles: Tuple[float, float] = (0.33, 0.66)
    start_urls: Optional[List[str]] = None


# =============================================================================
# UTILIDADES GENERALES
# =============================================================================

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def slugify(text: Any) -> str:
    text = str(text or "").strip().lower()
    text = re.sub(r"[^a-z0-9áéíóúñü]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "na"


def safe_float(text: Any) -> Optional[float]:
    if text is None:
        return None
    cleaned = re.sub(r"[^0-9.,-]", "", str(text))
    if not cleaned:
        return None
    cleaned = cleaned.replace(",", "")
    try:
        return float(cleaned)
    except ValueError:
        return None


def safe_int(text: Any) -> Optional[int]:
    if text is None:
        return None
    match = re.search(r"(\d+)", str(text))
    return int(match.group(1)) if match else None


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [slugify(c) for c in out.columns]
    return out


def flatten_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    flat = {}
    for key, value in metrics.items():
        if isinstance(value, np.ndarray):
            flat[key] = value.tolist()
        elif isinstance(value, (np.floating, np.integer)):
            flat[key] = value.item()
        else:
            flat[key] = value
    return flat


# =============================================================================
# WEB MINING
# =============================================================================

def extract_product_blocks(toolkit: Any, url: str) -> pd.DataFrame:
    """
    Extrae productos de la página usando el WebMiningToolkit del curso.
    """
    wm = toolkit.WebMiningToolkit()
    wm.fetch(url)

    fields = {
        "title": {"selector": ".title", "attr": "text", "default": None},
        "detail_href": {"selector": ".title", "attr": "href", "default": None},
        "price_text": {"selector": ".price", "attr": "text", "default": None},
        "description": {"selector": ".description", "attr": "text", "default": None},
        "rating_text": {"selector": ".ratings p[data-rating]", "attr": "data-rating", "default": None},
        "reviews_text": {"selector": ".ratings .pull-right", "attr": "text", "default": None},
    }

    records = wm.extract_records(item_selector=".thumbnail", fields=fields)
    if records.empty:
        return records

    records["source_url"] = url
    records["detail_url"] = records["detail_href"].apply(lambda x: urljoin(url, x) if pd.notna(x) else None)
    return records


def discover_next_page(toolkit: Any, current_url: str) -> Optional[str]:
    wm = toolkit.WebMiningToolkit()
    soup = wm.fetch(current_url)
    current_active = soup.select_one(".pagination .active")
    if current_active is None:
        return None
    next_li = current_active.find_next_sibling("li")
    if next_li is None:
        return None
    link = next_li.find("a", href=True)
    if link is None:
        return None
    return urljoin(current_url, link["href"])


def scrape_catalog(toolkit: Any, start_urls: Sequence[str], max_pages: int = 50) -> pd.DataFrame:
    visited = set()
    collected: List[pd.DataFrame] = []

    for start_url in start_urls:
        current = start_url
        pages_seen = 0

        while current and current not in visited and pages_seen < max_pages:
            visited.add(current)
            page_df = extract_product_blocks(toolkit, current)
            if not page_df.empty:
                collected.append(page_df)
            current = discover_next_page(toolkit, current)
            pages_seen += 1

    if not collected:
        return pd.DataFrame()

    raw = pd.concat(collected, ignore_index=True).drop_duplicates()
    return raw


# =============================================================================
# LIMPIEZA Y FEATURE ENGINEERING
# =============================================================================
KNOWN_BRANDS = [
    "asus", "acer", "hp", "dell", "lenovo", "apple", "msi", "toshiba",
    "samsung", "sony", "gigabyte", "medion", "huawei", "xiaomi",
]

RAM_REGEX = re.compile(r"(\d+)\s*gb\s*ram", re.IGNORECASE)
STORAGE_REGEX = re.compile(r"(\d+)\s*(gb|tb)\s*(ssd|hdd|emmc)?", re.IGNORECASE)
SCREEN_REGEX = re.compile(r"(\d+(?:\.\d+)?)\s*\"", re.IGNORECASE)
CPU_TOKEN_REGEX = re.compile(
    r"(i3|i5|i7|i9|ryzen\s*3|ryzen\s*5|ryzen\s*7|celeron|pentium|atom|xeon)",
    re.IGNORECASE,
)


def infer_brand(title: str) -> str:
    title_norm = str(title or "").lower()
    for brand in KNOWN_BRANDS:
        if brand in title_norm:
            return brand.title()
    first_word = str(title or "").split(" ")[0].strip()
    return first_word.title() if first_word else "Unknown"


def extract_ram_gb(text: str) -> Optional[int]:
    match = RAM_REGEX.search(str(text or ""))
    return int(match.group(1)) if match else None


def extract_storage(text: str) -> Tuple[Optional[float], Optional[str], Optional[str]]:
    match = STORAGE_REGEX.search(str(text or ""))
    if not match:
        return None, None, None

    amount = float(match.group(1))
    unit = str(match.group(2)).upper()
    disk_type = str(match.group(3)).upper() if match.group(3) else None
    storage_gb = amount * 1024 if unit == "TB" else amount
    return storage_gb, unit, disk_type


def extract_screen_inches(text: str) -> Optional[float]:
    match = SCREEN_REGEX.search(str(text or ""))
    return float(match.group(1)) if match else None


def infer_cpu_family(text: str) -> str:
    match = CPU_TOKEN_REGEX.search(str(text or ""))
    return match.group(1).upper().replace("  ", " ") if match else "OTHER"


def infer_storage_bucket(storage_gb: Optional[float]) -> str:
    if storage_gb is None or pd.isna(storage_gb):
        return "Unknown"
    if storage_gb <= 128:
        return "<=128GB"
    if storage_gb <= 256:
        return "129-256GB"
    if storage_gb <= 512:
        return "257-512GB"
    if storage_gb <= 1024:
        return "513GB-1TB"
    return ">1TB"


def infer_ram_bucket(ram_gb: Optional[int]) -> str:
    if ram_gb is None or pd.isna(ram_gb):
        return "Unknown"
    if ram_gb <= 4:
        return "<=4GB"
    if ram_gb <= 8:
        return "5-8GB"
    if ram_gb <= 16:
        return "9-16GB"
    return ">16GB"


def infer_rating_bucket(rating: Optional[float]) -> str:
    if rating is None or pd.isna(rating):
        return "Unknown"
    if rating <= 2:
        return "Low"
    if rating <= 3:
        return "Medium"
    return "High"


def add_keyword_flags(df: pd.DataFrame, top_k_words: int = 15) -> Tuple[pd.DataFrame, List[str]]:
    work = df.copy()
    corpus = (
        work["title"].fillna("") + " " + work["description"].fillna("")
    ).str.lower()

    tokens = (
        corpus.str.findall(r"[a-zA-Z]{4,}")
        .explode()
        .dropna()
    )

    stopwords = {
        "with", "from", "have", "this", "that", "your", "laptop", "notebook",
        "intel", "windows", "inch", "black", "white", "silver", "gray",
    }

    freq = tokens[~tokens.isin(stopwords)].value_counts()
    keywords = freq.head(top_k_words).index.tolist()

    for word in keywords:
        work[f"kw_{slugify(word)}"] = corpus.str.contains(rf"\b{re.escape(word)}\b", regex=True).astype(int)

    return work, keywords


def prepare_product_dataset(raw_df: pd.DataFrame, top_k_words: int = 15) -> pd.DataFrame:
    if raw_df.empty:
        raise ValueError("No se pudo construir el dataset porque el scraping retornó 0 filas.")

    df = raw_df.copy()
    df = normalize_columns(df)

    df["price"] = df["price_text"].apply(safe_float)
    df["rating"] = pd.to_numeric(df["rating_text"], errors="coerce")
    df["reviews_count"] = df["reviews_text"].apply(safe_int)
    df["brand"] = df["title"].apply(infer_brand)
    df["ram_gb"] = (df["title"].fillna("") + " " + df["description"].fillna("")).apply(extract_ram_gb)

    storage_parsed = (df["title"].fillna("") + " " + df["description"].fillna("")).apply(extract_storage)
    df[["storage_gb", "storage_unit", "storage_type"]] = pd.DataFrame(storage_parsed.tolist(), index=df.index)

    df["screen_inches"] = (df["title"].fillna("") + " " + df["description"].fillna("")).apply(extract_screen_inches)
    df["cpu_family"] = (df["title"].fillna("") + " " + df["description"].fillna("")).apply(infer_cpu_family)

    df["ram_bucket"] = df["ram_gb"].apply(infer_ram_bucket)
    df["storage_bucket"] = df["storage_gb"].apply(infer_storage_bucket)
    df["rating_bucket"] = df["rating"].apply(infer_rating_bucket)
    df["has_ssd"] = df["storage_type"].fillna("").str.upper().eq("SSD").astype(int)
    df["has_hdd"] = df["storage_type"].fillna("").str.upper().eq("HDD").astype(int)

    df, keywords = add_keyword_flags(df, top_k_words=top_k_words)

    # Limpieza mínima para downstream.
    df = df.drop_duplicates(subset=["title", "price", "description"]).reset_index(drop=True)
    df = df[df["price"].notna()].reset_index(drop=True)

    # Targets útiles para el caso.
    q1, q2 = df["price"].quantile([0.33, 0.66]).tolist()
    df["price_segment"] = pd.cut(
        df["price"],
        bins=[-np.inf, q1, q2, np.inf],
        labels=["Budget", "MidRange", "Premium"],
    ).astype(str)
    df["is_premium"] = (df["price_segment"] == "Premium").astype(int)

    df.attrs["keywords"] = keywords
    df.attrs["price_thresholds"] = {"q33": float(q1), "q66": float(q2)}
    return df


# =============================================================================
# REGLAS DE ASOCIACIÓN
# =============================================================================

def product_to_transaction(row: pd.Series) -> List[str]:
    items = [
        f"brand={slugify(row.get('brand'))}",
        f"price_segment={slugify(row.get('price_segment'))}",
        f"ram_bucket={slugify(row.get('ram_bucket'))}",
        f"storage_bucket={slugify(row.get('storage_bucket'))}",
        f"cpu_family={slugify(row.get('cpu_family'))}",
        f"rating_bucket={slugify(row.get('rating_bucket'))}",
    ]

    if int(row.get("has_ssd", 0)) == 1:
        items.append("storage_type=ssd")
    if int(row.get("has_hdd", 0)) == 1:
        items.append("storage_type=hdd")

    keyword_cols = [c for c in row.index if str(c).startswith("kw_")]
    for col in keyword_cols:
        if int(row[col]) == 1:
            items.append(col.replace("kw_", "keyword="))

    return items


def build_transactions_df(products_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for idx, row in products_df.iterrows():
        for item in product_to_transaction(row):
            rows.append({"transaction_id": f"product_{idx:04d}", "item": item})
    return pd.DataFrame(rows)


def run_association_rules(toolkit: Any, products_df: pd.DataFrame, min_support: float, min_confidence: float) -> Dict[str, pd.DataFrame]:
    tx_df = build_transactions_df(products_df)
    explorer = toolkit.AssociationRulesExplorer.from_transaction_df(
        tx_df, transaction_col="transaction_id", item_col="item"
    )
    encoded = explorer.encode_transactions()
    itemsets = explorer.fit_itemsets(min_support=min_support, use_colnames=True)
    rules = explorer.fit_rules(min_support=min_support, metric="confidence", min_threshold=min_confidence)
    top_items = explorer.top_items(top_n=15).rename_axis("item").reset_index(name="support_proxy")

    if not rules.empty:
        rules = rules.copy()
        rules["antecedents"] = rules["antecedents"].apply(lambda x: ", ".join(sorted(list(x))))
        rules["consequents"] = rules["consequents"].apply(lambda x: ", ".join(sorted(list(x))))

    if not itemsets.empty:
        itemsets = itemsets.copy()
        itemsets["itemsets"] = itemsets["itemsets"].apply(lambda x: ", ".join(sorted(list(x))))

    return {
        "transactions": tx_df,
        "encoded": encoded,
        "itemsets": itemsets,
        "rules": rules,
        "top_items": top_items,
    }


# =============================================================================
# REDES NEURONALES
# =============================================================================

def build_nn_experiments() -> List[Dict[str, Any]]:
    """
    Cinco configuraciones claramente diferenciadas para cumplir el requisito del caso.
    Están definidas para trabajar con NeuralNetworkRunner del toolkit.
    """
    return [
        {
            "model_name": "NN_01_Shallow_ReLU_Adam",
            "hidden_layer_sizes": (16,),
            "activation": "relu",
            "solver": "adam",
            "alpha": 0.0001,
            "learning_rate_init": 0.001,
            "max_iter": 600,
            "early_stopping": False,
        },
        {
            "model_name": "NN_02_Deep_ReLU_Adam",
            "hidden_layer_sizes": (64, 32, 16),
            "activation": "relu",
            "solver": "adam",
            "alpha": 0.0001,
            "learning_rate_init": 0.001,
            "max_iter": 900,
            "early_stopping": True,
        },
        {
            "model_name": "NN_03_Wide_Tanh_Adam",
            "hidden_layer_sizes": (128, 64),
            "activation": "tanh",
            "solver": "adam",
            "alpha": 0.0005,
            "learning_rate_init": 0.001,
            "max_iter": 900,
            "early_stopping": True,
        },
        {
            "model_name": "NN_04_Compact_Logistic_LBFGS",
            "hidden_layer_sizes": (32, 16),
            "activation": "logistic",
            "solver": "lbfgs",
            "alpha": 0.0010,
            "learning_rate_init": 0.001,
            "max_iter": 800,
            "early_stopping": False,
        },
        {
            "model_name": "NN_05_Tanh_SGD_MomentumLike",
            "hidden_layer_sizes": (48, 24, 12),
            "activation": "tanh",
            "solver": "sgd",
            "alpha": 0.0001,
            "learning_rate_init": 0.01,
            "max_iter": 1200,
            "early_stopping": True,
        },
    ]


def prepare_nn_dataset(products_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    base = products_df.copy()

    candidate_cols = [
        "price", "rating", "reviews_count", "ram_gb", "storage_gb", "screen_inches",
        "has_ssd", "has_hdd", "brand", "cpu_family", "ram_bucket", "storage_bucket", "rating_bucket",
    ] + [c for c in base.columns if c.startswith("kw_")]

    working = base[candidate_cols + ["price_segment"]].copy()

    # Imputación simple para robustez.
    numeric_cols = working.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in working.columns if c not in numeric_cols + ["price_segment"]]

    for col in numeric_cols:
        working[col] = working[col].fillna(working[col].median())
    for col in categorical_cols:
        working[col] = working[col].fillna("Unknown")

    working = pd.get_dummies(working, columns=categorical_cols, drop_first=False)
    feature_cols = [c for c in working.columns if c != "price_segment"]
    return working, feature_cols


def run_neural_networks(toolkit: Any, products_df: pd.DataFrame, random_state: int) -> Dict[str, Any]:
    dataset, feature_cols = prepare_nn_dataset(products_df)
    experiments = build_nn_experiments()

    holdout_rows: List[Dict[str, Any]] = []
    cv_rows: List[Dict[str, Any]] = []
    confusion_payload: Dict[str, Any] = {}
    architectures_payload: Dict[str, Any] = {}

    for exp in experiments:
        runner = toolkit.NeuralNetworkRunner(
            df=dataset,
            target="price_segment",
            task="classification",
            features=feature_cols,
            random_state=random_state,
            encode_target=True,
            pos_label=1,
            **{k: v for k, v in exp.items() if k != "model_name"},
        )

        holdout_metrics = flatten_metrics(runner.evaluate())
        cv_metrics = flatten_metrics(runner.evaluate_cv(n_splits=5, shuffle=True))
        architectures_payload[exp["model_name"]] = runner.architecture()

        cm = holdout_metrics.pop("ConfusionMatrix", None)
        if cm is not None:
            confusion_payload[exp["model_name"]] = cm

        holdout_rows.append({"model_name": exp["model_name"], **holdout_metrics})
        cv_rows.append({"model_name": exp["model_name"], **cv_metrics})

    holdout_df = pd.DataFrame(holdout_rows)
    cv_df = pd.DataFrame(cv_rows)

    holdout_sort_col = "F1_Pos" if "F1_Pos" in holdout_df.columns else (
        "Accuracy" if "Accuracy" in holdout_df.columns else None
    )
    cv_sort_col = "F1_Pos" if "F1_Pos" in cv_df.columns else (
        "Accuracy" if "Accuracy" in cv_df.columns else None
    )

    if holdout_sort_col is not None:
        holdout_df = holdout_df.sort_values(by=holdout_sort_col, ascending=False, na_position="last")

    if cv_sort_col is not None:
        cv_df = cv_df.sort_values(by=cv_sort_col, ascending=False, na_position="last")

    return {
        "dataset": dataset,
        "feature_cols": feature_cols,
        "holdout_results": holdout_df,
        "cv_results": cv_df,
        "confusion_matrices": confusion_payload,
        "architectures": architectures_payload,
    }


# =============================================================================
# EXPORTACIÓN PARA STREAMLIT
# =============================================================================

def export_dataframe(df: pd.DataFrame, path: Path) -> str:
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return str(path)


def export_json(payload: Dict[str, Any], path: Path) -> str:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return str(path)


def persist_outputs(
    config: PipelineConfig,
    raw_products: pd.DataFrame,
    products_df: pd.DataFrame,
    assoc_results: Dict[str, pd.DataFrame],
    nn_results: Dict[str, Any],
) -> Dict[str, str]:
    out_dir = ensure_dir(Path(config.output_dir))
    files: Dict[str, str] = {}

    files["raw_products_csv"] = export_dataframe(raw_products, out_dir / "01_raw_products.csv")
    files["products_csv"] = export_dataframe(products_df, out_dir / "02_products_clean.csv")
    files["association_transactions_csv"] = export_dataframe(assoc_results["transactions"], out_dir / "03_transactions.csv")
    files["association_itemsets_csv"] = export_dataframe(assoc_results["itemsets"], out_dir / "04_itemsets.csv")
    files["association_rules_csv"] = export_dataframe(assoc_results["rules"], out_dir / "05_association_rules.csv")
    files["association_top_items_csv"] = export_dataframe(assoc_results["top_items"], out_dir / "06_top_items.csv")
    files["nn_dataset_csv"] = export_dataframe(nn_results["dataset"], out_dir / "07_nn_dataset.csv")
    files["nn_holdout_results_csv"] = export_dataframe(nn_results["holdout_results"], out_dir / "08_nn_holdout_results.csv")
    files["nn_cv_results_csv"] = export_dataframe(nn_results["cv_results"], out_dir / "09_nn_cv_results.csv")

    files["nn_confusion_matrices_json"] = export_json(nn_results["confusion_matrices"], out_dir / "10_nn_confusion_matrices.json")
    files["nn_architectures_json"] = export_json(nn_results["architectures"], out_dir / "11_nn_architectures.json")

    manifest = {
        "scenario": "Catalogo de laptops: web mining + reglas de asociacion + redes neuronales",
        "toolkit_path": config.toolkit_path,
        "random_state": config.random_state,
        "min_support": config.min_support,
        "min_confidence": config.min_confidence,
        "sources": config.start_urls or DEFAULT_START_URLS,
        "row_counts": {
            "raw_products": int(len(raw_products)),
            "products_clean": int(len(products_df)),
            "transactions": int(len(assoc_results["transactions"])),
            "association_rules": int(len(assoc_results["rules"])),
            "nn_dataset": int(len(nn_results["dataset"])),
        },
        "keywords": products_df.attrs.get("keywords", []),
        "price_thresholds": products_df.attrs.get("price_thresholds", {}),
        "files": files,
    }
    files["manifest_json"] = export_json(manifest, out_dir / "00_manifest.json")
    return files


# =============================================================================
# ORQUESTACIÓN
# =============================================================================

def run_pipeline(config: PipelineConfig) -> Dict[str, Any]:
    toolkit = load_toolkit_module(Path(config.toolkit_path))
    start_urls = config.start_urls or DEFAULT_START_URLS

    raw_products = scrape_catalog(toolkit, start_urls=start_urls, max_pages=config.max_pages)
    products_df = prepare_product_dataset(raw_products, top_k_words=config.top_k_words)
    assoc_results = run_association_rules(
        toolkit,
        products_df=products_df,
        min_support=config.min_support,
        min_confidence=config.min_confidence,
    )
    nn_results = run_neural_networks(toolkit, products_df=products_df, random_state=config.random_state)
    files = persist_outputs(config, raw_products, products_df, assoc_results, nn_results)

    return {
        "config": asdict(config),
        "raw_products": raw_products,
        "products_df": products_df,
        "association": assoc_results,
        "neural_networks": nn_results,
        "files": files,
    }


# =============================================================================
# CLI
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Caso de estudio integrado usando ml_toolkit.py")
    parser.add_argument("--output-dir", default="outputs", help="Carpeta donde se guardarán CSV y JSON")
    parser.add_argument("--toolkit-path", default="../ml_toolkit.py", help="Ruta al archivo ml_toolkit.py")
    parser.add_argument("--random-state", type=int, default=42, help="Semilla global")
    parser.add_argument("--min-support", type=float, default=0.10, help="Soporte mínimo para Apriori")
    parser.add_argument("--min-confidence", type=float, default=0.55, help="Confianza mínima para reglas")
    parser.add_argument("--max-pages", type=int, default=50, help="Máximo de páginas a scrapear por URL inicial")
    parser.add_argument("--top-k-words", type=int, default=15, help="Cantidad de keywords binarias para features")
    parser.add_argument(
        "--start-url",
        action="append",
        dest="start_urls",
        help="URL inicial de scraping. Se puede repetir el argumento varias veces.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config = PipelineConfig(
        output_dir=args.output_dir,
        toolkit_path=args.toolkit_path,
        random_state=args.random_state,
        min_support=args.min_support,
        min_confidence=args.min_confidence,
        max_pages=args.max_pages,
        top_k_words=args.top_k_words,
        start_urls=args.start_urls,
    )

    results = run_pipeline(config)

    summary = {
        "productos_scrapeados": int(len(results["raw_products"])),
        "productos_limpios": int(len(results["products_df"])),
        "reglas_generadas": int(len(results["association"]["rules"])),
        "mejor_modelo_holdout": None,
        "archivo_manifest": results["files"].get("manifest_json"),
    }

    holdout_df = results["neural_networks"]["holdout_results"]
    if not holdout_df.empty:
        summary["mejor_modelo_holdout"] = holdout_df.iloc[0]["model_name"]

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
