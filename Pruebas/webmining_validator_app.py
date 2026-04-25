# Desarrollado por: Guissell Betancur Oviedo y Anyelin Arias Camacho
# Curso: Minería de Datos Avanzada

import json
from typing import Dict, Any

import pandas as pd
import streamlit as st

from ml_toolkit import WebMiningToolkit


st.set_page_config(page_title="Web Mining Validator", layout="wide")
st.title("Validador de Web Mining")
st.caption("Prueba rápida de las utilidades de scraping y extracción del toolkit.")


def parse_json_config(raw_text: str) -> Dict[str, Any]:
    raw_text = (raw_text or "").strip()
    if not raw_text:
        return {}
    return json.loads(raw_text)


def show_error(exc: Exception):
    st.error(f"{type(exc).__name__}: {exc}")


with st.sidebar:
    st.markdown("## Configuración")
    request_timeout = st.number_input("Timeout (segundos)", min_value=1, max_value=120, value=15)
    parser_name = st.selectbox("Parser HTML", options=["html.parser", "lxml"], index=0)
    user_agent = st.text_input("User-Agent", value="Mozilla/5.0")
    fetch_mode = st.radio("Modo de carga", options=["HTTP", "Dinámico (Selenium)"], index=0)
    selenium_driver = st.selectbox("Driver Selenium", options=["firefox", "chrome"], index=0)
    wait_seconds = st.number_input("Espera Selenium", min_value=0.0, max_value=30.0, value=2.0, step=0.5)


toolkit = WebMiningToolkit(
    headers={"User-Agent": user_agent},
    timeout=int(request_timeout),
    parser=parser_name,
)

url = st.text_input("URL a analizar", value="https://example.com")

tab_fetch, tab_text, tab_links, tab_table, tab_regex, tab_records = st.tabs(
    ["HTML", "Texto", "Links", "Tablas", "Regex", "Registros"]
)


with tab_fetch:
    st.subheader("Cargar HTML")
    left, right = st.columns([0.55, 0.45], gap="large")

    with left:
        run_fetch = st.button("Cargar página", use_container_width=True)

        if run_fetch:
            try:
                if fetch_mode == "HTTP":
                    toolkit.fetch(url)
                else:
                    browser = toolkit.fetch_dynamic(url, driver=selenium_driver, wait_seconds=float(wait_seconds))
                    browser.quit()
                st.session_state["wm_html"] = toolkit.last_html_
                st.success("Página cargada correctamente.")
            except Exception as exc:
                show_error(exc)

    with right:
        html_value = st.session_state.get("wm_html", "")
        st.text_area("HTML obtenido", value=html_value, height=420)


def ensure_loaded():
    html = st.session_state.get("wm_html")
    if not html:
        st.info("Primero carga una página en la pestaña HTML.")
        return False
    toolkit.last_html_ = html
    from bs4 import BeautifulSoup

    toolkit.last_soup_ = BeautifulSoup(html, toolkit.parser)
    return True


with tab_text:
    st.subheader("Extraer texto")
    col1, col2 = st.columns(2, gap="large")

    with col1:
        text_tag = st.text_input("Tag", value="p", key="text_tag")
        text_attrs_raw = st.text_area("Attrs JSON", value='{}', height=80, key="text_attrs")
        text_selector = st.text_input("CSS selector opcional", value="", key="text_selector")
        text_limit = st.number_input("Límite", min_value=0, max_value=500, value=10, key="text_limit")
        text_strip = st.toggle("Aplicar strip", value=True, key="text_strip")
        run_text = st.button("Extraer texto", use_container_width=True)

    with col2:
        if run_text and ensure_loaded():
            try:
                attrs = parse_json_config(text_attrs_raw)
                items = toolkit.extract_text(
                    tag=text_tag or None,
                    attrs=attrs or None,
                    css_selector=text_selector or None,
                    limit=None if int(text_limit) == 0 else int(text_limit),
                    strip=text_strip,
                )
                st.write(f"Resultados: {len(items)}")
                st.dataframe(pd.DataFrame({"texto": items}), width="stretch", hide_index=True)
            except Exception as exc:
                show_error(exc)


with tab_links:
    st.subheader("Extraer links")
    col1, col2 = st.columns(2, gap="large")

    with col1:
        link_tag = st.text_input("Tag", value="a", key="link_tag")
        link_attrs_raw = st.text_area("Attrs JSON", value="{}", height=80, key="link_attrs")
        href_contains = st.text_input("Filtrar href que contenga", value="", key="href_contains")
        run_links = st.button("Extraer links", use_container_width=True)

    with col2:
        if run_links and ensure_loaded():
            try:
                attrs = parse_json_config(link_attrs_raw)
                links = toolkit.extract_links(
                    tag=link_tag,
                    attrs=attrs or None,
                    href_contains=href_contains or None,
                )
                st.write(f"Resultados: {len(links)}")
                st.dataframe(pd.DataFrame({"href": links}), width="stretch", hide_index=True)
            except Exception as exc:
                show_error(exc)


with tab_table:
    st.subheader("Extraer tabla HTML")
    col1, col2 = st.columns(2, gap="large")

    with col1:
        table_attrs_raw = st.text_area("Attrs JSON", value="{}", height=80, key="table_attrs")
        table_index = st.number_input("Índice de tabla", min_value=0, max_value=50, value=0, key="table_index")
        run_table = st.button("Leer tabla", use_container_width=True)

    with col2:
        if run_table and ensure_loaded():
            try:
                attrs = parse_json_config(table_attrs_raw)
                df_table = toolkit.extract_table(attrs=attrs or None, index=int(table_index))
                st.dataframe(df_table, width="stretch")
            except Exception as exc:
                show_error(exc)


with tab_regex:
    st.subheader("Filtrar o extraer con regex")
    col1, col2 = st.columns(2, gap="large")

    with col1:
        regex_source_raw = st.text_area(
            "Texto fuente (una línea por registro)",
            value="Equipo A vs. Equipo B\nLiga X: Local vs. Visitante",
            height=200,
        )
        regex_pattern = st.text_input("Patrón regex", value=r"(.+?)\s+vs\.\s+(.+)")
        regex_group_names = st.text_input("Nombres de grupos separados por coma", value="local,visitante")
        regex_mode = st.radio("Modo", options=["extract", "filter"], horizontal=True)
        run_regex = st.button("Aplicar regex", use_container_width=True)

    with col2:
        if run_regex:
            try:
                texts = [line for line in regex_source_raw.splitlines() if line.strip()]
                if regex_mode == "filter":
                    filtered = toolkit.regex_filter(texts, regex_pattern)
                    st.dataframe(pd.DataFrame({"text": filtered}), width="stretch", hide_index=True)
                else:
                    group_names = [x.strip() for x in regex_group_names.split(",") if x.strip()]
                    df_regex = toolkit.regex_extract(
                        texts,
                        regex_pattern,
                        group_names=group_names or None,
                    )
                    st.dataframe(df_regex, width="stretch", hide_index=True)
            except Exception as exc:
                show_error(exc)


with tab_records:
    st.subheader("Extraer registros repetidos")
    st.caption("Útil para cards de productos, noticias, resultados deportivos, artículos y listados similares.")
    col1, col2 = st.columns(2, gap="large")

    with col1:
        item_selector = st.text_input("Selector del bloque repetido", value="div")
        fields_raw = st.text_area(
            "Config JSON de campos",
            value=json.dumps(
                {
                    "titulo": {"selector": "h1", "attr": "text", "default": None},
                    "link": {"selector": "a", "attr": "href", "default": None},
                },
                ensure_ascii=True,
                indent=2,
            ),
            height=240,
        )
        run_records = st.button("Extraer registros", use_container_width=True)

    with col2:
        if run_records and ensure_loaded():
            try:
                fields = parse_json_config(fields_raw)
                df_records = toolkit.extract_records(item_selector=item_selector, fields=fields)
                st.dataframe(df_records, width="stretch")
            except Exception as exc:
                show_error(exc)
