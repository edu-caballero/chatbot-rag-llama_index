#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chatbot RAG sobre CSV de Rotten Tomatoes con FunctionAgent.

- LLM: OpenAI gpt-4o-mini-2024-07-18 (OPENAI_API_KEY)
- Embeddings: sentence-transformers/all-MiniLM-L6-v2 (HuggingFace/local)
- Persistencia: storage_movies/ (NO reindexa si ya existe)
- Tools: rag_search (RAG) + movie_stats (tabular)
- Logging: registra qué tool se ejecuta en cada respuesta.
"""

import os
import re
import hashlib
import asyncio
import logging
from datetime import datetime
from typing import Optional, Literal, List

import pandas as pd
from dotenv import load_dotenv

from llama_index.core import (
    Document,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    Settings,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler

# ==============
# Config / Paths
# ==============
load_dotenv()

CSV_PATH = os.getenv("CSV_PATH", "data/rotten_tomatoes_movies.csv")
PERSIST_DIR = "storage_movies"
FINGERPRINT_FILE = os.path.join(PERSIST_DIR, ".embed_fingerprint.txt")

OPENAI_LLM_MODEL = "gpt-4o-mini-2024-07-18"
EMBED_MODEL_ID = os.getenv(
    "EMBED_MODEL_ID",
    "sentence-transformers/all-MiniLM-L6-v2"  # para ES/EN podrías usar "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# -------------
# Logging
# -------------
logger = logging.getLogger("movies_rag")
logger.setLevel(logging.INFO)
fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
ch = logging.StreamHandler(); ch.setFormatter(fmt); ch.setLevel(logging.INFO)
fh = logging.FileHandler("movies_rag.log", encoding="utf-8"); fh.setFormatter(fmt); fh.setLevel(logging.INFO)
logger.addHandler(ch); logger.addHandler(fh)

# =====================
# Utilidades persistencia
# =====================
def embed_fingerprint(provider: str, model_name: str) -> str:
    base = f"{provider}:{model_name}".encode("utf-8")
    return hashlib.sha256(base).hexdigest()[:16]

def read_fingerprint() -> str:
    try:
        with open(FINGERPRINT_FILE, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        return ""

def write_fingerprint(fp: str) -> None:
    os.makedirs(PERSIST_DIR, exist_ok=True)
    with open(FINGERPRINT_FILE, "w", encoding="utf-8") as f:
        f.write(fp)

# =====================
# Configurar modelos
# =====================
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI as OpenAILLM

def configure_models():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Falta OPENAI_API_KEY en el entorno o .env")

    # LLM OpenAI
    Settings.llm = OpenAILLM(
        model=OPENAI_LLM_MODEL,
        api_key=api_key,
        temperature=0.2,
    )

    # Embeddings HuggingFace
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=EMBED_MODEL_ID,
        device="cpu"  # "cuda" si tenés GPU
    )

    # Parser de nodos para RAG
    Settings.node_parser = SentenceSplitter(
        chunk_size=2048,
        chunk_overlap=64,
        include_metadata=False,
        include_prev_next_rel=False,
    )

    # Debug de prompts/recuperación
    debug_handler = LlamaDebugHandler(print_trace_on_end=True)
    Settings.callback_manager = CallbackManager([debug_handler])

# =====================
# Cargar CSV
# =====================
EXPECTED_COLS = [
    "rotten_tomatoes_link","movie_title","movie_info","critics_consensus",
    "content_rating","genres","directors","authors","actors",
    "original_release_date","streaming_release_date","runtime",
    "production_company","tomatometer_status","tomatometer_rating","tomatometer_count",
    "audience_status","audience_rating","audience_count",
    "tomatometer_top_critics_count","tomatometer_fresh_critics_count","tomatometer_rotten_critics_count",
]

def load_dataframe() -> pd.DataFrame:
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"No se encontró el CSV en {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    missing = [c for c in EXPECTED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}")

    # Normalizamos tipos a str donde aplica
    for c in EXPECTED_COLS:
        if c in df.columns and c not in ("tomatometer_rating", "audience_rating"):
            df[c] = df[c].astype(str).fillna("")
    # numéricos
    df["tomatometer_rating"] = pd.to_numeric(df["tomatometer_rating"], errors="coerce")
    df["audience_rating"] = pd.to_numeric(df["audience_rating"], errors="coerce")

    # derivadas útiles
    def _year(s):
        if not isinstance(s, str): return ""
        m = re.search(r"(\d{4})", s)
        return m.group(1) if m else ""
    df["year"] = df["original_release_date"].apply(_year)

    # Filtramos filas sin sinopsis
    df = df[df["movie_info"].astype(str).str.strip() != ""].reset_index(drop=True)
    logger.info(f"CSV cargado: {CSV_PATH} | Filas: {len(df)} | Columnas: {len(df.columns)}")
    return df

# =====================
# Construir documentos
# =====================
def make_documents_from_df(df: pd.DataFrame) -> List[Document]:
    docs: List[Document] = []
    for _, r in df.iterrows():
        parts = [
            f"Título: {r.get('movie_title','')}",
            f"Descripción: {r.get('movie_info','')}",
        ]
        if str(r.get("critics_consensus","")).strip():
            parts.append(f"Consensus de críticos: {r.get('critics_consensus','')}")
        if str(r.get("genres","")).strip():
            parts.append(f"Géneros: {r.get('genres','')}")
        if str(r.get("directors","")).strip():
            parts.append(f"Directores: {r.get('directors','')}")
        if str(r.get("actors","")).strip():
            parts.append(f"Actores: {r.get('actors','')}")
        if pd.notna(r.get("tomatometer_rating")):
            parts.append(f"Tomatometer: {r.get('tomatometer_rating')}")
        if pd.notna(r.get("audience_rating")):
            parts.append(f"Audience score: {r.get('audience_rating')}")

        text = "\n".join(parts)
        md = {
            "movie_title": r.get("movie_title",""),
            "genres": r.get("genres",""),
            "directors": r.get("directors",""),
            "actors": r.get("actors",""),
            "year": r.get("year",""),
            "tomatometer_rating": r.get("tomatometer_rating", None),
            "audience_rating": r.get("audience_rating", None),
            "rotten_tomatoes_link": r.get("rotten_tomatoes_link",""),
        }
        docs.append(Document(text=text, metadata=md))
    logger.info(f"Documentos RAG creados: {len(docs)}")
    return docs

# =====================
# Index: cargar o construir (NO reindex si existe)
# =====================
def build_or_load_index(docs: List[Document]) -> VectorStoreIndex:
    provider = "hf"
    model_id = EMBED_MODEL_ID
    current_fp = embed_fingerprint(provider, model_id)
    stored_fp = read_fingerprint()
    required = {"docstore.json", "index_store.json", "vector_store.json"}

    if os.path.isdir(PERSIST_DIR):
        existing = set(os.listdir(PERSIST_DIR))
        if required.issubset(existing) and stored_fp == current_fp:
            logger.info(f"Cargando índice persistido desde {PERSIST_DIR} (sin reindexar)…")
            storage_ctx = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
            return load_index_from_storage(storage_ctx, embed_model=Settings.embed_model)
        elif required.issubset(existing) and stored_fp and stored_fp != current_fp:
            # Existe índice pero fue hecho con otro embedder → NO reindexar automáticamente.
            raise RuntimeError(
                "Hay un índice persistido con fingerprint distinto (otro modelo de embeddings). "
                "Por política de 'no reindexar', aborto. Si querés reconstruir, borra 'storage_movies/' manualmente."
            )

    # Si no existe persistencia válida, construimos UNA vez
    logger.info("No hay índice persistido válido. Construyendo por única vez…")
    index = VectorStoreIndex.from_documents(docs, embed_model=Settings.embed_model)
    os.makedirs(PERSIST_DIR, exist_ok=True)
    index.storage_context.persist(persist_dir=PERSIST_DIR)
    write_fingerprint(current_fp)
    logger.info(f"Índice creado y persistido en {PERSIST_DIR}")
    return index

# =====================
# Tools del agente
# =====================
def build_rag_tool(index: VectorStoreIndex):
    query_engine = index.as_query_engine(similarity_top_k=4, verbose=True)

    async def rag_search(query: str) -> str:
        """RAG: responde preguntas sobre las películas (sinopsis, consenso, etc.)."""
        t0 = datetime.now()
        resp = await query_engine.aquery(query)
        dt = (datetime.now() - t0).total_seconds()

        fuentes = []
        for sn in getattr(resp, "source_nodes", []) or []:
            meta = sn.node.metadata or {}
            title = meta.get("movie_title") or "(sin título)"
            score = getattr(sn, "score", None)
            fuentes.append(f"- {title} (score={score:.3f})")

        answer = str(resp)
        if fuentes:
            answer += "\n\nFuentes recuperadas:\n" + "\n".join(fuentes)
        answer += f"\n\n[diag] RAG latency: {dt:.3f}s"
        return answer

    return rag_search

def build_movie_stats_tool(df: pd.DataFrame):
    def movie_stats(
        question: str,
        metric: Literal["tomatometer_rating", "audience_rating"] = "tomatometer_rating",
        mode: Literal["top", "bottom"] = "bottom",
        n: int = 5,
        by: Optional[str] = None,
        value: Optional[str] = None,
    ) -> str:
        """Consultas tabulares: rankings/filtrado directo sobre el CSV."""
        t0 = datetime.now()
        work = df.copy()

        if by and value:
            if by not in work.columns:
                return f"⚠️ La columna '{by}' no existe."
            work = work[work[by].astype(str).str.contains(str(value), case=False, na=False)]

        if metric not in work.columns:
            return f"⚠️ La métrica '{metric}' no existe en el CSV."
        work[metric] = pd.to_numeric(work[metric], errors="coerce")
        work = work.dropna(subset=[metric])

        if work.empty:
            return "⚠️ No hay filas que coincidan con el filtro/métrica."

        ascending = (mode == "bottom")
        work = work.sort_values(metric, ascending=ascending).head(n)

        lines = [f"Resultados ({mode} por {metric}) - n={n}"]
        for _, r in work.iterrows():
            lines.append(
                f"- {r.get('movie_title','')} | {metric}={r.get(metric,'')} | "
                f"audience={r.get('audience_rating','')} | año={r.get('year','')} | "
                f"director={r.get('directors','')}"
            )
        dt = (datetime.now() - t0).total_seconds()
        lines.append(f"[diag] Filas tras filtro: {len(work)} | latency={dt:.3f}s")
        return "\n".join(lines)

    return movie_stats

# ============
# Tool logger
# ============
def wrap_tool(tool_fn, name: str):
    """Envuelve una tool para loguear cada invocación."""
    async def async_wrapper(*args, **kwargs):
        logger.info(f"[TOOL] Ejecutando tool: {name}")
        return await tool_fn(*args, **kwargs)

    def sync_wrapper(*args, **kwargs):
        logger.info(f"[TOOL] Ejecutando tool: {name}")
        return tool_fn(*args, **kwargs)

    return async_wrapper if asyncio.iscoroutinefunction(tool_fn) else sync_wrapper

# =====================
# CLI
# =====================
INTRO = """
Chatbot RAG persistente — CSV Rotten Tomatoes
---------------------------------------------
Ejemplos:
 - "¿De qué trata The Godfather?"
 - "Top 5 por audience_rating del director Christopher Nolan"
 - "¿Cuáles son las 10 peores por tomatometer_rating?"
Comandos: /salir
"""

async def run_cli(agent: FunctionAgent):
    print(INTRO)
    while True:
        q = input("Tú> ").strip()
        if q.lower() in {"/salir", "salir", "exit", "quit"}:
            print("¡Adiós!")
            break
        if not q:
            continue
        try:
            t0 = datetime.now()
            resp = await agent.run(q, max_iterations=40)
            dt = (datetime.now() - t0).total_seconds()
            logger.info(f"[AGENT] ok | latency={dt:.3f}s | q='{q}'")
            print(f"\nBot>\n{resp}\n")
        except Exception as e:
            logger.exception("[AGENT] error")
            print(f"[ERROR] {e}")

# =====================
# MAIN
# =====================
def main():
    configure_models()
    df = load_dataframe()
    docs = make_documents_from_df(df)

    # Cargar o construir (si no existe); si existe con otro embedder → no reindexa y avisa.
    index = build_or_load_index(docs)

    # Tools
    raw_rag_search = build_rag_tool(index)          # async
    raw_movie_stats = build_movie_stats_tool(df)    # sync

    # Envolvemos tools para loguear cada invocación
    rag_search = wrap_tool(raw_rag_search, "rag_search")
    movie_stats = wrap_tool(raw_movie_stats, "movie_stats")

    # FunctionAgent con 2 tools
    agent = FunctionAgent(
        tools=[rag_search, movie_stats],
        llm=Settings.llm,
        system_prompt=(
            "Eres un asistente sobre películas basadas en un CSV de Rotten Tomatoes.\n"
            "Tienes SOLO dos herramientas: 'rag_search' y 'movie_stats'.\n"
            "- Usa 'rag_search' para preguntas narrativas o de sinopsis.\n"
            "- Usa 'movie_stats' para rankings o consultas tabulares.\n"
            "Devuelve SIEMPRE la respuesta final en texto plano en español, sin volver a llamar a ninguna tool.\n"
        )
    )

    asyncio.run(run_cli(agent))

if __name__ == "__main__":
    main()
