#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chatbot RAG (Rotten Tomatoes) con FunctionAgent + Context + ChatMemoryBuffer (persistente) + Reranker.

- Usa SÓLO el agente (FunctionAgent) como en el ejemplo:
    ctx = Context(agent)
    await agent.run(prompt, ctx=ctx, memory=memory)

- NO reindexa si ya existe el índice (persist_dir="storage_4").
- Memoria persistente en JSON (ChatMemoryBuffer + SimpleChatStore).

ENV opcionales:
- OPENAI_API_KEY=...
- CSV_PATH=data/rotten_tomatoes_movies.csv
- EMBED_LOCAL_PATH= (ruta a modelo local HuggingFace, opcional)
- PERSIST_DIR=storage_4
- CHAT_STORE_PATH=storage_4/chat_store.json
- CHAT_STORE_KEY=movies_rag_session
- MEMORY_TOKEN_LIMIT=3900
- RETRIEVER_TOP_K=12
- RERANKER_TOP_N=4
- RERANKER_MODEL_ID=cross-encoder/ms-marco-MiniLM-L-6-v2
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

# LlamaIndex core
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
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.storage.chat_store import SimpleChatStore
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.workflow import Context

# LLM/Embeddings
from llama_index.llms.openai import OpenAI as OpenAILLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Reranker (vía extra SBERT)
from llama_index.postprocessor.sbert_rerank import SentenceTransformerRerank

# =========
# Config
# =========
load_dotenv()

CSV_PATH = os.getenv("CSV_PATH", "data/rotten_tomatoes_movies.csv")
PERSIST_DIR = os.getenv("PERSIST_DIR", "storage_4")
os.makedirs(PERSIST_DIR, exist_ok=True)

FINGERPRINT_FILE = os.path.join(PERSIST_DIR, ".embed_fingerprint.txt")

CHAT_STORE_PATH = os.getenv("CHAT_STORE_PATH", os.path.join(PERSIST_DIR, "chat_store.json"))
CHAT_STORE_KEY = os.getenv("CHAT_STORE_KEY", "movies_rag_session")
MEMORY_TOKEN_LIMIT = int(os.getenv("MEMORY_TOKEN_LIMIT", "3900"))

OPENAI_LLM_MODEL = "gpt-4o-mini-2024-07-18"

EMBED_LOCAL_PATH = os.getenv("EMBED_LOCAL_PATH", "").strip()
DEFAULT_EMBED_ID = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_MODEL_ID = EMBED_LOCAL_PATH if EMBED_LOCAL_PATH else DEFAULT_EMBED_ID

RETRIEVER_TOP_K = int(os.getenv("RETRIEVER_TOP_K", "12"))
RERANKER_TOP_N = int(os.getenv("RERANKER_TOP_N", "4"))
RERANKER_MODEL_ID = os.getenv("RERANKER_MODEL_ID", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# -------------
# Logging
# -------------
logger = logging.getLogger("movies_rag_agent_only")
logger.setLevel(logging.INFO)
fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
ch = logging.StreamHandler(); ch.setFormatter(fmt); ch.setLevel(logging.INFO)
fh = logging.FileHandler(os.path.join(PERSIST_DIR, "movies_rag.log"), encoding="utf-8")
fh.setFormatter(fmt); fh.setLevel(logging.INFO)
logger.addHandler(ch); logger.addHandler(fh)

# =====================
# Persistencia: índice
# =====================
def embed_fingerprint(provider: str, model_name: str) -> str:
    return hashlib.sha256(f"{provider}:{model_name}".encode("utf-8")).hexdigest()[:16]

def read_fingerprint() -> str:
    try:
        with open(FINGERPRINT_FILE, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        return ""

def write_fingerprint(fp: str) -> None:
    with open(FINGERPRINT_FILE, "w", encoding="utf-8") as f:
        f.write(fp)

# =====================
# Modelos / Settings
# =====================
def configure_models():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Falta OPENAI_API_KEY en el entorno o .env")

    Settings.llm = OpenAILLM(
        model=OPENAI_LLM_MODEL,
        api_key=api_key,
        temperature=0.2,
    )
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=EMBED_MODEL_ID,
        device="cpu"  # cambia a "cuda" si tenés GPU
    )
    Settings.node_parser = SentenceSplitter(
        chunk_size=2048,
        chunk_overlap=64,
        include_metadata=False,
        include_prev_next_rel=False,
    )
    debug_handler = LlamaDebugHandler(print_trace_on_end=True)
    Settings.callback_manager = CallbackManager([debug_handler])

# =====================
# CSV → DataFrame
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

    for c in EXPECTED_COLS:
        if c in df.columns and c not in ("tomatometer_rating", "audience_rating"):
            df[c] = df[c].astype(str).fillna("")
    df["tomatometer_rating"] = pd.to_numeric(df["tomatometer_rating"], errors="coerce")
    df["audience_rating"] = pd.to_numeric(df["audience_rating"], errors="coerce")

    df["year"] = df["original_release_date"].apply(
        lambda s: (re.search(r"(\d{4})", s).group(1) if isinstance(s, str) and re.search(r"(\d{4})", s) else "")
    )
    df = df[df["movie_info"].astype(str).str.strip() != ""].reset_index(drop=True)
    logger.info(f"CSV cargado: {CSV_PATH} | Filas: {len(df)} | Columnas: {len(df.columns)}")
    return df

# =====================
# DF → Documentos
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
# Cargar o construir índice (NO reindex si existe)
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
            raise RuntimeError(
                "Hay un índice persistido con fingerprint distinto (otro modelo de embeddings). "
                "Por política de 'no reindexar', aborto. Borra el directorio si querés reconstruir."
            )

    logger.info("No hay índice persistido válido. Construyendo por única vez…")
    index = VectorStoreIndex.from_documents(docs, embed_model=Settings.embed_model)
    index.storage_context.persist(persist_dir=PERSIST_DIR)
    write_fingerprint(current_fp)
    logger.info(f"Índice creado y persistido en {PERSIST_DIR}")
    return index

# =====================
# Memoria (persistente)
# =====================
def load_chat_memory() -> ChatMemoryBuffer:
    try:
        chat_store = SimpleChatStore.from_persist_path(CHAT_STORE_PATH)
    except Exception:
        chat_store = SimpleChatStore()
    memory = ChatMemoryBuffer.from_defaults(
        token_limit=MEMORY_TOKEN_LIMIT,
        chat_store=chat_store,
        chat_store_key=CHAT_STORE_KEY,
    )
    return memory

def persist_chat_memory(memory: ChatMemoryBuffer) -> None:
    # guarda el chat_store en disco
    memory.chat_store.persist(persist_path=CHAT_STORE_PATH)

# =====================
# Tools del agente
# =====================
def build_rag_tool(index: VectorStoreIndex):
    reranker = SentenceTransformerRerank(model=RERANKER_MODEL_ID, top_n=RERANKER_TOP_N)

    query_engine = index.as_query_engine(
        similarity_top_k=RETRIEVER_TOP_K,
        node_postprocessors=[reranker],
        verbose=True,
    )

    async def rag_search(query: str) -> str:
        t0 = datetime.now()
        resp = await query_engine.aquery(query)
        dt = (datetime.now() - t0).total_seconds()

        fuentes = []
        for sn in getattr(resp, "source_nodes", []) or []:
            meta = sn.node.metadata or {}
            title = meta.get("movie_title") or "(sin título)"
            score = getattr(sn, "score", None)
            try:
                fuentes.append(f"- {title} (score={score:.3f})")
            except Exception:
                fuentes.append(f"- {title}")

        answer = str(resp)
        if fuentes:
            answer += "\n\nFuentes recuperadas (tras rerank):\n" + "\n".join(fuentes)
        answer += f"\n\n[diag] top_k={RETRIEVER_TOP_K} | rerank_top_n={RERANKER_TOP_N} | latency={dt:.3f}s"
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

# =====================
# CLI (usa sólo agent + ctx + memory)
# =====================
INTRO = """
Chatbot RAG — CSV Rotten Tomatoes (FunctionAgent + Context + Memory)
-------------------------------------------------------------------
Ejemplos:
 - "¿De qué trata The Godfather?"
 - "Top 5 por audience_rating del director Christopher Nolan"
 - "¿Cuáles son las 10 peores por tomatometer_rating?"

Comandos: /salir | /hist (ver historial) | /reset (limpia memoria)
"""

async def run_cli(agent: FunctionAgent, memory: ChatMemoryBuffer):
    print(INTRO)
    ctx = Context(agent)  # contexto para mantener estado del agente (igual que en el ejemplo)

    while True:
        q = input("Tú> ").strip()
        if not q:
            continue

        if q.lower() in {"/salir", "salir", "exit", "quit"}:
            persist_chat_memory(memory)
            print("¡Adiós!")
            break

        if q.lower() in {"/hist", "/history"}:
            msgs = memory.get()
            if not msgs:
                print("\nBot>\n(Historial vacío)\n")
            else:
                print("\nBot> Historial:")
                for m in msgs[-12:]:
                    role = "Tú" if m.role == MessageRole.USER else "Bot"
                    print(f"- {role}: {m.content[:200]}")
                print()
            continue

        if q.lower() in {"/reset"}:
            memory.chat_store.delete_messages(CHAT_STORE_KEY)
            print("\nBot>\nMemoria borrada para esta sesión.\n")
            continue

        try:
            t0 = datetime.now()
            # Ejecuta el agente EXACTAMENTE como en tu ejemplo (con ctx y memory)
            resp = await agent.run(q, ctx=ctx, memory=memory)
            dt = (datetime.now() - t0).total_seconds()

            print(f"\nBot>\n{resp}\n")
            persist_chat_memory(memory)
            logger.info(f"[AGENT] ok | latency={dt:.3f}s | q='{q}'")

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

    # Cargar o construir índice (NO reindex si ya existe)
    index = build_or_load_index(docs)

    # Tools
    rag_search = build_rag_tool(index)
    movie_stats = build_movie_stats_tool(df)

    # Agent (sin .chat; sólo .run con ctx + memory)
    agent = FunctionAgent(
        tools=[rag_search, movie_stats],
        llm=Settings.llm,
        system_prompt=(
            "Eres un asistente sobre películas basadas en un CSV de Rotten Tomatoes.\n"
            "- Si la pregunta es abierta (sinopsis/consenso/elenco), usa 'rag_search'.\n"
            "- Si pide rankings o filtros de columnas (top/bottom, director, género, etc.), usa 'movie_stats'.\n"
            "Responde en español y, si usas RAG, lista 'Fuentes recuperadas'."
        ),
    )

    # Memoria persistente + CLI
    memory = load_chat_memory()
    asyncio.run(run_cli(agent, memory))

if __name__ == "__main__":
    main()
