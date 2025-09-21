#!/usr/bin/env python3
"""
Chatbot RAG híbrido (LlamaIndex) para el dataset MPST (Movie Plot Synopses with Tags).

- RAG (embeddings + similitud) para preguntas de texto libre sobre sinopsis/tramas.
- Consultas estructuradas (text-to-Pandas) para contar/filtrar por columnas del CSV.
- Un Router decide a cuál motor enviar cada pregunta.
- Loop de chat en consola.

CSV esperado (columnas):
  imdb_id, title, plot_synopsis, tags, split, synopsis_source

Variables de entorno (.env):
  OPENAI_API_KEY=sk-...

Ejecución:
  python app.py
"""

import os
import sys
import hashlib
import pandas as pd
from dotenv import load_dotenv

# ---- LlamaIndex: core, LLM y embeddings ----
from llama_index.core import (
    Document, VectorStoreIndex, Settings, StorageContext
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata

# LLM (OpenAI)
from llama_index.llms.openai import OpenAI

# Embeddings HuggingFace (gratis)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# QueryEngine para DataFrames
from llama_index.experimental.query_engine import PandasQueryEngine
from sympy.printing.pytorch import torch

from llama_index.core import load_index_from_storage

# ---------------------------
# 0) Configuración y helpers
# ---------------------------

DATA_PATH = "data/mpst.csv"
PERSIST_DIR = "storage"
FINGERPRINT_FILE = os.path.join(PERSIST_DIR, ".embed_fingerprint.txt")
EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"  # o el que elijas

WELCOME = """
Chat RAG sobre MPST. Escribí 'salir' para terminar.

Ejemplos:
 - ¿De qué trata 'Inception'?
 - Recomendame una película con tags 'revenge|romance'.
 - ¿Cuántas películas tienen el tag 'revenge'?
 - Mostrá 10 títulos con el tag 'comedy'.
"""

def ensure_paths():
    if not os.path.exists(DATA_PATH):
        print(f"[ERROR] No se encontró el CSV en: {DATA_PATH}")
        sys.exit(1)
    os.makedirs(PERSIST_DIR, exist_ok=True)

def read_fingerprint() -> str:
    try:
        with open(FINGERPRINT_FILE, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        return ""

def write_fingerprint(fp: str) -> None:
    with open(FINGERPRINT_FILE, "w", encoding="utf-8") as f:
        f.write(fp)

def embed_fingerprint(provider: str, model_name: str) -> str:
    base = f"{provider}:{model_name}".encode("utf-8")
    return hashlib.sha256(base).hexdigest()[:16]

# -------------------------------------
# 1) Cargar CSV a DataFrame
# -------------------------------------

def load_dataframe() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)

    for col in ["imdb_id", "title", "plot_synopsis", "tags", "split", "synopsis_source"]:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna("")

    if "tags" in df.columns:
        df["tags_list"] = df["tags"].apply(
            lambda s: [t.strip() for t in s.split("|") if t.strip()]
        )

    if "plot_synopsis" in df.columns:
        df = df[df["plot_synopsis"].str.strip() != ""].reset_index(drop=True)

    return df

def make_documents_from_df(df: pd.DataFrame) -> list:
    docs = []
    for _, r in df.iterrows():
        text = r.get("plot_synopsis", "")
        if not text.strip():
            continue
        md = {
            "imdb_id": r.get("imdb_id", "").strip(),
            "title": r.get("title", "").strip(),
            "tags": r.get("tags", "").strip(),
            "split": r.get("split", "").strip(),
            "synopsis_source": r.get("synopsis_source", "").strip(),
        }
        docs.append(Document(text=text, metadata=md))
    print(f"[INFO] Documentos creados para RAG: {len(docs)}")
    return docs

# ----------------------------------------------------
# 2) Configuración de modelos (LLM + Embeddings)
# ----------------------------------------------------

def configure_models():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("[ERROR] Falta OPENAI_API_KEY en .env")
        sys.exit(1)

    # LLM de OpenAI (chat)
    Settings.llm = OpenAI(
        model="gpt-4o-mini-2024-07-18",
        temperature=0.2,
    )

    # Embeddings: SIEMPRE HuggingFace (gratis, multilingüe)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    Settings.embed_model = HuggingFaceEmbedding(
        model_name=EMBED_MODEL_ID,
        device=device,
        embed_batch_size=64  # podés bajar a 32 si tu máquina es justa
    )
    print(f"[INFO] Embeddings {EMBED_MODEL_ID} en device={device}, batch=64")

    # Chunking
    Settings.node_parser = SentenceSplitter(chunk_size=1000, chunk_overlap=120)

# --------------------------------------------
# 3) Construir índice RAG (vectorial) persist.
# --------------------------------------------

def build_or_load_vector_index(docs: list) -> VectorStoreIndex:
    provider = "hf"
    model_id = EMBED_MODEL_ID
    current_fp = embed_fingerprint(provider, model_id)
    stored_fp = read_fingerprint()

    if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR) and stored_fp == current_fp:
        try:
            storage_ctx = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
            index = load_index_from_storage(storage_ctx)
            print(f"[OK] Índice cargado desde 'storage/' (fp {provider}:{model_id}).")
            return index
        except Exception as e:
            print(f"[WARN] No se pudo cargar índice existente: {e}. Reindexando…")

    print(f"[INFO] Construyendo índice vectorial con {provider}:{model_id}…")
    idx = VectorStoreIndex.from_documents(docs)
    idx.storage_context.persist(persist_dir=PERSIST_DIR)
    write_fingerprint(current_fp)
    print(f"[OK] Índice creado y persistido (fp {provider}:{model_id}).")
    return idx

# ---------------------------------------
# 4) Motores: estructurado y RAG + router
# ---------------------------------------

def build_structured_engine(df: pd.DataFrame) -> QueryEngineTool:
    schema_hint = (
        "Tenés un DataFrame llamado df con columnas: "
        "imdb_id, title, plot_synopsis, tags, tags_list, split, synopsis_source. "
        "NO usar imports ni dunder (__xxx__). Usar solo pandas. "
        "Ejemplos: "
        " - df[df['tags_list'].apply(lambda xs: 'comedy' in xs)]['title'].head(5).tolist() "
        " - df[df['split']=='test']['title'].head(10).tolist() "
        " - len(df)"
    )
    pandas_engine = PandasQueryEngine(df=df, verbose=False, instruction_str=schema_hint)
    return QueryEngineTool(
        query_engine=pandas_engine,
        metadata=ToolMetadata(
            name="structured_movies",
            description="Consultas sobre columnas/valores del CSV (tags, split, etc.)."
        ),
    )

def build_rag_engine(index: VectorStoreIndex) -> QueryEngineTool:
    rag_engine = index.as_query_engine(similarity_top_k=6)
    return QueryEngineTool(
        query_engine=rag_engine,
        metadata=ToolMetadata(
            name="synopsis_rag",
            description="Preguntas de texto libre sobre sinopsis/tramas."
        ),
    )

def build_router(tools: list) -> RouterQueryEngine:
    return RouterQueryEngine.from_defaults(
        query_engine_tools=tools,
        select_multi=False,
    )

# -------------------------
# 5) Loop de chat
# -------------------------

def run_chat(router: RouterQueryEngine):
    print(WELCOME)
    while True:
        try:
            q = input("Tú: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[Adiós!]")
            break
        if q.lower() in {"salir", "exit", "quit"}:
            print("[Adiós!]")
            break
        if not q:
            continue
        try:
            ans = router.query(q)
            print("Bot:", getattr(ans, "response", str(ans)))
        except Exception as e:
            print(f"[ERROR] {e}")

# -------------
# 6) Main
# -------------

def main():
    ensure_paths()
    configure_models()
    df = load_dataframe()
    if df.empty:
        print("[ERROR] El DataFrame está vacío o no tiene 'plot_synopsis'.")
        sys.exit(1)
    docs = make_documents_from_df(df)
    if not docs:
        print("[ERROR] No se generaron documentos para RAG.")
        sys.exit(1)
    index = build_or_load_vector_index(docs)
    structured_tool = build_structured_engine(df)
    rag_tool = build_rag_engine(index)
    router = build_router([structured_tool, rag_tool])
    run_chat(router)

if __name__ == "__main__":
    main()
