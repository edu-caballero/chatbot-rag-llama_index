#!/usr/bin/env python3
"""
Chatbot RAG híbrido (LlamaIndex) para dataset de películas (Rotten Tomatoes-like).

- RAG sobre `movie_info` (sinopsis/descripción).
- Consultas estructuradas (text-to-Pandas) sobre columnas típicas (géneros, ratings, etc.).
- Router decide a cuál motor enviar cada pregunta.
- Loop de chat con log de pasajes recuperados y prompts enviados al LLM.

"""

import os
import sys
import re
import hashlib
import pandas as pd
from dotenv import load_dotenv

# ---- LlamaIndex core ----
from llama_index.core import (
    Document, VectorStoreIndex, Settings, StorageContext, load_index_from_storage
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata

# LLM (OpenAI para generación)
from llama_index.llms.openai import OpenAI

# Embeddings (HuggingFace: gratis)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# QueryEngine para DataFrames (text-to-Pandas)
from llama_index.experimental.query_engine import PandasQueryEngine

# Callbacks para debug
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler

# ---------------------------
# 0) Configuración y helpers
# ---------------------------

DATA_PATH = "data/rotten_tomatoes_movies.csv"
PERSIST_DIR = "storage_2"
FINGERPRINT_FILE = os.path.join(PERSIST_DIR, ".embed_fingerprint.txt")

# Embedding fijo → fingerprint estable → NO reindexa si no cambia
EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"  # rápido (EN). Para queries en ES: "paraphrase-multilingual-MiniLM-L12-v2"

WELCOME = """
Chat RAG sobre películas. Escribí 'salir' para terminar.

Ejemplos:
 - ¿De qué trata 'Alien'?
 - Recomendame una de horror sin asesinatos ni gore.
 - Mostrá 5 títulos con género 'Horror'.
 - ¿Cuáles tienen audience_rating > 4.5 y tomatometer_rating > 80?
"""

ROUTING_KEYWORDS_STRUCT = {
    "rating", "ratings", "tomatometer", "audience", "genres", "genre", "content_rating",
    "year", "count", "how many", "cuántas", ">", "<", ">=", "<=", "top", "list", "titles",
}

def explain_route(query: str, used_engine_name: str) -> str:
    ql = query.lower()
    # heurística simple para “por qué”
    hits = [k for k in ROUTING_KEYWORDS_STRUCT if k in ql]
    if used_engine_name == "structured_movies":
        why = "parece una consulta tabular (columnas/valores/ratings/filtros)"
        if hits:
            why += f"; palabras clave detectadas: {', '.join(hits)}"
    else:
        why = "parece una consulta de texto libre narrativo (sinopsis/temas)"
    return f"[ROUTER] Estrategia: {used_engine_name}. Motivo: {why}."

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
# 1) Cargar CSV a DataFrame (estruct.)
# -------------------------------------

def parse_year(date_str: str) -> str:
    if not isinstance(date_str, str):
        return ""
    m = re.search(r"(\d{4})", date_str)
    return m.group(1) if m else ""

def load_dataframe() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)

    keep_cols = [
        "movie_title", "movie_info", "genres", "content_rating", "directors", "actors",
        "original_release_date", "runtime", "tomatometer_rating", "audience_rating"
    ]
    for c in keep_cols:
        if c not in df.columns:
            df[c] = ""
    df = df[keep_cols].copy()

    for c in keep_cols:
        df[c] = df[c].astype(str).fillna("")

    if "genres" in df.columns:
        df["genres_list"] = df["genres"].apply(
            lambda s: [t.strip() for t in re.split(r"[|,]", s) if t.strip()]
        )

    df["year"] = df["original_release_date"].apply(parse_year)

    def to_float(x):
        try:
            return float(x)
        except Exception:
            return None
    df["tomatometer_rating_num"] = df["tomatometer_rating"].apply(to_float)
    df["audience_rating_num"] = df["audience_rating"].apply(to_float)

    df = df[df["movie_info"].str.strip() != ""].reset_index(drop=True)

    MAX_ROWS = int(os.getenv("MAX_ROWS", "0") or "0")
    if MAX_ROWS > 0:
        df = df.head(MAX_ROWS).reset_index(drop=True)
        print(f"[INFO] Usando solo {MAX_ROWS} filas.")

    return df

def make_documents_from_df(df: pd.DataFrame) -> list:
    docs = []
    for _, r in df.iterrows():
        text = r.get("movie_info", "").strip()
        if not text:
            continue
        md = {
            "movie_title": r.get("movie_title", "").strip(),
            "genres": r.get("genres", "").strip(),
            "content_rating": r.get("content_rating", "").strip(),
            "directors": r.get("directors", "").strip(),
            "actors": r.get("actors", "").strip(),
            "year": r.get("year", "").strip(),
            "tomatometer_rating": r.get("tomatometer_rating", "").strip(),
            "audience_rating": r.get("audience_rating", "").strip(),
        }
        docs.append(Document(text=text, metadata=md))
    print(f"[INFO] Documentos creados para RAG: {len(docs)}")
    return docs

# ----------------------------------------------------
# 2) Configuración de modelos (LLM + Embeddings + NLP)
# ----------------------------------------------------

def configure_models():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("[ERROR] Falta OPENAI_API_KEY en .env")
        sys.exit(1)

    Settings.llm = OpenAI(
        model="gpt-4o-mini-2024-07-18",
        temperature=0.2,
    )

    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    except Exception:
        device = "cpu"

    Settings.embed_model = HuggingFaceEmbedding(
        model_name=EMBED_MODEL_ID,
        device=device,
        embed_batch_size=64
    )
    print(f"[INFO] Embeddings {EMBED_MODEL_ID} en device={device}, batch=64")

    Settings.node_parser = SentenceSplitter(chunk_size=2048, chunk_overlap=64)

    # --- Debug Handler para loguear prompts completos ---
    debug_handler = LlamaDebugHandler(print_trace_on_end=True)
    Settings.callback_manager = CallbackManager([debug_handler])

# --------------------------------------------
# 3) Construir índice RAG (vectorial) persist.
# --------------------------------------------

def build_or_load_vector_index(docs: list) -> VectorStoreIndex:
    provider = "hf"
    model_id = EMBED_MODEL_ID
    current_fp = embed_fingerprint(provider, model_id)
    stored_fp = read_fingerprint()
    print(f"[DEBUG] stored_fp={stored_fp} current_fp={current_fp}")

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
        "You have a pandas DataFrame named df with columns: "
        "movie_title (str), movie_info (str), genres (str), genres_list (list[str]), "
        "content_rating (str), directors (str), actors (str), year (str), runtime (str), "
        "tomatometer_rating (str), audience_rating (str), "
        "tomatometer_rating_num (float), audience_rating_num (float). "
        "\n\nTASK: Translate the user request into ONE valid Python expression over df."
        "\nThe expression must directly compute the answer."
        "\n\nOUTPUT RULES (STRICT):"
        "\n- OUTPUT ONLY a single Python expression."
        "\n- NO backticks, NO Markdown fences, NO comments, NO explanations."
        "\n- Prefer returning a small DataFrame with informative columns "
        "  (e.g., ['movie_title','tomatometer_rating_num','audience_rating_num','year'])."
        "\n- For short lists use .head(N)."
        "\n- Use df['genres_list'].apply(lambda xs: 'Horror' in xs) for genre membership."
        "\n- Use numeric columns for comparisons (tomatometer_rating_num, audience_rating_num)."
        "\n\nEXAMPLES:"
        "\n# worst rated movie (tomatometer) for an actor (case-insensitive):"
        "\n(df[df['actors'].str.contains('Robert De Niro', case=False, na=False)]"
        "\n   .sort_values('tomatometer_rating_num', ascending=True)"
        "\n   [['movie_title','tomatometer_rating_num','year']].head(1))"
        "\n\n# top-5 comedy titles with ratings:"
        "\n(df[df['genres_list'].apply(lambda xs: 'Comedy' in xs)]"
        "\n   [['movie_title','tomatometer_rating_num','audience_rating_num','year']].head(5))"
    )

    pandas_engine = PandasQueryEngine(
        df=df,
        verbose=False,
        instruction_str=schema_hint
    )

    return QueryEngineTool(
        query_engine=pandas_engine,
        metadata=ToolMetadata(
            name="structured_movies",
            description="Consultas tabulares (géneros, ratings, filtros por columnas)."
        ),
    )

def build_rag_engine(index: VectorStoreIndex) -> QueryEngineTool:
    rag_engine = index.as_query_engine(similarity_top_k=3, verbose=True)
    return QueryEngineTool(
        query_engine=rag_engine,
        metadata=ToolMetadata(
            name="synopsis_rag",
            description="Preguntas de texto libre sobre sinopsis/temas narrativos."
        ),
    )

def build_router(tools: list) -> RouterQueryEngine:
    return RouterQueryEngine.from_defaults(
        query_engine_tools=tools,
        select_multi=False,
    )

# -------------------------
# 5) Loop de chat (consola)
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

            used_engine = "synopsis_rag" if (hasattr(ans, "source_nodes") and ans.source_nodes) else "structured_movies"
            print(explain_route(q, used_engine))

            # Mostrar pasajes recuperados (si fue RAG)
            if hasattr(ans, "source_nodes") and ans.source_nodes:
                print("\n[DEBUG] Pasajes recuperados (contexto RAG):")
                for i, sn in enumerate(ans.source_nodes, 1):
                    meta = sn.node.metadata or {}
                    title = meta.get("movie_title", meta.get("title", "(sin título)"))
                    snippet = sn.node.get_content()[:200].replace("\n", " ")
                    print(f"  {i}. {title} | {snippet}...")
                print()

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
        print("[ERROR] El DataFrame está vacío o no tiene 'movie_info'.")
        sys.exit(1)

    docs = make_documents_from_df(df)
    if not docs:
        print("[ERROR] No se generaron documentos para RAG (revisá 'movie_info').")
        sys.exit(1)

    index = build_or_load_vector_index(docs)

    structured_tool = build_structured_engine(df)
    rag_tool = build_rag_engine(index)
    router = build_router([structured_tool, rag_tool])

    run_chat(router)

if __name__ == "__main__":
    main()
