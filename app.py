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
  # (opcionales si usás Projects/Organizations)
  OPENAI_ORGANIZATION=org_xxx
  OPENAI_PROJECT=proj_xxx

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

# Embeddings (dos opciones: OpenAI u HuggingFace)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# QueryEngine para DataFrames (text-to-Pandas)
from llama_index.experimental.query_engine import PandasQueryEngine

# ---------------------------
# 0) Configuración y helpers
# ---------------------------

DATA_PATH = "data/mpst.csv"
PERSIST_DIR = "storage"           # carpeta de persistencia del índice vectorial
FINGERPRINT_FILE = os.path.join(PERSIST_DIR, ".embed_fingerprint.txt")

WELCOME = """
Chat RAG sobre MPST. Escribí 'salir' para terminar.

Ejemplos:
 - ¿De qué trata 'Inception'?
 - Recomendame una película con tags 'revenge|romance'.
 - ¿Cuántas películas tienen el tag 'revenge'?
 - Mostrá 10 títulos con el tag 'comedy'.
"""

def ensure_paths():
    """Verifica rutas mínimas y crea storage si no existe."""
    if not os.path.exists(DATA_PATH):
        print(f"[ERROR] No se encontró el CSV en: {DATA_PATH}")
        print("Asegurate de colocar mpst.csv en la carpeta data/ o ajustá DATA_PATH.")
        sys.exit(1)
    os.makedirs(PERSIST_DIR, exist_ok=True)

def read_fingerprint() -> str:
    """Lee el fingerprint de embeddings guardado en storage/."""
    try:
        with open(FINGERPRINT_FILE, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        return ""

def write_fingerprint(fp: str) -> None:
    """Escribe el fingerprint de embeddings en storage/."""
    with open(FINGERPRINT_FILE, "w", encoding="utf-8") as f:
        f.write(fp)

def embed_fingerprint(provider: str, model_name: str) -> str:
    """
    Genera un fingerprint simple para detectar cambios de embedding
    (proveedor + modelo) y reindexar automáticamente si cambian.
    """
    base = f"{provider}:{model_name}".encode("utf-8")
    return hashlib.sha256(base).hexdigest()[:16]

# -------------------------------------
# 1) Cargar CSV a DataFrame (estruct.)
# -------------------------------------

def load_dataframe() -> pd.DataFrame:
    """
    Carga el CSV a un DataFrame y normaliza tipos.
    Añade 'tags_list' para facilitar consultas.
    """
    df = pd.read_csv(DATA_PATH)

    for col in ["imdb_id", "title", "plot_synopsis", "tags", "split", "synopsis_source"]:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna("")

    # Lista de tags
    if "tags" in df.columns:
        df["tags_list"] = df["tags"].apply(
            lambda s: [t.strip() for t in s.split("|") if t.strip()]
        )

    # Quitar filas sin sinopsis (no sirven para RAG)
    if "plot_synopsis" in df.columns:
        df = df[df["plot_synopsis"].str.strip() != ""].reset_index(drop=True)

    return df

def make_documents_from_df(df: pd.DataFrame) -> list:
    """
    Convierte filas del DataFrame en Document(s) para el índice RAG.
    - text = plot_synopsis
    - metadata = imdb_id, title, tags, split, synopsis_source
    """
    docs = []
    for _, r in df.iterrows():
        text = r.get("plot_synopsis", "")
        if not text or str(text).strip() == "":
            continue

        md = {
            "imdb_id": r.get("imdb_id", "").strip(),
            "title": r.get("title", "").strip(),
            "tags": r.get("tags", "").strip(),
            "split": r.get("split", "").strip(),
            "synopsis_source": r.get("synopsis_source", "").strip(),
        }
        docs.append(Document(text=str(text), metadata=md))

    print(f"[INFO] Documentos creados para RAG: {len(docs)}")
    return docs

# ----------------------------------------------------
# 2) Configuración de modelos (LLM + Embeddings + NLP)
# ----------------------------------------------------

def configure_models():
    """
    Configura:
      - LLM de OpenAI para generación (requiere OPENAI_API_KEY)
      - Modelo de embeddings:
          * OpenAI (si disponible)
          * Fallback automático a HuggingFace si hay 403/permiso faltante
      - Node parser (chunking)
    """
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("[ERROR] Falta OPENAI_API_KEY en .env")
        sys.exit(1)

    # LLM generativo OpenAI (podés ajustar modelo/costo/calidad)
    Settings.llm = OpenAI(
        model="gpt-4o-mini-2024-07-18",
        temperature=0.2,
    )

    # Definimos un embedding por defecto (OpenAI) y un fallback local (HF)
    # NOTA: Usaremos OpenAIEmbedding primero; si al indexar hay 403, reintentamos con HF.
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

    # Chunking: partir sinopsis largas en fragmentos
    Settings.node_parser = SentenceSplitter(chunk_size=1000, chunk_overlap=120)

# --------------------------------------------
# 3) Construir índice RAG (vectorial) persist.
# --------------------------------------------

def try_build_index_with_current_embed(docs: list) -> VectorStoreIndex:
    """
    Intenta construir un índice con el embedding configurado en Settings.embed_model.
    Si falla por 403 de OpenAI, levanta excepción para que el caller haga fallback.
    """
    index = VectorStoreIndex.from_documents(docs)
    index.storage_context.persist(persist_dir=PERSIST_DIR)
    return index

# --- NUEVO: util para identificar el embedding activo de forma robusta ---
def resolve_embedding_id():
    """
    Devuelve (provider, model_id) del embedding activo en Settings.embed_model
    sin depender de atributos frágiles. Soporta OpenAIEmbedding y HF.
    """
    em = Settings.embed_model
    # Detectar proveedor por clase
    if em.__class__.__name__.lower().startswith("openaiencod") or em.__class__.__name__.lower().startswith("openaie"):
        provider = "openai"
    elif em.__class__.__name__.lower().startswith("huggingface"):
        provider = "hf"
    else:
        provider = em.__class__.__name__.lower()

    # Intentar distintos nombres de atributo en distintas versiones
    possible_attrs = ["model", "model_name", "model_name_or_path", "_model", "_model_name"]
    model_id = None
    for a in possible_attrs:
        if hasattr(em, a):
            try:
                v = getattr(em, a)
                model_id = v() if callable(v) else v
                if model_id:
                    break
            except Exception:
                pass

    # Fallbacks: mirar repr o clase si no encontramos nada
    if not model_id:
        # intentar parsear del repr
        try:
            model_id = repr(em)
        except Exception:
            model_id = em.__class__.__name__

    # Normalizar a str corto
    model_id = str(model_id)
    if len(model_id) > 80:
        model_id = model_id[:80]

    return provider, model_id


def build_or_load_vector_index(docs: list) -> VectorStoreIndex:
    """
    Crea (o carga) un índice vectorial persistente. Si hay índice existente, lo carga
    sólo si el fingerprint de embeddings coincide; si no, reindexa.
    - Fallback automático a HuggingFaceEmbedding si OpenAI embeddings da 403.
    """
    # --- usar el resolver robusto ---
    provider, model_id = resolve_embedding_id()
    current_fp = embed_fingerprint(provider, model_id)
    stored_fp = read_fingerprint()

    # Intento de carga si existe storage y fingerprint coincide
    if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR) and stored_fp == current_fp:
        try:
            storage_ctx = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
            index = VectorStoreIndex.from_storage(storage_ctx)
            print(f"[OK] Índice vectorial cargado desde 'storage/' (fp {provider}:{model_id}).")
            return index
        except Exception as e:
            print(f"[WARN] No se pudo cargar índice existente: {e}. Se recreará.")

    # Si llegamos acá, hay que (re)construir
    print(f"[INFO] Construyendo índice vectorial con {provider}:{model_id}…")
    try:
        idx = try_build_index_with_current_embed(docs)
        write_fingerprint(current_fp)
        print(f"[OK] Índice creado y persistido (fp {provider}:{model_id}).")
        return idx
    except Exception as e:
        msg = str(e).lower()
        if "403" in msg or "forbidden" in msg or "insufficient" in msg:
            print("[WARN] Falla con OpenAI embeddings (403/permiso). Probando fallback HuggingFace…")
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding
            Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

            # recalcular fingerprint con el nuevo embedding
            provider_hf, model_hf = resolve_embedding_id()
            current_fp_hf = embed_fingerprint(provider_hf, model_hf)

            try:
                idx = try_build_index_with_current_embed(docs)
                write_fingerprint(current_fp_hf)
                print(f"[OK] Índice creado con HF y persistido (fp {provider_hf}:{model_hf}).")
                return idx
            except Exception as e2:
                print(f"[ERROR] Fallback HF también falló: {e2}")
                sys.exit(1)
        else:
            print(f"[ERROR] No se pudo construir el índice: {e}")
            sys.exit(1)

# ---------------------------------------
# 4) Motores: estructurado y RAG + router
# ---------------------------------------

def build_structured_engine(df: pd.DataFrame) -> QueryEngineTool:
    """
    Motor de consultas estructuradas (text-to-Pandas) con guías estrictas.
    Regla: NO usar imports, dunder (__xxx__), eval/exec, ni builtins peligrosos.
    Trabajar solo con el DataFrame `df` ya provisto.
    """
    schema_hint = (
        "Tenés un DataFrame llamado df con columnas: "
        "imdb_id (str), title (str), plot_synopsis (str), "
        "tags (str con '|' como separador), tags_list (list[str]), "
        "split (str), synopsis_source (str). "
        "RESTRICCIONES ESTRICTAS: "
        " - NO usar imports ni dunder (__xxx__). "
        " - NO usar eval/exec ni builtins peligrosos. "
        " - Usar solo operaciones de pandas sobre df. "
        " - Para trabajar con tags, usá df['tags_list']. "
        "EJEMPLOS PERMITIDOS: "
        " - Contar por tag: df[df['tags_list'].apply(lambda xs: 'comedy' in xs)].shape[0] "
        " - Listar títulos por tag: df[df['tags_list'].apply(lambda xs: 'comedy' in xs)]['title'].head(5).tolist() "
        " - Filtrar por split: df[df['split']=='test']['title'].head(10).tolist() "
        " - Contar total: len(df) "
        "Devolvé resultados concisos y si devolvés listas, que sean cortas (top-N)."
    )
    pandas_engine = PandasQueryEngine(df=df, verbose=False, instruction_str=schema_hint)
    return QueryEngineTool(
        query_engine=pandas_engine,
        metadata=ToolMetadata(
            name="structured_movies",
            description=("Consultas sobre columnas/valores del CSV: conteos por tag, "
                         "búsquedas por título, filtros por split/synopsis_source.")
        ),
    )

def build_rag_engine(index: VectorStoreIndex) -> QueryEngineTool:
    """
    Motor RAG (vectorial) sobre sinopsis.
    similarity_top_k define cuántos fragmentos relevantes recuperar.
    """
    rag_engine = index.as_query_engine(similarity_top_k=6)
    return QueryEngineTool(
        query_engine=rag_engine,
        metadata=ToolMetadata(
            name="synopsis_rag",
            description=("Usa esto para preguntas de texto libre sobre tramas/sinopsis, "
                         "recomendaciones temáticas y descripciones narrativas de películas.")
        ),
    )

def build_router(tools: list) -> RouterQueryEngine:
    """
    Router que decide automáticamente a qué motor enviar la consulta.
    Usa un prompt interno para clasificar intención (estructurado vs libre).
    """
    router = RouterQueryEngine.from_defaults(
        query_engine_tools=tools,
        select_multi=False,   # enviamos a un solo motor por pregunta
    )
    return router

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
            print("Bot:", getattr(ans, "response", str(ans)))
        except Exception as e:
            # Mensajes más amigables si falla el LLM
            emsg = str(e)
            if "403" in emsg:
                print("[ERROR] Respuesta 403 del proveedor LLM/embeddings. Verificá tu plan/permisos.")
            else:
                print(f"[ERROR] Ocurrió un problema respondiendo: {e}")

# -------------
# 6) Main
# -------------

def main():
    ensure_paths()
    configure_models()

    # Cargar CSV a DataFrame
    df = load_dataframe()
    if df.empty:
        print("[ERROR] El DataFrame está vacío o no tiene 'plot_synopsis' con contenido.")
        sys.exit(1)

    # Construir documentos para RAG + índice vectorial
    docs = make_documents_from_df(df)
    if not docs:
        print("[ERROR] No se generaron documentos para RAG. Revisá la columna 'plot_synopsis'.")
        sys.exit(1)

    # Construir/cargar índice
    index = build_or_load_vector_index(docs)

    # Motores (estructurado y RAG)
    structured_tool = build_structured_engine(df)
    rag_tool = build_rag_engine(index)

    # Router (elige a qué motor enviar cada pregunta)
    router = build_router([structured_tool, rag_tool])

    # Iniciar chat en consola
    run_chat(router)

if __name__ == "__main__":
    main()
