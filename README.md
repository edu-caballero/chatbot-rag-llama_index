# Chatbots RAG con LlamaIndex 🎬🤖

Este proyecto contiene **cinco chatbots** implementados con [LlamaIndex](https://www.llamaindex.ai/) sobre un dataset de [Rotten Tomatoes](https://www.rottentomatoes.com/).  
Cada chatbot explora diferentes features del framework: desde un RAG básico hasta un workflow con memoria persistente y reranker.  

---

## 📂 Archivos principales

- **chat.py**  
  Chatbot inicial de prueba (baseline simple).  

- **chat_2_rt.py**  
  Chatbot con `RouterQueryEngine`: mezcla un índice vectorial (RAG) y consultas tabulares con `PandasQueryEngine`.  

- **chat_3_rt_agents.py**  
  Chatbot con `FunctionAgent` y tools (`rag_search`, `movie_stats`), reemplazando al router.  

- **chat_4_memory_rerank.py**  
  Chatbot con agente + memoria persistente (`ChatMemoryBuffer`, `SimpleChatStore`) y un reranker cross-encoder para mayor precisión.  

- **chat_5_wf.py**  
  Chatbot más completo, organizado como workflow paso a paso. Integra:  
  - Configuración de modelos  
  - Carga de dataset (CSV Rotten Tomatoes)  
  - Creación de documentos y vector index  
  - Tools + agente  
  - Memoria persistente  
  - Reranker  
  - Loop interactivo  

---

## 🛠️ Requisitos

- Python 3.10+  
- Instalar dependencias principales:  

```bash
pip install "llama-index>=0.11" llama-index-llms-openai llama-index-embeddings-huggingface
pip install sentence-transformers pandas python-dotenv
