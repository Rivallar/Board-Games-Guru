import os
from dotenv import load_dotenv

load_dotenv()

SMALL_LLM = os.getenv("SMALL_LLM", "qwen3:4b-q4_K_M")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3:8b")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./storage/chroma")

MAX_CHUNK_TOKENS = int(os.getenv("MAX_CHUNK_TOKENS", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

TOP_K = int(os.getenv("TOP_K", "8"))
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "6"))

LLM_KWARGS = {
    "temperature": 0.2,
    "top_p": 0.9,
    "repeat_penalty": 1.05,
    "num_ctx": 8192,
    "request_timeout": 120,
}
