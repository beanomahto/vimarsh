import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_PERSIST_DIR = str(BASE_DIR / "chroma_db")
DOCUMENTS_DIR = str(BASE_DIR / "documents")

PROVIDERS = {
    "groq": {
        "api_key": GROQ_API_KEY,
        "base_url": "https://api.groq.com/openai/v1",
        "models": [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "meta-llama/llama-4-scout-17b-16e-instruct",
            "meta-llama/llama-4-maverick-17b-128e-instruct",
            "qwen/qwen-3-32b",
        ],
        "default_model": "llama-3.3-70b-versatile",
    },
    "openai": {
        "api_key": OPENAI_API_KEY,
        "base_url": None,
        "models": [
            "gpt-4.1-nano",
            "gpt-4.1-mini",
            "gpt-4.1",
            "gpt-4o-mini",
        ],
        "default_model": "gpt-4.1-nano",
    },
}

DEFAULT_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()

# Chunking (parent-child)
PARENT_CHUNK_SIZE = 2000
PARENT_CHUNK_OVERLAP = 200
CHILD_CHUNK_SIZE = 400
CHILD_CHUNK_OVERLAP = 50

# Retrieval
TOP_K_VECTOR = 20
TOP_K_BM25 = 20
TOP_K_FINAL = 6

# Multi-turn
MAX_HISTORY_TURNS = 5
