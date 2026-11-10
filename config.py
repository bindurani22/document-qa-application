from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DIR_PATH = BASE_DIR / "data"
CHROMA_PATH = "chroma_db"

EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "llama2"

CHUNK_SIZE = 1000
CHUNK_OVERLAP =200
TOP_K = 3