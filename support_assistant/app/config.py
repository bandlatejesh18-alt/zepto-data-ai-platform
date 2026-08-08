import os

from dotenv import load_dotenv

load_dotenv()

MOCK_LLM = os.getenv(
    "MOCK_LLM",
    "1",
)

CHUNK_SIZE = 500

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

COLLECTION_NAME = "zepto_policies"

CHROMA_DB_PATH = "data/chroma_db"

DOCUMENTS_PATH = "data/documents"