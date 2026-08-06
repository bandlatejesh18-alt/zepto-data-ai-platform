from chromadb import PersistentClient
from config import CHROMA_DB_PATH, COLLECTION_NAME

client = PersistentClient(path=CHROMA_DB_PATH)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME
)