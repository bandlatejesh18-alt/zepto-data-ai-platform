import chromadb

from app.config import (
    CHROMA_DB_PATH,
    COLLECTION_NAME,
)


def get_collection():
    """
    Create or load the ChromaDB collection.

    Returns:
        chromadb.Collection:
            Chroma collection.
    """

    client = chromadb.PersistentClient(
        path=CHROMA_DB_PATH,
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
    )
    
    return collection


def store_embeddings(
    collection,
    chunks,
):
    """
    Store chunk embeddings
    inside ChromaDB.

    Args:
        collection:
            ChromaDB collection.

        chunks:
            List of chunk dictionaries.

    Returns:
        None
    """

    collection.add(
        ids=[
            chunk["chunk_id"]
            for chunk in chunks
        ],
        embeddings=[
            chunk["embedding"].tolist()
            for chunk in chunks
        ],
        documents=[
            chunk["text"]
            for chunk in chunks
        ],
        metadatas=[
            {
                "document_id": chunk["document_id"]
            }
            for chunk in chunks
        ],
    )

    print(
        f"Stored {len(chunks)} chunks in ChromaDB."
    )


def search_embeddings(
    collection,
    query_embedding,
    top_k=3,
):
    """
    Search for the most similar
    document chunks.

    Args:
        collection:
            ChromaDB collection.

        query_embedding:
            Embedding vector of
            the user query.

        top_k:
            Number of chunks
            to retrieve.

    Returns:
        dict:
            ChromaDB query results.
    """

    results = collection.query(
        query_embeddings=[
            query_embedding.tolist()
        ],
        n_results=top_k,
    )

    return results