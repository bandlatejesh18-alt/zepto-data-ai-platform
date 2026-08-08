from app.embeddings import (
    embedding_model,
)

from app.chroma_store import (
    get_collection,
    search_embeddings,
)


def retrieve_documents(
    query,
    top_k=3,
):
    """
    Retrieve the most relevant
    document chunks.

    Args:
        query:
            User question.

        top_k:
            Number of chunks
            to retrieve.

    Returns:
        dict:
            Retrieved documents.
    """

    collection = get_collection()

    query_embedding = embedding_model.encode(
        query,
    )

    results = search_embeddings(
        collection,
        query_embedding,
        top_k,
    )
    
    return results


def format_context(
    results,
):
    """
    Convert retrieved documents
    into a single context string.

    Args:
        results:
            ChromaDB query results.

    Returns:
        str:
            Formatted context.
    """

    documents = results[
        "documents"
    ][0]

    context = "\n\n".join(
        documents,
    )

    return context