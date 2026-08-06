from pathlib import Path
from sentence_transformers import SentenceTransformer

from config import CHUNK_SIZE
from retrieval import collection


# Load the embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


# Path to the documents folder
DOCUMENTS_PATH = Path(__file__).parent.parent / "data" / "documents"


def load_documents():
    """
    Load all text documents from the documents directory.
    Returns:
        list[dict]: A list of dictionaries containing
                    document id and text.
    """

    documents = []

    # Read all .txt files
    for file_path in sorted(DOCUMENTS_PATH.glob("*.txt")):

        with open(file_path, "r", encoding="utf-8") as file:

            text = file.read().strip()

        documents.append(
            {
                "id": file_path.stem,
                "text": text,
            }
        )

    return documents


def chunk_documents(documents):
    """
    Split documents into fixed-size chunks.

    Each document is divided into chunks of CHUNK_SIZE characters.
    Every chunk is assigned a unique chunk ID while maintaining
    a reference to its original document.

    Args:
        documents (list):
            List of dictionaries containing document IDs and text.

    Returns:
        list:
            A list of chunk dictionaries with:
                - chunk_id
                - document_id
                - text
    """
    chunks = []

    for document in documents:
        text = document["text"]

        for index, start in enumerate(range(0, len(text), CHUNK_SIZE), start=1):
            chunk_text = text[start:start + CHUNK_SIZE]

            chunks.append(
                {
                    "chunk_id": f"{document['id']}_chunk_{index}",
                    "document_id": document["id"],
                    "text": chunk_text,
                }
            )

    return chunks


def generate_embeddings(chunks):
    """
    Generate embeddings for every chunk.

    Args:
        chunks (list):
            List of chunk dictionaries.

    Returns:
        list:
            The same chunk dictionaries with an additional
            'embedding' field containing the embedding vector.
    """

    for chunk in chunks:
        embedding = embedding_model.encode(chunk["text"])

        chunk["embedding"] = embedding

    return chunks


def store_embeddings(chunks):
    """
    Store chunk embeddings inside ChromaDB.

    Args:
        chunks (list):
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
        ]
    )

    print(f"Stored {len(chunks)} chunks in ChromaDB.")


if __name__ == "__main__":

    print("=" * 70)
    print("Zepto Support Assistant - Embedding Pipeline")
    print("=" * 70)

    # Step 1: Load documents
    documents = load_documents()
    print(f"\n✅ Loaded {len(documents)} documents.")

    # Step 2: Chunk documents
    chunks = chunk_documents(documents)
    print(f"✅ Created {len(chunks)} chunks.")

    # Step 3: Generate embeddings
    chunks = generate_embeddings(chunks)
    print(f"✅ Generated embeddings for {len(chunks)} chunks.")

    # Preview
    first_chunk = chunks[0]

    print("\n" + "=" * 70)
    print("Sample Chunk")
    print("=" * 70)

    print(f"Chunk ID      : {first_chunk['chunk_id']}")
    print(f"Document ID   : {first_chunk['document_id']}")
    print(f"Text Length   : {len(first_chunk['text'])} characters")
    print(f"Embedding Dim : {len(first_chunk['embedding'])}")

    print("\nFirst 10 embedding values:")
    print(first_chunk["embedding"][:10])

    print("\n✅ Embedding pipeline completed successfully.")