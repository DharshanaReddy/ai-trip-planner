"""
ChromaDB vector store for travel knowledge retrieval.
Uses OpenAI embeddings for semantic search over the knowledge base.
"""

import chromadb
from chromadb.utils import embedding_functions
from langsmith import traceable
import os
import logging
from .knowledge_base import TRAVEL_DOCUMENTS

logger = logging.getLogger(__name__)

_client: chromadb.Client | None = None
_collection: chromadb.Collection | None = None
COLLECTION_NAME = "travel_knowledge"


def _get_embedding_fn():
    return embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name="text-embedding-3-small",
    )


def get_collection() -> chromadb.Collection:
    global _client, _collection
    if _collection is not None:
        return _collection

    _client = chromadb.Client()
    ef = _get_embedding_fn()

    try:
        _collection = _client.get_collection(name=COLLECTION_NAME, embedding_function=ef)
        logger.info("Loaded existing ChromaDB collection")
    except Exception:
        _collection = _client.create_collection(name=COLLECTION_NAME, embedding_function=ef)
        _seed_knowledge_base(_collection)
        logger.info("Created and seeded ChromaDB collection")

    return _collection


def _seed_knowledge_base(collection: chromadb.Collection):
    collection.add(
        ids=[doc["id"] for doc in TRAVEL_DOCUMENTS],
        documents=[doc["content"] for doc in TRAVEL_DOCUMENTS],
        metadatas=[doc["metadata"] for doc in TRAVEL_DOCUMENTS],
    )
    logger.info(f"Seeded {len(TRAVEL_DOCUMENTS)} documents into ChromaDB")


@traceable(name="rag-retrieval")
def retrieve_context(query: str, n_results: int = 3) -> str:
    """Retrieve relevant travel knowledge for a given query."""
    try:
        collection = get_collection()
        results = collection.query(query_texts=[query], n_results=n_results)
        docs = results.get("documents", [[]])[0]
        return "\n\n---\n\n".join(docs) if docs else ""
    except Exception as e:
        logger.warning(f"RAG retrieval failed: {e}")
        return ""
