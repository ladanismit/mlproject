"""Vector embedding service module for DocuMind AI.

This module provides the embedding abstraction used to convert document chunks
and user search queries into vector representations for vector storage,
FAISS indexing, and semantic retrieval.
"""

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)

SUPPORTED_PROVIDERS = {"openai"}


class EmbeddingService:
    """Manages embedding generation for document chunks and user queries."""

    def __init__(
        self,
        provider: str | None = None,
        model_name: str | None = None,
    ) -> None:
        """Initialize the embedding service with the configured provider and model.

        Args:
            provider: Embedding provider name (defaults to settings.EMBEDDING_PROVIDER).
            model_name: Name of the embedding model (defaults to settings.EMBEDDING_MODEL).

        Raises:
            ValueError: If the provider is unsupported or required API keys are missing.
        """
        self.provider = (provider or settings.EMBEDDING_PROVIDER).lower()
        self.model_name = model_name or settings.EMBEDDING_MODEL

        if self.provider not in SUPPORTED_PROVIDERS:
            raise ValueError(
                f"Unsupported embedding provider '{self.provider}'. "
                f"Supported providers: {sorted(SUPPORTED_PROVIDERS)}"
            )

        self._embeddings: Embeddings = self._initialize_embeddings()

        logger.info(
            "EmbeddingService initialized successfully (provider=%s, model=%s)",
            self.provider,
            self.model_name,
        )

    def _initialize_embeddings(self) -> Embeddings:
        """Instantiate and configure the underlying provider embedding model.

        Returns:
            Embeddings: Initialized LangChain embeddings instance.

        Raises:
            ValueError: If required API credentials are not set.
        """
        if self.provider == "openai":
            api_key = settings.OPENAI_API_KEY
            if not api_key or not api_key.strip():
                raise ValueError(
                    "OPENAI_API_KEY is not configured. Please set OPENAI_API_KEY in your "
                    "environment or .env file to initialize the OpenAI embedding service."
                )

            return OpenAIEmbeddings(
                model=self.model_name,
                api_key=api_key,
            )

        raise ValueError(f"Unhandled provider: {self.provider}")

    def get_embeddings(self) -> Embeddings:
        """Return the initialized LangChain Embeddings instance.

        Returns:
            Embeddings: The active embedding model instance for vector store integration.
        """
        return self._embeddings

    def embed_documents(self, documents: list[Document]) -> list[list[float]]:
        """Generate embedding vectors for a collection of LangChain Document chunks.

        Args:
            documents: List of LangChain Document instances.

        Returns:
            list[list[float]]: List of embedding vectors corresponding to each document chunk.

        Raises:
            ValueError: If documents argument is not a list.
            RuntimeError: If embedding generation fails.
        """
        if not isinstance(documents, list):
            raise ValueError("Input 'documents' must be a list of LangChain Document objects")

        if not documents:
            logger.info("No documents provided for embedding. Returning empty vector list.")
            return []

        doc_count = len(documents)
        logger.info("Generating embeddings for %d document chunk(s)...", doc_count)

        try:
            texts = [doc.page_content for doc in documents]
            vectors = self._embeddings.embed_documents(texts)
            logger.info("Successfully generated embeddings for %d document chunk(s).", doc_count)
            return vectors
        except Exception as exc:
            logger.error("Failed to generate document embeddings: %s", exc)
            raise RuntimeError(f"Embedding generation failed for document chunks: {exc}") from exc

    def embed_query(self, query: str) -> list[float]:
        """Generate an embedding vector for a user query string.

        Args:
            query: The query text to embed.

        Returns:
            list[float]: Embedding vector for semantic retrieval.

        Raises:
            ValueError: If query is not a non-empty string.
            RuntimeError: If query embedding generation fails.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string")

        try:
            vector = self._embeddings.embed_query(query.strip())
            return vector
        except Exception as exc:
            logger.error("Failed to generate query embedding: %s", exc)
            raise RuntimeError(f"Query embedding generation failed: {exc}") from exc
