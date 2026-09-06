"""Document semantic retriever module for DocuMind AI.

This module performs semantic similarity search over the persistent FAISS vector store
while strictly preserving chunk metadata (e.g. document_id, filename, page_number)
to power downstream RAG generation and precise source citations.
"""

from langchain_core.documents import Document

from app.core.config import settings
from app.core.logger import get_logger
from app.rag.vector_store import VectorStoreService

logger = get_logger(__name__)


class DocumentRetriever:
    """Retrieves relevant document chunks from the FAISS vector store using semantic similarity."""

    def __init__(
        self,
        vector_store_service: VectorStoreService | None = None,
        default_top_k: int | None = None,
    ) -> None:
        """Initialize the DocumentRetriever.

        Args:
            vector_store_service: Optional VectorStoreService instance. Defaults to a new instance.
            default_top_k: Default number of documents to retrieve. Defaults to settings.TOP_K_RETRIEVAL.

        Raises:
            ValueError: If default_top_k is less than 1.
        """
        self.vector_store_service = vector_store_service or VectorStoreService()
        self.default_top_k = (
            default_top_k if default_top_k is not None else settings.TOP_K_RETRIEVAL
        )

        if self.default_top_k < 1:
            raise ValueError(f"default_top_k must be at least 1, got {self.default_top_k}")

        logger.info(
            "DocumentRetriever initialized (default_top_k=%d)",
            self.default_top_k,
        )

    def _validate_query(self, query: str) -> str:
        """Validate that the query is a non-empty string.

        Args:
            query: Input search string.

        Returns:
            str: Cleaned search query.

        Raises:
            ValueError: If query is None, not a string, or contains only whitespace.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Search query must be a non-empty string")
        return query.strip()

    def _resolve_top_k(self, top_k: int | None) -> int:
        """Resolve and validate the top_k parameter.

        Args:
            top_k: Optional integer specifying number of results.

        Returns:
            int: Validated top_k value.

        Raises:
            ValueError: If top_k is less than 1.
        """
        if top_k is None:
            return self.default_top_k

        if not isinstance(top_k, int) or top_k < 1:
            raise ValueError(f"top_k must be an integer >= 1, got {top_k}")

        return top_k

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        document_ids: list[str] | None = None,
    ) -> list[Document]:
        """Retrieve the most semantically relevant document chunks for a query.

        Args:
            query: Natural language query string.
            top_k: Optional number of chunks to return (overrides default_top_k).
            document_ids: Optional list of document IDs to filter the retrieval scope.

        Returns:
            list[Document]: Ranked list of relevant LangChain Document chunks.

        Raises:
            ValueError: If query or top_k parameters are invalid.
            FileNotFoundError: If the underlying FAISS index does not exist.
            RuntimeError: If similarity search fails.
        """
        scored_results = self.retrieve_with_scores(
            query=query,
            top_k=top_k,
            document_ids=document_ids,
        )
        return [doc for doc, _ in scored_results]

    def retrieve_with_scores(
        self,
        query: str,
        top_k: int | None = None,
        document_ids: list[str] | None = None,
    ) -> list[tuple[Document, float]]:
        """Retrieve relevant document chunks alongside their FAISS similarity distance scores.

        Note:
            In standard FAISS index configurations (e.g. L2 distance), lower score values
            indicate closer semantic proximity / higher similarity.

        Args:
            query: Natural language query string.
            top_k: Optional number of chunks to return.
            document_ids: Optional list of document IDs to filter candidates.

        Returns:
            list[tuple[Document, float]]: Pairs of (Document, retrieval_score).

        Raises:
            ValueError: If query or top_k parameters are invalid.
            FileNotFoundError: If the underlying FAISS index does not exist.
            RuntimeError: If retrieval execution fails.
        """
        cleaned_query = self._validate_query(query)
        resolved_k = self._resolve_top_k(top_k)
        has_doc_filter = document_ids is not None

        logger.info(
            "Executing similarity retrieval (top_k=%d, document_filter=%s)",
            resolved_k,
            has_doc_filter,
        )

        try:
            vector_store = self.vector_store_service.get_store()

            if not has_doc_filter:
                results = vector_store.similarity_search_with_score(
                    cleaned_query,
                    k=resolved_k,
                )
                logger.info("Retrieved %d chunk(s) from vector store.", len(results))
                return results

            # Handle empty document filter list edge case
            if len(document_ids) == 0:
                logger.info("Empty document_ids filter supplied. Returning 0 chunks.")
                return []

            target_doc_ids = set(document_ids)
            candidate_k = max(resolved_k * 3, resolved_k)

            candidates = vector_store.similarity_search_with_score(
                cleaned_query,
                k=candidate_k,
            )

            # Filter candidates by document_id while preserving distance scores
            filtered_results = [
                (doc, score)
                for doc, score in candidates
                if doc.metadata.get("document_id") in target_doc_ids
            ]

            final_results = filtered_results[:resolved_k]
            logger.info(
                "Retrieved %d chunk(s) matching document filter from %d candidates.",
                len(final_results),
                len(candidates),
            )
            return final_results

        except (ValueError, FileNotFoundError):
            raise
        except Exception as exc:
            logger.error("Similarity retrieval failed: %s", exc)
            raise RuntimeError(f"Semantic retrieval failed: {exc}") from exc
