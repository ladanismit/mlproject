"""Persistent FAISS vector store service for DocuMind AI.

This module provides vector storage, persistence, indexing, and loading capabilities
for document chunks and metadata using LangChain's FAISS integration and embedding service.
"""

from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from app.core.config import settings
from app.core.logger import get_logger
from app.rag.embeddings import EmbeddingService

logger = get_logger(__name__)

DEFAULT_INDEX_NAME = "documind_index"


class VectorStoreService:
    """Manages persistent FAISS vector stores for indexing and semantic retrieval."""

    def __init__(
        self,
        embedding_service: EmbeddingService | None = None,
        persist_directory: str | Path | None = None,
        index_name: str = DEFAULT_INDEX_NAME,
    ) -> None:
        """Initialize the VectorStoreService.

        Args:
            embedding_service: Optional EmbeddingService instance. Defaults to a new instance.
            persist_directory: Filesystem path to persist FAISS index files. Defaults to settings.VECTOR_STORE_DIR.
            index_name: Name of the index files (without extension). Defaults to 'documind_index'.
        """
        self.embedding_service = embedding_service or EmbeddingService()
        self.persist_directory = Path(persist_directory or settings.VECTOR_STORE_DIR).resolve()
        self.index_name = index_name

        # Ensure persist directory exists
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        logger.info(
            "VectorStoreService initialized (persist_directory=%s, index_name=%s)",
            self.persist_directory,
            self.index_name,
        )

    def _index_files_exist(self) -> bool:
        """Check if both required FAISS index files (.faiss and .pkl) exist on disk.

        Returns:
            bool: True if both files exist, False otherwise.
        """
        faiss_file = self.persist_directory / f"{self.index_name}.faiss"
        pkl_file = self.persist_directory / f"{self.index_name}.pkl"
        return faiss_file.is_file() and pkl_file.is_file()

    def _validate_documents(self, documents: list[Document]) -> None:
        """Validate that the documents argument is a non-empty list of LangChain Documents.

        Args:
            documents: List of LangChain Document objects.

        Raises:
            ValueError: If input is not a list, is empty, or contains invalid elements.
        """
        if not isinstance(documents, list):
            raise ValueError("Input 'documents' must be a list of LangChain Document objects")

        if not documents:
            raise ValueError("Cannot create or update vector store with an empty list of documents")

        for idx, doc in enumerate(documents):
            if not isinstance(doc, Document):
                raise ValueError(
                    f"Item at index {idx} is not a valid langchain_core.documents.Document instance"
                )

    def exists(self) -> bool:
        """Check whether a persisted FAISS vector index is present on disk.

        Returns:
            bool: True if the index exists, False otherwise.
        """
        return self._index_files_exist()

    def create(self, documents: list[Document]) -> FAISS:
        """Create a new FAISS vector store from document chunks and save it to disk.

        Args:
            documents: List of LangChain Document objects with chunk metadata.

        Returns:
            FAISS: Initialized and persisted LangChain FAISS vector store.

        Raises:
            ValueError: If document validation fails.
            RuntimeError: If FAISS creation or persistence fails.
        """
        self._validate_documents(documents)
        doc_count = len(documents)

        logger.info(
            "Creating new FAISS vector index '%s' from %d document chunk(s)...",
            self.index_name,
            doc_count,
        )

        try:
            embeddings = self.embedding_service.get_embeddings()
            vector_store = FAISS.from_documents(documents, embeddings)

            vector_store.save_local(
                str(self.persist_directory),
                index_name=self.index_name,
            )

            logger.info(
                "FAISS vector store '%s' successfully created and saved to %s",
                self.index_name,
                self.persist_directory,
            )
            return vector_store
        except (ValueError, FileNotFoundError):
            raise
        except Exception as exc:
            logger.error("Failed to create FAISS vector store: %s", exc)
            raise RuntimeError(f"Failed to create FAISS vector store: {exc}") from exc

    def load(self) -> FAISS:
        """Load an existing persisted FAISS vector store from disk.

        Returns:
            FAISS: Loaded LangChain FAISS vector store.

        Raises:
            FileNotFoundError: If the index files do not exist on disk.
            RuntimeError: If loading the index fails.
        """
        if not self.exists():
            logger.error(
                "Vector store index '%s' not found in %s",
                self.index_name,
                self.persist_directory,
            )
            raise FileNotFoundError(
                f"FAISS index '{self.index_name}' not found in '{self.persist_directory}'. "
                f"Please create the vector store first using create() or rebuild()."
            )

        logger.info(
            "Loading persisted FAISS vector store '%s' from %s...",
            self.index_name,
            self.persist_directory,
        )

        try:
            embeddings = self.embedding_service.get_embeddings()

            # Note: allow_dangerous_deserialization=True is used safely here because
            # we are loading our own trusted, locally generated index.pkl file.
            vector_store = FAISS.load_local(
                folder_path=str(self.persist_directory),
                embeddings=embeddings,
                index_name=self.index_name,
                allow_dangerous_deserialization=True,
            )

            logger.info("FAISS vector store '%s' loaded successfully.", self.index_name)
            return vector_store
        except FileNotFoundError:
            raise
        except Exception as exc:
            logger.error("Failed to load FAISS vector store: %s", exc)
            raise RuntimeError(f"Failed to load FAISS vector store: {exc}") from exc

    def add_documents(self, documents: list[Document]) -> None:
        """Add newly processed document chunks to the existing persisted FAISS index.

        Args:
            documents: List of LangChain Document objects to add.

        Raises:
            FileNotFoundError: If the vector store does not already exist.
            ValueError: If the documents list is invalid or empty.
            RuntimeError: If adding documents or persisting the index fails.
        """
        self._validate_documents(documents)
        doc_count = len(documents)

        logger.info(
            "Adding %d new document chunk(s) to vector store '%s'...",
            doc_count,
            self.index_name,
        )

        try:
            vector_store = self.load()
            vector_store.add_documents(documents)
            vector_store.save_local(
                str(self.persist_directory),
                index_name=self.index_name,
            )

            logger.info(
                "Successfully added %d document chunk(s) to '%s' and saved index to disk.",
                doc_count,
                self.index_name,
            )
        except (ValueError, FileNotFoundError):
            raise
        except Exception as exc:
            logger.error("Failed to add documents to FAISS vector store: %s", exc)
            raise RuntimeError(f"Failed to add documents to FAISS vector store: {exc}") from exc

    def rebuild(self, documents: list[Document]) -> FAISS:
        """Completely replace the existing persisted FAISS index with a new set of documents.

        Args:
            documents: Complete list of LangChain Document chunks to index.

        Returns:
            FAISS: The newly created and persisted FAISS vector store.

        Raises:
            ValueError: If documents list is invalid or empty.
            RuntimeError: If rebuild fails.
        """
        logger.info(
            "Rebuilding FAISS vector store '%s' with %d document chunk(s)...",
            self.index_name,
            len(documents) if isinstance(documents, list) else 0,
        )
        return self.create(documents)

    def get_store(self) -> FAISS:
        """Retrieve the active FAISS vector store, loading it from disk if available.

        Returns:
            FAISS: Loaded FAISS vector store.

        Raises:
            FileNotFoundError: If the store does not exist.
        """
        return self.load()
