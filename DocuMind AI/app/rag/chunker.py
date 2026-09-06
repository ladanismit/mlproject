"""Document chunking and text splitting module for DocuMind AI.

This module converts processed document pages into metadata-rich LangChain Document
chunks for downstream embedding, vector storage, semantic retrieval, and page-level
source citations.
"""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import settings
from app.core.logger import get_logger
from app.models.schemas import ProcessedDocument

logger = get_logger(__name__)


class DocumentChunker:
    """Splits processed document pages into semantically bounded, citation-ready LangChain Document chunks."""

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
        """Initialize the document chunker.

        Args:
            chunk_size: Maximum number of characters per chunk (defaults to settings.CHUNK_SIZE).
            chunk_overlap: Number of overlapping characters between adjacent chunks (defaults to settings.CHUNK_OVERLAP).

        Raises:
            ValueError: If chunk_size <= 0 or chunk_overlap is negative or >= chunk_size.
        """
        self.chunk_size = chunk_size if chunk_size is not None else settings.CHUNK_SIZE
        self.chunk_overlap = (
            chunk_overlap if chunk_overlap is not None else settings.CHUNK_OVERLAP
        )

        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be greater than 0, got {self.chunk_size}")

        if not (0 <= self.chunk_overlap < self.chunk_size):
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must satisfy 0 <= chunk_overlap < chunk_size ({self.chunk_size})"
            )

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )

    def chunk(self, document: ProcessedDocument) -> list[Document]:
        """Split a ProcessedDocument page by page into LangChain Document chunks with source metadata.

        Args:
            document: ProcessedDocument containing metadata and page contents.

        Returns:
            list[Document]: List of LangChain Document chunks enriched with page-level citation metadata.

        Raises:
            ValueError: If the input is not a valid ProcessedDocument.
            RuntimeError: If an unexpected error occurs during splitting.
        """
        if not isinstance(document, ProcessedDocument):
            raise ValueError("Input must be an instance of ProcessedDocument")

        logger.info(
            "Starting chunking for document: '%s' (chunk_size=%d, chunk_overlap=%d)",
            document.metadata.filename,
            self.chunk_size,
            self.chunk_overlap,
        )

        try:
            chunks: list[Document] = []
            pages_processed = 0
            pages_skipped = 0
            chunk_index = 0

            for page in document.pages:
                page_text = page.text.strip() if page.text else ""

                if not page_text:
                    pages_skipped += 1
                    continue

                pages_processed += 1
                page_splits = self.splitter.split_text(page_text)

                for split_text in page_splits:
                    clean_split = split_text.strip()
                    if not clean_split:
                        continue

                    chunk_metadata = {
                        "document_id": document.metadata.document_id,
                        "filename": document.metadata.filename,
                        "file_type": document.metadata.file_type,
                        "source": document.metadata.source,
                        "page_number": page.page_number,
                        "chunk_index": chunk_index,
                        "ocr_used": document.metadata.ocr_used,
                    }

                    chunks.append(
                        Document(
                            page_content=clean_split,
                            metadata=chunk_metadata,
                        )
                    )
                    chunk_index += 1

            logger.info(
                "Chunking completed for '%s': %d total chunks generated from %d page(s) (%d page(s) skipped).",
                document.metadata.filename,
                len(chunks),
                pages_processed,
                pages_skipped,
            )

            return chunks

        except Exception as exc:
            logger.error(
                "Failed to chunk document '%s': %s",
                document.metadata.filename,
                exc,
            )
            raise RuntimeError(
                f"Chunking failed for document '{document.metadata.filename}': {exc}"
            ) from exc
