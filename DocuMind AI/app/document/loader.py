"""Document loader module for DocuMind AI.

This module is responsible for loading supported file formats (PDF, PNG, JPG, JPEG)
from disk and converting them into structured ProcessedDocument instances containing
DocumentMetadata and page-by-page PageContent objects.
"""

from IPython import paths
from pathlib import Path
import uuid
from PIL import Image
import fitz  # PyMuPDF

from app.core.logger import get_logger
from app.models.schemas import DocumentMetadata, PageContent, ProcessedDocument

logger = get_logger(__name__)

SUPPORTED_PDF_EXTENSIONS = {".pdf"}
SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
SUPPORTED_EXTENSIONS = SUPPORTED_PDF_EXTENSIONS | SUPPORTED_IMAGE_EXTENSIONS


class DocumentLoader:
    """Loads documents from disk into standardized ProcessedDocument representations."""

    def load(self, file_path: str | Path) -> ProcessedDocument:
        """Load a document from disk and extract its initial pages and metadata.

        Args:
            file_path: Path to the target document (PDF or image).

        Returns:
            ProcessedDocument: Standardized document containing metadata and page contents.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file extension is unsupported or file is empty/corrupt.
            RuntimeError: If an unexpected error occurs during loading.
        """
        path = Path(file_path).resolve()

        if not path.is_file():
            logger.error("Document not found: %s", path)
            raise FileNotFoundError(f"File not found: {path}")

        ext = path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            logger.error("Unsupported file extension: %s (path: %s)", ext, path)
            raise ValueError(
                f"Unsupported file format '{ext}'. Supported formats: {sorted(SUPPORTED_EXTENSIONS)}"
            )

        logger.info("Loading document: %s", path.name)

        try:
            if ext in SUPPORTED_PDF_EXTENSIONS:
                return self._load_pdf(path)
            elif ext in SUPPORTED_IMAGE_EXTENSIONS:
                return self._load_image(path)
            else:
                raise ValueError(f"Unhandled file extension: {ext}")
        except (FileNotFoundError, ValueError):
            raise
        except Exception as exc:
            logger.error("Failed to load document %s: %s", path.name, exc)
            raise RuntimeError(f"Failed to load document '{path.name}': {exc}") from exc

    def _load_pdf(self, path: Path) -> ProcessedDocument:
        """Extract pages and native digital text from a PDF file.

        Args:
            path: Resolved Path to the PDF file.

        Returns:
            ProcessedDocument: Structured document with 1-based page contents.
        """
        logger.info("Detected document type: PDF (%s)", path.name)
        pages: list[PageContent] = []

        try:
            with fitz.open(path) as doc:
                total_pages = len(doc)
                logger.info("PDF '%s' contains %d page(s)", path.name, total_pages)

                for index, page in enumerate(doc):
                    page_number = index + 1  # 1-based indexing
                    text = page.get_text() or ""
                    pages.append(
                        PageContent(
                            page_number=page_number,
                            text=text,
                            metadata={"page_index": index},
                        )
                    )
        except Exception as exc:
            raise ValueError(f"Corrupted or unreadable PDF '{path.name}': {exc}") from exc

        full_text = "\n\n".join(page.text for page in pages)

        metadata = DocumentMetadata(
            document_id=str(uuid.uuid4()),
            filename=path.name,
            file_path=str(path),
            file_type="pdf",
            source=None,
            total_pages=len(pages),
            ocr_used=False,
        )

        logger.info("Document loaded successfully: %s (%d pages)", path.name, len(pages))
        return ProcessedDocument(metadata=metadata, pages=pages, full_text=full_text)

    def _load_image(self, path: Path) -> ProcessedDocument:
        """Validate an image file and initialize a single-page document placeholder.

        Args:
            path: Resolved Path to the image file.

        Returns:
            ProcessedDocument: Structured document ready for subsequent OCR processing.
        """
        file_type = path.suffix.lower().lstrip(".")
        logger.info("Detected document type: Image (%s, format: %s)", path.name, file_type)

        try:
            with Image.open(path) as img:
                img.verify()  # Validate that image is not corrupted
        except Exception as exc:
            raise ValueError(f"Corrupted or unreadable image '{path.name}': {exc}") from exc

        # Images start with empty text; text extraction/OCR is performed downstream
        pages = [
            PageContent(
                page_number=1,
                text="",
                metadata={"format": file_type},
            )
        ]

        metadata = DocumentMetadata(
            document_id=str(uuid.uuid4()),
            filename=path.name,
            file_path=str(path),
            file_type=file_type,
            source=None,
            total_pages=1,
            ocr_used=False,
        )

        logger.info("Document loaded successfully: %s (1 page)", path.name)
        return ProcessedDocument(metadata=metadata, pages=pages, full_text="")
