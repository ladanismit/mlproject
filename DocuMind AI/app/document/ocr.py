"""Optical Character Recognition (OCR) module for DocuMind AI.

This module extracts text from document images (PNG, JPG, JPEG) and scanned
pages using pytesseract and Pillow, updating ProcessedDocument schemas with
the extracted textual content and marking OCR execution in document metadata.
"""

import os
from pathlib import Path
from PIL import Image
import pytesseract

from app.core.logger import get_logger
from app.models.schemas import DocumentMetadata, PageContent, ProcessedDocument

logger = get_logger(__name__)

IMAGE_EXTENSIONS = {"png", "jpg", "jpeg"}


class OCRProcessor:
    """Processes document images and populates page text via Optical Character Recognition."""

    def __init__(
        self,
        lang: str = "eng",
        config: str = "--psm 6",
        tesseract_cmd: str | None = None,
    ) -> None:
        """Initialize the OCR processor.

        Args:
            lang: Tesseract language code (default 'eng').
            config: Tesseract configuration parameters (default '--psm 6').
            tesseract_cmd: Optional explicit path to the tesseract executable.
        """
        self.lang = lang
        self.config = config

        # Set custom tesseract executable path from parameter, env var, or system PATH
        explicit_cmd = tesseract_cmd or os.getenv("TESSERACT_CMD")
        if explicit_cmd:
            pytesseract.pytesseract.tesseract_cmd = explicit_cmd

    def process(self, document: ProcessedDocument) -> ProcessedDocument:
        """Perform OCR on a document if required and update its text content.

        Args:
            document: ProcessedDocument instance from the loader.

        Returns:
            ProcessedDocument: Updated document with extracted text and updated metadata.

        Raises:
            FileNotFoundError: If the underlying document file is missing.
            RuntimeError: If Tesseract is not installed, not found, or fails execution.
            ValueError: If the image cannot be read or processed.
        """
        file_type = document.metadata.file_type.lower()
        file_path = Path(document.metadata.file_path)

        if file_type not in IMAGE_EXTENSIONS:
            logger.info(
                "Skipping OCR for '%s' (type: %s). Native text is preserved.",
                document.metadata.filename,
                file_type,
            )
            return document

        if not file_path.is_file():
            logger.error("Document file not found for OCR: %s", file_path)
            raise FileNotFoundError(f"File not found for OCR: {file_path}")

        logger.info("Starting OCR for image document: %s", document.metadata.filename)

        extracted_text = self._ocr_image(file_path)
        char_count = len(extracted_text)
        logger.info(
            "OCR completed successfully for '%s'. Extracted ~%d characters.",
            document.metadata.filename,
            char_count,
        )

        # Update page content
        updated_pages = [
            PageContent(
                page_number=1,
                text=extracted_text,
                metadata={
                    **document.pages[0].metadata,
                    "ocr_engine": "tesseract",
                    "lang": self.lang,
                },
            )
        ]

        # Update metadata to reflect OCR execution
        updated_metadata = DocumentMetadata(
            document_id=document.metadata.document_id,
            filename=document.metadata.filename,
            file_path=document.metadata.file_path,
            file_type=document.metadata.file_type,
            source=document.metadata.source,
            total_pages=document.metadata.total_pages,
            ocr_used=True,
        )

        return ProcessedDocument(
            metadata=updated_metadata,
            pages=updated_pages,
            full_text=extracted_text,
        )

    def _ocr_image(self, image_path: Path) -> str:
        """Run Tesseract OCR on an image file.

        Args:
            image_path: Path to the image file.

        Returns:
            str: Raw text extracted from the image.

        Raises:
            RuntimeError: If Tesseract executable is missing or execution fails.
            ValueError: If the image cannot be opened by Pillow.
        """
        try:
            with Image.open(image_path) as img:
                # Convert to RGB if needed (e.g., RGBA or palette-based images)
                if img.mode not in ("L", "RGB"):
                    img = img.convert("RGB")

                text: str = pytesseract.image_to_string(
                    img,
                    lang=self.lang,
                    config=self.config,
                )
                return text.strip()
        except pytesseract.TesseractNotFoundError as exc:
            logger.error(
                "Tesseract executable not found. Please install Tesseract OCR "
                "or set TESSERACT_CMD environment variable."
            )
            raise RuntimeError(
                "Tesseract OCR is not installed or not found in system PATH. "
                "Install Tesseract OCR and configure TESSERACT_CMD."
            ) from exc
        except pytesseract.TesseractError as exc:
            logger.error("Tesseract execution failed on '%s': %s", image_path.name, exc)
            raise RuntimeError(f"OCR execution failed on '{image_path.name}': {exc}") from exc
        except Exception as exc:
            logger.error("Failed to open or process image '%s': %s", image_path.name, exc)
            raise ValueError(f"Failed to process image '{image_path.name}': {exc}") from exc
