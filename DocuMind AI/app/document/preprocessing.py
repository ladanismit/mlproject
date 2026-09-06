"""Document text preprocessing and normalization module for DocuMind AI.

This module provides conservative text cleaning and normalization for extracted
document text, stripping unwanted control characters and redundant whitespace while
strictly preserving punctuation, currency symbols, numbers, Unicode scripts, and formatting.
"""

import re
from app.core.logger import get_logger
from app.models.schemas import PageContent, ProcessedDocument

logger = get_logger(__name__)


class TextPreprocessor:
    """Conservatively cleans and normalizes text across document pages and full text."""

    def process(self, document: ProcessedDocument) -> ProcessedDocument:
        """Clean and normalize the text of each page in the ProcessedDocument.

        Args:
            document: ProcessedDocument with raw text from loader or OCR.

        Returns:
            ProcessedDocument: Updated document with normalized text content.

        Raises:
            ValueError: If the input document is not a valid ProcessedDocument.
            RuntimeError: If an unexpected error occurs during text processing.
        """
        if not isinstance(document, ProcessedDocument):
            raise ValueError("Input must be an instance of ProcessedDocument")

        logger.info(
            "Starting text preprocessing for document: %s (%d pages)",
            document.metadata.filename,
            len(document.pages),
        )

        try:
            initial_char_count = len(document.full_text)
            cleaned_pages: list[PageContent] = []

            for page in document.pages:
                cleaned_text = self._clean_text(page.text)
                cleaned_pages.append(
                    PageContent(
                        page_number=page.page_number,
                        text=cleaned_text,
                        metadata=page.metadata,
                    )
                )

            # Reconstruct consolidated full_text from processed pages
            cleaned_full_text = "\n\n".join(
                page.text for page in cleaned_pages
            )

            final_char_count = len(cleaned_full_text)
            logger.info(
                "Preprocessing completed for '%s'. Character count: %d -> %d.",
                document.metadata.filename,
                initial_char_count,
                final_char_count,
            )

            return ProcessedDocument(
                metadata=document.metadata,
                pages=cleaned_pages,
                full_text=cleaned_full_text,
            )
        except Exception as exc:
            logger.error(
                "Failed to preprocess text for document '%s': %s",
                document.metadata.filename,
                exc,
            )
            raise RuntimeError(
                f"Text preprocessing failed for '{document.metadata.filename}': {exc}"
            ) from exc

    def _clean_text(self, text: str) -> str:
        """Apply safe, conservative normalization rules to a block of text.

        Args:
            text: Raw input text string.

        Returns:
            str: Cleaned and normalized text string.
        """
        if not text:
            return ""

        # 1. Normalize line endings (\r\n and \r -> \n)
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")

        # 2. Remove null bytes and non-printable control characters (except \n and \t)
        normalized = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", normalized)

        # 3. Clean line-by-line: strip trailing whitespace and collapse excessive horizontal spaces
        cleaned_lines: list[str] = []
        for line in normalized.split("\n"):
            # Collapse runs of 2+ tabs/spaces into a single space while keeping line intact
            line_cleaned = line.rstrip()
            cleaned_lines.append(line_cleaned)

        normalized = "\n".join(cleaned_lines)

        # 4. Collapse 3+ consecutive newlines down to 2 newlines (preserves paragraph breaks)
        normalized = re.sub(r"\n{3,}", "\n\n", normalized)

        return normalized.strip()
