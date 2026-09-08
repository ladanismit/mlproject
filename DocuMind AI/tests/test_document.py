"""Unit and integration tests for the DocuMind AI document ingestion pipeline.

This module tests DocumentLoader, OCRProcessor, TextPreprocessor, and the end-to-end
ingestion pipeline across PDF and image files, verifying text extraction, OCR integration,
conservative normalization, and page-level metadata preservation.
"""

from pathlib import Path
from unittest.mock import MagicMock
import fitz  # PyMuPDF
from PIL import Image
import pytest
import pytesseract

from app.document.loader import (
    SUPPORTED_EXTENSIONS,
    SUPPORTED_IMAGE_EXTENSIONS,
    SUPPORTED_PDF_EXTENSIONS,
    DocumentLoader,
)
from app.document.ocr import OCRProcessor
from app.document.preprocessing import TextPreprocessor
from app.models.schemas import DocumentMetadata, PageContent, ProcessedDocument


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture
def sample_pdf_path(tmp_path: Path) -> Path:
    """Fixture creating a two-page PDF file using PyMuPDF."""
    pdf_path = tmp_path / "test_contract.pdf"
    doc = fitz.open()

    page1 = doc.new_page()
    page1.insert_text((50, 72), "Contract Agreement\nParties: Alpha Corp & Beta LLC")

    page2 = doc.new_page()
    page2.insert_text((50, 72), "Payment Terms\nTotal Consideration: $50,000.00")

    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


@pytest.fixture
def sample_png_path(tmp_path: Path) -> Path:
    """Fixture creating a sample PNG image file using Pillow."""
    image_path = tmp_path / "test_invoice.png"
    img = Image.new("RGB", (300, 150), color=(255, 255, 255))
    img.save(image_path)
    return image_path


@pytest.fixture
def sample_jpg_path(tmp_path: Path) -> Path:
    """Fixture creating a sample JPEG image file using Pillow."""
    image_path = tmp_path / "test_receipt.jpg"
    img = Image.new("RGB", (300, 150), color=(240, 240, 240))
    img.save(image_path, format="JPEG")
    return image_path


# =====================================================================
# 1. DocumentLoader Tests
# =====================================================================

def test_document_loader_constants() -> None:
    """Verify DocumentLoader exposes expected supported extension constants."""
    assert ".pdf" in SUPPORTED_PDF_EXTENSIONS
    assert ".png" in SUPPORTED_IMAGE_EXTENSIONS
    assert ".jpg" in SUPPORTED_IMAGE_EXTENSIONS
    assert ".jpeg" in SUPPORTED_IMAGE_EXTENSIONS
    assert SUPPORTED_EXTENSIONS == (SUPPORTED_PDF_EXTENSIONS | SUPPORTED_IMAGE_EXTENSIONS)


def test_load_pdf_successful(sample_pdf_path: Path) -> None:
    """Verify loading a multi-page PDF extracts pages, text, and generates metadata."""
    loader = DocumentLoader()
    doc = loader.load(sample_pdf_path)

    assert isinstance(doc, ProcessedDocument)
    assert doc.metadata.filename == "test_contract.pdf"
    assert doc.metadata.file_type == "pdf"
    assert doc.metadata.total_pages == 2
    assert doc.metadata.ocr_used is False
    assert len(doc.metadata.document_id) > 0

    assert len(doc.pages) == 2
    assert doc.pages[0].page_number == 1
    assert "Contract Agreement" in doc.pages[0].text
    assert doc.pages[1].page_number == 2
    assert "Total Consideration: $50,000.00" in doc.pages[1].text

    assert "Contract Agreement" in doc.full_text
    assert "Payment Terms" in doc.full_text


def test_load_image_successful(sample_png_path: Path, sample_jpg_path: Path) -> None:
    """Verify loading images initializes a single-page placeholder ready for downstream OCR."""
    loader = DocumentLoader()

    # PNG test
    png_doc = loader.load(sample_png_path)
    assert png_doc.metadata.filename == "test_invoice.png"
    assert png_doc.metadata.file_type == "png"
    assert png_doc.metadata.total_pages == 1
    assert len(png_doc.pages) == 1
    assert png_doc.pages[0].page_number == 1
    assert png_doc.pages[0].text == ""  # Text is empty prior to OCR

    # JPG test
    jpg_doc = loader.load(sample_jpg_path)
    assert jpg_doc.metadata.filename == "test_receipt.jpg"
    assert jpg_doc.metadata.file_type == "jpg"
    assert jpg_doc.metadata.total_pages == 1


def test_load_file_not_found(tmp_path: Path) -> None:
    """Verify loader raises FileNotFoundError when target file does not exist."""
    loader = DocumentLoader()
    missing_path = tmp_path / "non_existent.pdf"
    with pytest.raises(FileNotFoundError, match="File not found"):
        loader.load(missing_path)


def test_load_unsupported_extension(tmp_path: Path) -> None:
    """Verify loader raises ValueError when given an unsupported file format."""
    loader = DocumentLoader()
    unsupported_file = tmp_path / "test.docx"
    unsupported_file.write_text("dummy docx content")

    with pytest.raises(ValueError, match="Unsupported file format '.docx'"):
        loader.load(unsupported_file)


def test_load_corrupted_pdf(tmp_path: Path) -> None:
    """Verify loader raises ValueError when attempting to open a corrupted PDF file."""
    loader = DocumentLoader()
    corrupt_pdf = tmp_path / "corrupted.pdf"
    corrupt_pdf.write_bytes(b"%PDF-1.4 completely corrupted binary data")

    with pytest.raises(ValueError, match="Corrupted or unreadable PDF"):
        loader.load(corrupt_pdf)


# =====================================================================
# 2. OCRProcessor Tests
# =====================================================================

def test_ocr_processor_initialization() -> None:
    """Verify OCRProcessor initializes with default and custom configurations."""
    ocr_default = OCRProcessor()
    assert ocr_default.lang == "eng"
    assert ocr_default.config == "--psm 6"

    ocr_custom = OCRProcessor(lang="fra", config="--psm 3", tesseract_cmd="/usr/bin/tesseract")
    assert ocr_custom.lang == "fra"
    assert ocr_custom.config == "--psm 3"


def test_ocr_process_image_success(
    sample_png_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify OCRProcessor extracts text from image documents and updates metadata."""
    loader = DocumentLoader()
    raw_doc = loader.load(sample_png_path)

    # Mock pytesseract.image_to_string
    mock_extracted_text = "Invoice Number: INV-001\nTotal: ₹50,000"
    monkeypatch.setattr(
        "app.document.ocr.pytesseract.image_to_string",
        lambda img, lang, config: mock_extracted_text,
    )

    ocr = OCRProcessor()
    processed_doc = ocr.process(raw_doc)

    assert processed_doc.metadata.ocr_used is True
    assert len(processed_doc.pages) == 1
    assert processed_doc.pages[0].text == mock_extracted_text
    assert processed_doc.pages[0].metadata.get("ocr_engine") == "tesseract"
    assert processed_doc.full_text == mock_extracted_text


def test_ocr_skips_native_pdf(sample_pdf_path: Path) -> None:
    """Verify OCRProcessor skips OCR for PDF documents with native text."""
    loader = DocumentLoader()
    pdf_doc = loader.load(sample_pdf_path)

    ocr = OCRProcessor()
    result_doc = ocr.process(pdf_doc)

    # Preserves native document without modifying OCR flag
    assert result_doc.metadata.ocr_used is False
    assert "Contract Agreement" in result_doc.full_text


def test_ocr_missing_file_error(sample_png_path: Path) -> None:
    """Verify OCRProcessor raises FileNotFoundError if underlying image file was deleted."""
    loader = DocumentLoader()
    raw_doc = loader.load(sample_png_path)
    sample_png_path.unlink()  # Remove underlying file

    ocr = OCRProcessor()
    with pytest.raises(FileNotFoundError, match="File not found for OCR"):
        ocr.process(raw_doc)


def test_ocr_tesseract_error_handling(
    sample_png_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify OCRProcessor translates TesseractError into RuntimeError."""
    loader = DocumentLoader()
    raw_doc = loader.load(sample_png_path)

    def mock_tesseract_failure(*args: object, **kwargs: object) -> str:
        raise pytesseract.TesseractError(status=1, message="Tesseract engine crash")

    monkeypatch.setattr(
        "app.document.ocr.pytesseract.image_to_string",
        mock_tesseract_failure,
    )

    ocr = OCRProcessor()
    with pytest.raises(RuntimeError, match="OCR execution failed on 'test_invoice.png'"):
        ocr.process(raw_doc)


# =====================================================================
# 3. TextPreprocessor Tests
# =====================================================================

def test_text_preprocessor_basic_cleaning() -> None:
    """Verify TextPreprocessor removes redundant whitespace, normalizes line endings, and collapses empty lines."""
    raw_text = (
        "Invoice   Number:   INV-001\r\n\r\n\r\n\r\n"
        "Total Amount:   ₹50,000  \t  \n\n\n"
        "Status: Paid \x00\x07"
    )

    doc = ProcessedDocument(
        metadata=DocumentMetadata(
            document_id="doc-clean-1",
            filename="clean.pdf",
            file_path="clean.pdf",
            file_type="pdf",
            total_pages=1,
            ocr_used=False,
        ),
        pages=[PageContent(page_number=1, text=raw_text)],
        full_text=raw_text,
    )

    preprocessor = TextPreprocessor()
    cleaned_doc = preprocessor.process(doc)

    page_text = cleaned_doc.pages[0].text
    # Preserves content while cleaning redundant empty lines down to 2
    assert "Invoice   Number:   INV-001" in page_text
    assert "Total Amount:   ₹50,000" in page_text
    assert "\x00" not in page_text  # Null byte stripped
    assert "\r" not in page_text    # Normalized line breaks
    assert "\n\n\n" not in page_text  # 3+ newlines collapsed to 2


def test_text_preprocessor_page_preservation() -> None:
    """Verify TextPreprocessor preserves page numbers, page boundaries, and page metadata."""
    pages = [
        PageContent(page_number=1, text="Page 1 header and content.", metadata={"custom_key": "val1"}),
        PageContent(page_number=2, text="Page 2 financial details.", metadata={"custom_key": "val2"}),
    ]
    doc = ProcessedDocument(
        metadata=DocumentMetadata(
            document_id="doc-pages-1",
            filename="multipage.pdf",
            file_path="multipage.pdf",
            file_type="pdf",
            total_pages=2,
            ocr_used=False,
        ),
        pages=pages,
        full_text="Page 1 header and content.\n\nPage 2 financial details.",
    )

    preprocessor = TextPreprocessor()
    processed_doc = preprocessor.process(doc)

    assert len(processed_doc.pages) == 2
    assert processed_doc.pages[0].page_number == 1
    assert processed_doc.pages[0].metadata["custom_key"] == "val1"
    assert processed_doc.pages[1].page_number == 2
    assert processed_doc.pages[1].metadata["custom_key"] == "val2"

    # Verify full_text joins pages with double newlines
    assert processed_doc.full_text == "Page 1 header and content.\n\nPage 2 financial details."


def test_text_preprocessor_empty_pages() -> None:
    """Verify TextPreprocessor handles empty or whitespace-only pages without errors."""
    pages = [
        PageContent(page_number=1, text=""),
        PageContent(page_number=2, text="   \n \t "),
    ]
    doc = ProcessedDocument(
        metadata=DocumentMetadata(
            document_id="doc-empty-p",
            filename="empty_pages.pdf",
            file_path="empty_pages.pdf",
            file_type="pdf",
            total_pages=2,
            ocr_used=False,
        ),
        pages=pages,
        full_text="",
    )

    preprocessor = TextPreprocessor()
    processed_doc = preprocessor.process(doc)

    assert len(processed_doc.pages) == 2
    assert processed_doc.pages[0].text == ""
    assert processed_doc.pages[1].text == ""
    assert processed_doc.full_text == ""


def test_text_preprocessor_special_characters_and_unicode() -> None:
    """Verify TextPreprocessor strictly preserves currency symbols, punctuation, and Unicode scripts."""
    special_text = "₹50,000 | €4,200 | $1,500.50 | 18% GST | Ref: #123/ABC (Legal-Agreement) | ગુજરાતી"
    doc = ProcessedDocument(
        metadata=DocumentMetadata(
            document_id="doc-unicode-1",
            filename="unicode.pdf",
            file_path="unicode.pdf",
            file_type="pdf",
            total_pages=1,
            ocr_used=False,
        ),
        pages=[PageContent(page_number=1, text=special_text)],
        full_text=special_text,
    )

    preprocessor = TextPreprocessor()
    cleaned_doc = preprocessor.process(doc)

    assert "₹50,000" in cleaned_doc.pages[0].text
    assert "€4,200" in cleaned_doc.pages[0].text
    assert "$1,500.50" in cleaned_doc.pages[0].text
    assert "18% GST" in cleaned_doc.pages[0].text
    assert "Ref: #123/ABC (Legal-Agreement)" in cleaned_doc.pages[0].text
    assert "ગુજરાતી" in cleaned_doc.pages[0].text


def test_text_preprocessor_invalid_input() -> None:
    """Verify TextPreprocessor raises ValueError for non-ProcessedDocument inputs."""
    preprocessor = TextPreprocessor()
    with pytest.raises(ValueError, match="Input must be an instance of ProcessedDocument"):
        preprocessor.process("invalid_input_string")  # type: ignore[arg-type]


# =====================================================================
# 4. Pipeline Integration Tests
# =====================================================================

def test_pdf_ingestion_pipeline(sample_pdf_path: Path) -> None:
    """Verify end-to-end PDF loading -> OCR check -> text preprocessing pipeline."""
    loader = DocumentLoader()
    ocr = OCRProcessor()
    preprocessor = TextPreprocessor()

    # Step 1: Load
    loaded_doc = loader.load(sample_pdf_path)
    assert len(loaded_doc.pages) == 2

    # Step 2: OCR (skipped for PDF)
    ocr_doc = ocr.process(loaded_doc)
    assert ocr_doc.metadata.ocr_used is False

    # Step 3: Preprocess
    final_doc = preprocessor.process(ocr_doc)

    assert final_doc.metadata.filename == "test_contract.pdf"
    assert final_doc.metadata.total_pages == 2
    assert final_doc.pages[0].page_number == 1
    assert "Contract Agreement" in final_doc.pages[0].text
    assert final_doc.pages[1].page_number == 2
    assert "$50,000.00" in final_doc.pages[1].text


def test_image_ingestion_pipeline(
    sample_png_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify end-to-end Image loading -> mocked OCR -> text preprocessing pipeline."""
    # Mock OCR extraction
    mock_ocr_output = "Invoice Number:   INV-999  \nTotal Due:   $1,200.00  \n\n\nThank you!"
    monkeypatch.setattr(
        "app.document.ocr.pytesseract.image_to_string",
        lambda img, lang, config: mock_ocr_output,
    )

    loader = DocumentLoader()
    ocr = OCRProcessor()
    preprocessor = TextPreprocessor()

    # Step 1: Load image placeholder
    loaded_doc = loader.load(sample_png_path)
    assert loaded_doc.pages[0].text == ""

    # Step 2: Extract text via OCR
    ocr_doc = ocr.process(loaded_doc)
    assert ocr_doc.metadata.ocr_used is True

    # Step 3: Preprocess and clean text
    final_doc = preprocessor.process(ocr_doc)

    assert final_doc.metadata.ocr_used is True
    assert final_doc.metadata.filename == "test_invoice.png"
    assert len(final_doc.pages) == 1
    assert "Invoice Number:   INV-999" in final_doc.pages[0].text
    assert "Total Due:   $1,200.00" in final_doc.pages[0].text
    assert "Thank you!" in final_doc.pages[0].text


# Run this test file using:
# pytest tests/test_document.py -v
