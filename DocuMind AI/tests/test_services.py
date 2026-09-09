"""Comprehensive unit tests for the DocuMind AI Extraction and Comparison Services.

This module tests structured entity extraction, text preparation, schema validation,
field normalization, value reconciliation, mismatch detection, and two-document comparison
using mocked LLM backends to ensure deterministic, offline execution.
"""

from unittest.mock import MagicMock
import pytest

from app.core.config import settings
from app.models.schemas import (
    ComparisonField,
    ComparisonResult,
    DocumentMetadata,
    ExtractedField,
    PageContent,
    ProcessedDocument,
)
from app.services.comparison_service import ComparisonService
from app.services.extraction_service import (
    MAX_EXTRACTION_TEXT_LENGTH,
    ExtractionService,
    StructuredExtractionResult,
)


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture
def sample_processed_doc() -> ProcessedDocument:
    """Fixture providing a multi-page ProcessedDocument for extraction tests."""
    metadata = DocumentMetadata(
        document_id="doc-extract-101",
        filename="invoice_sample.pdf",
        file_path="data/raw/invoice_sample.pdf",
        file_type="pdf",
        source="invoice",
        total_pages=2,
        ocr_used=False,
    )
    pages = [
        PageContent(
            page_number=1,
            text="Invoice Number: INV-2026-001\nDate: 2026-09-01\nVendor: Alpha Supplies Ltd.",
            metadata={"page_index": 0},
        ),
        PageContent(
            page_number=2,
            text="Customer: Beta Corp\nSubtotal: $4,500.00\nTax: $500.00\nTotal Amount: $5,000.00",
            metadata={"page_index": 1},
        ),
    ]
    full_text = "\n\n".join(p.text for p in pages)
    return ProcessedDocument(metadata=metadata, pages=pages, full_text=full_text)


@pytest.fixture
def mock_extraction_service() -> MagicMock:
    """Fixture providing a preconfigured mock ExtractionService."""
    service = MagicMock(spec=ExtractionService)
    return service


# =====================================================================
# 1. ExtractionService Tests
# =====================================================================

def test_extraction_service_unsupported_provider() -> None:
    """Verify ExtractionService rejects unsupported LLM providers."""
    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        ExtractionService(provider="unsupported_llm_provider")


def test_extraction_service_missing_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify ExtractionService raises ValueError when GEMINI_API_KEY is not configured."""
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "")
    monkeypatch.setattr(settings, "GOOGLE_API_KEY", "")
    with pytest.raises(ValueError, match="GEMINI_API_KEY is not configured"):
        ExtractionService(provider="gemini")


def test_extraction_service_initialization_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify ExtractionService instantiates ChatGoogleGenerativeAI and builds structured prompt."""
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "mock-extraction-key")
    mock_chat_gemini = MagicMock()
    monkeypatch.setattr("app.services.extraction_service.ChatGoogleGenerativeAI", lambda **kwargs: mock_chat_gemini)

    service = ExtractionService(provider="gemini", model_name="gemini-1.5-flash", temperature=0.0)
    assert service.provider == "gemini"
    assert service.model_name == "gemini-1.5-flash"
    assert service.temperature == 0.0
    mock_chat_gemini.with_structured_output.assert_called_once_with(StructuredExtractionResult)


def test_prepare_document_text_formatting(
    monkeypatch: pytest.MonkeyPatch,
    sample_processed_doc: ProcessedDocument,
) -> None:
    """Verify _prepare_document_text embeds page markers and preserves multi-page contents."""
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "mock-key")
    monkeypatch.setattr("app.services.extraction_service.ChatGoogleGenerativeAI", lambda **kwargs: MagicMock())

    service = ExtractionService()
    prepared_text = service._prepare_document_text(sample_processed_doc)

    assert "[Page 1]" in prepared_text
    assert "[Page 2]" in prepared_text
    assert "INV-2026-001" in prepared_text
    assert "$5,000.00" in prepared_text


def test_prepare_document_text_empty_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify _prepare_document_text raises ValueError if document has no text."""
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "mock-key")
    monkeypatch.setattr("app.services.extraction_service.ChatGoogleGenerativeAI", lambda **kwargs: MagicMock())

    empty_doc = ProcessedDocument(
        metadata=DocumentMetadata(
            document_id="empty-1",
            filename="empty.pdf",
            file_path="empty.pdf",
            file_type="pdf",
            total_pages=1,
            ocr_used=False,
        ),
        pages=[],
        full_text="",
    )

    service = ExtractionService()
    with pytest.raises(ValueError, match="contains no readable text"):
        service._prepare_document_text(empty_doc)


def test_prepare_document_text_truncation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify _prepare_document_text truncates oversized documents preserving head and tail."""
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "mock-key")
    monkeypatch.setattr("app.services.extraction_service.ChatGoogleGenerativeAI", lambda **kwargs: MagicMock())

    # Create oversized text exceeding MAX_EXTRACTION_TEXT_LENGTH (30,000 characters)
    huge_text_page1 = "START_OF_DOCUMENT " + ("A" * (MAX_EXTRACTION_TEXT_LENGTH // 2 + 1000))
    huge_text_page2 = ("B" * (MAX_EXTRACTION_TEXT_LENGTH // 2 + 1000)) + " END_OF_DOCUMENT"

    huge_doc = ProcessedDocument(
        metadata=DocumentMetadata(
            document_id="huge-1",
            filename="huge.pdf",
            file_path="huge.pdf",
            file_type="pdf",
            total_pages=2,
            ocr_used=False,
        ),
        pages=[
            PageContent(page_number=1, text=huge_text_page1),
            PageContent(page_number=2, text=huge_text_page2),
        ],
        full_text=huge_text_page1 + "\n\n" + huge_text_page2,
    )

    service = ExtractionService()
    prepared_text = service._prepare_document_text(huge_doc)

    assert "START_OF_DOCUMENT" in prepared_text
    assert "TRUNCATED" in prepared_text


def test_extract_from_document_success(
    monkeypatch: pytest.MonkeyPatch,
    sample_processed_doc: ProcessedDocument,
) -> None:
    """Verify extract_from_document successfully returns validated StructuredExtractionResult."""
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "mock-key")

    mock_llm = MagicMock()
    monkeypatch.setattr("app.services.extraction_service.ChatGoogleGenerativeAI", lambda **kwargs: mock_llm)

    service = ExtractionService()

    # Mock structured output invocation
    mock_extracted = StructuredExtractionResult(
        document_id="will-be-aligned",
        filename="will-be-aligned.pdf",
        document_type="invoice",
        fields=[
            ExtractedField(field_name="invoice_number", value="INV-2026-001", confidence=0.99, source_page=1),
            ExtractedField(field_name="total_amount", value=5000.00, confidence=0.95, source_page=2),
        ],
        summary="Invoice INV-2026-001 from Alpha Supplies Ltd totaling $5,000.00.",
        raw_text_used=True,
    )

    monkeypatch.setattr(
        service.prompt_template.__class__,
        "__or__",
        lambda self, other: MagicMock(invoke=lambda x: mock_extracted),
    )

    result = service.extract_from_document(sample_processed_doc, document_type="invoice")

    assert result.document_id == "doc-extract-101"
    assert result.filename == "invoice_sample.pdf"
    assert result.document_type == "invoice"
    assert len(result.fields) == 2
    assert result.fields[0].field_name == "invoice_number"
    assert result.fields[0].value == "INV-2026-001"
    assert result.fields[1].value == 5000.00
    assert result.raw_text_used is True


def test_validate_result_invalid_confidence(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify _validate_result raises ValueError when field confidence is out of [0.0, 1.0] bounds."""
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "mock-key")
    monkeypatch.setattr("app.services.extraction_service.ChatGoogleGenerativeAI", lambda **kwargs: MagicMock())

    service = ExtractionService()

    invalid_result = StructuredExtractionResult.model_construct(
        document_id="doc-1",
        filename="doc.pdf",
        fields=[
            ExtractedField.model_construct(field_name="total", value=100, confidence=1.5),  # Invalid confidence > 1.0
        ],
        summary="Summary text",
        raw_text_used=True,
    )

    with pytest.raises(ValueError, match="Invalid confidence for field 'total'"):
        service._validate_result(invalid_result, "doc-1", "doc.pdf")


def test_extraction_service_error_handling(
    monkeypatch: pytest.MonkeyPatch,
    sample_processed_doc: ProcessedDocument,
) -> None:
    """Verify extract_from_document wraps unexpected LLM failures in RuntimeError."""
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "mock-key")
    monkeypatch.setattr("app.services.extraction_service.ChatGoogleGenerativeAI", lambda **kwargs: MagicMock())

    service = ExtractionService()
    mock_chain = MagicMock()
    mock_chain.invoke.side_effect = RuntimeError("LLM structured output failed")
    monkeypatch.setattr(
        service.prompt_template.__class__,
        "__or__",
        lambda self, other: mock_chain,
    )

    with pytest.raises(RuntimeError, match="Structured extraction failed for 'invoice_sample.pdf'"):
        service.extract_from_document(sample_processed_doc)


# =====================================================================
# 2. ComparisonService Tests
# =====================================================================

def test_comparison_service_initialization() -> None:
    """Verify ComparisonService initializes with default or injected ExtractionService."""
    mock_extractor = MagicMock(spec=ExtractionService)
    service = ComparisonService(extraction_service=mock_extractor)
    assert service.extraction_service == mock_extractor


@pytest.mark.parametrize(
    ("raw_name", "expected_normalized"),
    [
        ("Invoice No.", "invoice_number"),
        ("Inv_Num", "invoice_number"),
        ("Bill Number", "invoice_number"),
        ("Date", "invoice_date"),
        ("Agreement Date", "contract_date"),
        ("Vendor", "vendor_name"),
        ("Supplier Name", "vendor_name"),
        ("Buyer", "buyer_name"),
        ("Total", "total_amount"),
        ("Grand Total", "total_amount"),
        ("GST", "tax"),
        ("VAT", "tax"),
        ("Jurisdiction", "governing_law"),
        ("Custom Unique Field", "custom_unique_field"),
    ],
)
def test_normalize_field_name(raw_name: str, expected_normalized: str) -> None:
    """Verify _normalize_field_name standardizes canonical synonyms and formats snake_case."""
    service = ComparisonService(extraction_service=MagicMock())
    assert service._normalize_field_name(raw_name) == expected_normalized


@pytest.mark.parametrize(
    ("raw_val", "expected_val"),
    [
        ("₹50,000", 50000),
        ("$1,500.50", 1500.5),
        ("50,000.00", 50000),
        ("2026-09-01", "2026-09-01"),
        ("01/09/2026", "2026-09-01"),
        ("1 September 2026", "2026-09-01"),
        ("ABC Industries Pvt. Ltd.", "abc industries pvt ltd"),
        ("  Extra   Spaces  ", "extra spaces"),
        ("00123", 123),  # Numeric-looking strings parse as integer
        (100, 100),
        (None, None),
    ],
)
def test_normalize_value(raw_val: object, expected_val: object) -> None:
    """Verify _normalize_value handles currency, dates, numbers, and text casing."""
    service = ComparisonService(extraction_service=MagicMock())
    assert service._normalize_value(raw_val) == expected_val


def test_build_field_map() -> None:
    """Verify _build_field_map maps ExtractedField objects by canonical normalized names."""
    service = ComparisonService(extraction_service=MagicMock())
    fields = [
        ExtractedField(field_name="Invoice No.", value="INV-001"),
        ExtractedField(field_name="Grand Total", value="₹50,000"),
    ]

    field_map = service._build_field_map(fields)
    assert "invoice_number" in field_map
    assert "total_amount" in field_map
    assert field_map["invoice_number"].value == "INV-001"
    assert field_map["total_amount"].value == "₹50,000"


def test_compare_identical_documents() -> None:
    """Verify comparing identical field maps reports zero mismatches."""
    service = ComparisonService(extraction_service=MagicMock())
    map_a = {
        "invoice_number": ExtractedField(field_name="Invoice No.", value="INV-001"),
        "total_amount": ExtractedField(field_name="Total", value="$5,000.00"),
    }
    map_b = {
        "invoice_number": ExtractedField(field_name="Invoice #", value="INV-001"),
        "total_amount": ExtractedField(field_name="Grand Total", value="5000"),
    }

    results = service._compare_fields(map_a, map_b)
    assert len(results) == 2
    assert all(field.match for field in results)

    summary = service._build_summary(results)
    assert "All 2 comparable fields match" in summary


def test_compare_single_mismatch() -> None:
    """Verify comparing documents with one changed value flags the discrepancy."""
    service = ComparisonService(extraction_service=MagicMock())
    map_a = {
        "total_amount": ExtractedField(field_name="Total", value="₹50,000"),
    }
    map_b = {
        "total_amount": ExtractedField(field_name="Total", value="₹55,000"),
    }

    results = service._compare_fields(map_a, map_b)
    assert len(results) == 1
    assert results[0].match is False
    assert results[0].field_name == "total_amount"
    assert "Values differ" in (results[0].details or "")


def test_compare_missing_fields() -> None:
    """Verify fields missing from either document are appropriately flagged as mismatches."""
    service = ComparisonService(extraction_service=MagicMock())
    map_a = {
        "vendor_name": ExtractedField(field_name="Vendor", value="Alpha Corp"),
        "tax": ExtractedField(field_name="Tax", value="500"),
    }
    map_b = {
        "vendor_name": ExtractedField(field_name="Vendor", value="Alpha Corp"),
        "buyer_name": ExtractedField(field_name="Buyer", value="Beta Ltd"),
    }

    results = service._compare_fields(map_a, map_b)
    assert len(results) == 3

    # vendor_name matches
    vendor_field = next(f for f in results if f.field_name == "vendor_name")
    assert vendor_field.match is True

    # tax is missing from Document B
    tax_field = next(f for f in results if f.field_name == "tax")
    assert tax_field.match is False
    assert "missing from Document B" in (tax_field.details or "")

    # buyer_name is missing from Document A
    buyer_field = next(f for f in results if f.field_name == "buyer_name")
    assert buyer_field.match is False
    assert "missing from Document A" in (buyer_field.details or "")


def test_compare_multiple_mismatches() -> None:
    """Verify multiple simultaneous discrepancies (amount, party, date) are captured."""
    service = ComparisonService(extraction_service=MagicMock())
    map_a = {
        "total_amount": ExtractedField(field_name="Total", value="₹50,000"),
        "vendor_name": ExtractedField(field_name="Vendor", value="Acme Corp"),
        "invoice_date": ExtractedField(field_name="Date", value="2026-09-01"),
    }
    map_b = {
        "total_amount": ExtractedField(field_name="Total", value="₹55,000"),
        "vendor_name": ExtractedField(field_name="Vendor", value="Beta Corp"),
        "invoice_date": ExtractedField(field_name="Date", value="2026-09-15"),
    }

    results = service._compare_fields(map_a, map_b)
    assert len(results) == 3
    assert all(not f.match for f in results)

    summary = service._build_summary(results)
    assert "3 fields contain mismatches" in summary


def test_compare_documents_validation_errors() -> None:
    """Verify compare_documents validates non-empty inputs and distinct document IDs."""
    service = ComparisonService(extraction_service=MagicMock())

    # Empty text validation
    with pytest.raises(ValueError, match="Document A text must be a non-empty string"):
        service.compare_documents(
            document_a_text="",
            document_b_text="text B",
            document_a_id="id-1",
            document_b_id="id-2",
            document_a_filename="a.pdf",
            document_b_filename="b.pdf",
        )

    # Identical document IDs validation
    with pytest.raises(ValueError, match="Document IDs must be different for comparison"):
        service.compare_documents(
            document_a_text="text A",
            document_b_text="text B",
            document_a_id="same-id",
            document_b_id="same-id",
            document_a_filename="a.pdf",
            document_b_filename="b.pdf",
        )


def test_compare_documents_end_to_end() -> None:
    """Verify compare_documents end-to-end flow with mocked ExtractionService."""
    mock_extractor = MagicMock(spec=ExtractionService)

    # Document A extracted fields
    result_a = StructuredExtractionResult(
        document_id="doc-a",
        filename="invoice_a.pdf",
        document_type="invoice",
        fields=[
            ExtractedField(field_name="Invoice Number", value="INV-001"),
            ExtractedField(field_name="Total Amount", value="₹50,000"),
        ],
        summary="Invoice A summary",
        raw_text_used=True,
    )

    # Document B extracted fields (with mismatch on amount)
    result_b = StructuredExtractionResult(
        document_id="doc-b",
        filename="invoice_b.pdf",
        document_type="invoice",
        fields=[
            ExtractedField(field_name="Invoice Number", value="INV-001"),
            ExtractedField(field_name="Total Amount", value="₹55,000"),
        ],
        summary="Invoice B summary",
        raw_text_used=True,
    )

    mock_extractor.extract_from_document.side_effect = [result_a, result_b]

    service = ComparisonService(extraction_service=mock_extractor)
    comparison: ComparisonResult = service.compare_documents(
        document_a_text="Invoice Number: INV-001\nTotal Amount: ₹50,000",
        document_b_text="Invoice Number: INV-001\nTotal Amount: ₹55,000",
        document_a_id="doc-a",
        document_b_id="doc-b",
        document_a_filename="invoice_a.pdf",
        document_b_filename="invoice_b.pdf"
    )

    assert comparison.document_a == "invoice_a.pdf"
    assert comparison.document_b == "invoice_b.pdf"
    assert comparison.has_mismatches is True
    assert len(comparison.fields) == 2

    match_field = next(f for f in comparison.fields if f.field_name == "invoice_number")
    assert match_field.match is True

    mismatch_field = next(f for f in comparison.fields if f.field_name == "total_amount")
    assert mismatch_field.match is False
    assert mismatch_field.document_a_value == "₹50,000"
    assert mismatch_field.document_b_value == "₹55,000"

    assert mock_extractor.extract_from_document.call_count == 2


# Run this test file using:
# pytest tests/test_services.py -v
