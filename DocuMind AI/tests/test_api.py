"""Comprehensive unit and integration tests for the DocuMind AI FastAPI endpoints.

This module tests all REST API routes in app.api.main using pytest and FastAPI TestClient,
mocking underlying LLM, embedding, vector store, and document processing services to ensure
fast, deterministic, and isolated execution without requiring external dependencies or credentials.
"""

from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock
import pytest
from fastapi.testclient import TestClient

from app.api.main import app
from app.models.schemas import (
    ChatResponse,
    ComparisonField,
    ComparisonResult,
    DocumentMetadata,
    ExtractedField,
    PageContent,
    ProcessedDocument,
    SourceCitation,
)
from app.services.extraction_service import StructuredExtractionResult


@pytest.fixture
def client() -> TestClient:
    """Fixture providing a FastAPI TestClient instance."""
    return TestClient(app)


@pytest.fixture
def sample_processed_document() -> ProcessedDocument:
    """Fixture providing a mock ProcessedDocument."""
    metadata = DocumentMetadata(
        document_id="test-doc-uuid-1234",
        filename="test_document.pdf",
        file_path="data/raw/uploads/test_document.pdf",
        file_type="pdf",
        source=None,
        total_pages=2,
        ocr_used=False,
    )
    pages = [
        PageContent(page_number=1, text="Page 1 sample content.", metadata={"page_index": 0}),
        PageContent(page_number=2, text="Page 2 sample content.", metadata={"page_index": 1}),
    ]
    full_text = "Page 1 sample content.\n\nPage 2 sample content."
    return ProcessedDocument(metadata=metadata, pages=pages, full_text=full_text)


# =====================================================================
# 1. Health Check Endpoint Tests
# =====================================================================

def test_health_endpoint(client: TestClient) -> None:
    """Verify GET /health returns HTTP 200 with healthy operational status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "documind-ai"


# =====================================================================
# 2. Chat (RAG Q&A) Endpoint Tests
# =====================================================================

def test_chat_success(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify POST /chat successfully retrieves answers and source citations."""
    mock_rag = MagicMock()
    mock_rag.answer.return_value = ChatResponse(
        answer="The effective date of the agreement is January 1, 2025.",
        sources=[
            SourceCitation(
                document_id="test-doc-uuid-1234",
                filename="agreement.pdf",
                page_number=1,
                content="This Agreement is entered into on January 1, 2025.",
            )
        ],
    )
    monkeypatch.setattr("app.api.main.get_rag_pipeline", lambda: mock_rag)

    payload = {
        "question": "What is the effective date?",
        "document_ids": ["test-doc-uuid-1234"],
        "top_k": 4,
    }
    response = client.post("/chat", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["answer"] == "The effective date of the agreement is January 1, 2025."
    assert len(data["sources"]) == 1
    assert data["sources"][0]["document_id"] == "test-doc-uuid-1234"
    assert data["sources"][0]["filename"] == "agreement.pdf"
    assert data["sources"][0]["page_number"] == 1

    mock_rag.answer.assert_called_once_with(
        question="What is the effective date?",
        top_k=4,
        document_ids=["test-doc-uuid-1234"],
    )


def test_chat_validation_error_missing_question(client: TestClient) -> None:
    """Verify POST /chat returns 422 when required question is missing."""
    response = client.post("/chat", json={"document_ids": ["doc-1"]})
    assert response.status_code == 422


def test_chat_vector_store_not_found(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify POST /chat returns 404 when vector store index is not found."""
    mock_rag = MagicMock()
    mock_rag.answer.side_effect = FileNotFoundError("FAISS index not found on disk.")
    monkeypatch.setattr("app.api.main.get_rag_pipeline", lambda: mock_rag)

    response = client.post("/chat", json={"question": "Where is the contract?"})
    assert response.status_code == 404
    assert "Vector store index not found" in response.json()["detail"]


def test_chat_pipeline_internal_error(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify POST /chat returns 500 when LLM generation encounters an unexpected error."""
    mock_rag = MagicMock()
    mock_rag.answer.side_effect = RuntimeError("API rate limit exceeded.")
    monkeypatch.setattr("app.api.main.get_rag_pipeline", lambda: mock_rag)

    response = client.post("/chat", json={"question": "Summarize page 1"})
    assert response.status_code == 500
    assert "Failed to generate answer" in response.json()["detail"]


# =====================================================================
# 3. Document Upload & Indexing Endpoint Tests
# =====================================================================

def test_upload_success(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    sample_processed_document: ProcessedDocument,
) -> None:
    """Verify POST /documents/upload processes, chunks, and indexes a valid document."""
    # Mock file saving and document loading
    monkeypatch.setattr(
        "app.api.main._save_upload_file",
        lambda file: Path("/tmp/mock_upload.pdf"),
    )
    monkeypatch.setattr(
        "app.api.main._load_and_preprocess_document",
        lambda path: sample_processed_document,
    )

    # Mock chunker
    mock_chunker = MagicMock()
    mock_chunks = [MagicMock(), MagicMock()]
    mock_chunker.chunk.return_value = mock_chunks
    monkeypatch.setattr("app.api.main.get_document_chunker", lambda: mock_chunker)

    # Mock vector store service
    mock_vector_store = MagicMock()
    mock_vector_store.exists.return_value = True
    monkeypatch.setattr("app.api.main.get_vector_store_service", lambda: mock_vector_store)

    fake_pdf = BytesIO(b"%PDF-1.4 mock content")
    response = client.post(
        "/documents/upload",
        files={"file": ("test_document.pdf", fake_pdf, "application/pdf")},
    )

    assert response.status_code == 201
    data = response.json()
    assert data["message"] == "Document processed successfully"
    assert data["document_id"] == "test-doc-uuid-1234"
    assert data["filename"] == "test_document.pdf"
    assert data["document_type"] == "pdf"
    assert data["total_pages"] == 2
    assert data["chunks_indexed"] == 2

    mock_vector_store.add_documents.assert_called_once_with(mock_chunks)


def test_upload_creates_new_vector_store(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    sample_processed_document: ProcessedDocument,
) -> None:
    """Verify POST /documents/upload initializes a new FAISS store if none exists."""
    monkeypatch.setattr(
        "app.api.main._save_upload_file",
        lambda file: Path("/tmp/mock_upload.pdf"),
    )
    monkeypatch.setattr(
        "app.api.main._load_and_preprocess_document",
        lambda path: sample_processed_document,
    )

    mock_chunker = MagicMock()
    mock_chunks = [MagicMock()]
    mock_chunker.chunk.return_value = mock_chunks
    monkeypatch.setattr("app.api.main.get_document_chunker", lambda: mock_chunker)

    mock_vector_store = MagicMock()
    mock_vector_store.exists.return_value = False
    monkeypatch.setattr("app.api.main.get_vector_store_service", lambda: mock_vector_store)

    fake_pdf = BytesIO(b"%PDF-1.4 mock content")
    response = client.post(
        "/documents/upload",
        files={"file": ("test_document.pdf", fake_pdf, "application/pdf")},
    )

    assert response.status_code == 201
    mock_vector_store.create.assert_called_once_with(mock_chunks)


def test_upload_invalid_file_extension(client: TestClient) -> None:
    """Verify POST /documents/upload rejects unsupported file extensions."""
    fake_txt = BytesIO(b"Unsupported text file content")
    response = client.post(
        "/documents/upload",
        files={"file": ("unsupported_document.txt", fake_txt, "text/plain")},
    )
    assert response.status_code == 400
    assert "Unsupported file format" in response.json()["detail"]


def test_upload_empty_file(client: TestClient) -> None:
    """Verify POST /documents/upload rejects empty files."""
    empty_pdf = BytesIO(b"")
    response = client.post(
        "/documents/upload",
        files={"file": ("empty_document.pdf", empty_pdf, "application/pdf")},
    )
    assert response.status_code == 400
    assert "The uploaded file is empty" in response.json()["detail"]


# =====================================================================
# 4. Structured Extraction Endpoint Tests
# =====================================================================

def test_extract_success(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    sample_processed_document: ProcessedDocument,
) -> None:
    """Verify POST /documents/extract returns structured entities and executive summary."""
    monkeypatch.setattr(
        "app.api.main._save_upload_file",
        lambda file: Path("/tmp/mock_invoice.pdf"),
    )
    monkeypatch.setattr(
        "app.api.main._load_and_preprocess_document",
        lambda path: sample_processed_document,
    )

    mock_extraction_service = MagicMock()
    mock_extraction_service.extract_from_document.return_value = StructuredExtractionResult(
        document_id="test-doc-uuid-1234",
        filename="test_document.pdf",
        document_type="invoice",
        fields=[
            ExtractedField(
                field_name="invoice_number",
                value="INV-2025-001",
                confidence=0.98,
                source_page=1,
            ),
            ExtractedField(
                field_name="total_amount",
                value=1500.50,
                confidence=0.95,
                source_page=1,
            ),
        ],
        summary="Invoice INV-2025-001 totaling $1,500.50.",
        raw_text_used=True,
    )
    monkeypatch.setattr("app.api.main.get_extraction_service", lambda: mock_extraction_service)

    fake_pdf = BytesIO(b"%PDF-1.4 invoice content")
    response = client.post(
        "/documents/extract",
        files={"file": ("test_document.pdf", fake_pdf, "application/pdf")},
        data={"document_type": "invoice"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["document_id"] == "test-doc-uuid-1234"
    assert data["document_type"] == "invoice"
    assert len(data["fields"]) == 2
    assert data["fields"][0]["field_name"] == "invoice_number"
    assert data["fields"][0]["value"] == "INV-2025-001"
    assert data["fields"][1]["field_name"] == "total_amount"
    assert data["fields"][1]["value"] == 1500.50
    assert data["summary"] == "Invoice INV-2025-001 totaling $1,500.50."


def test_extract_service_error(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    sample_processed_document: ProcessedDocument,
) -> None:
    """Verify POST /documents/extract returns 500 when extraction service encounters an error."""
    monkeypatch.setattr(
        "app.api.main._save_upload_file",
        lambda file: Path("/tmp/mock_invoice.pdf"),
    )
    monkeypatch.setattr(
        "app.api.main._load_and_preprocess_document",
        lambda path: sample_processed_document,
    )

    mock_extraction_service = MagicMock()
    mock_extraction_service.extract_from_document.side_effect = RuntimeError("Extraction failed.")
    monkeypatch.setattr("app.api.main.get_extraction_service", lambda: mock_extraction_service)

    fake_pdf = BytesIO(b"%PDF-1.4 invoice content")
    response = client.post(
        "/documents/extract",
        files={"file": ("test_document.pdf", fake_pdf, "application/pdf")},
    )

    assert response.status_code == 500
    assert "Failed to extract structured entities" in response.json()["detail"]


# =====================================================================
# 5. Document Comparison Endpoint Tests
# =====================================================================

def test_compare_success(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify POST /documents/compare compares two documents and returns ComparisonResult."""
    # Create two distinct sample documents
    doc_a = ProcessedDocument(
        metadata=DocumentMetadata(
            document_id="doc-a-uuid",
            filename="contract_v1.pdf",
            file_path="data/raw/uploads/contract_v1.pdf",
            file_type="pdf",
            total_pages=1,
            ocr_used=False,
        ),
        pages=[PageContent(page_number=1, text="Contract Amount: $5000")],
        full_text="Contract Amount: $5000",
    )
    doc_b = ProcessedDocument(
        metadata=DocumentMetadata(
            document_id="doc-b-uuid",
            filename="contract_v2.pdf",
            file_path="data/raw/uploads/contract_v2.pdf",
            file_type="pdf",
            total_pages=1,
            ocr_used=False,
        ),
        pages=[PageContent(page_number=1, text="Contract Amount: $6000")],
        full_text="Contract Amount: $6000",
    )

    # Return different documents sequentially
    docs_iter = iter([doc_a, doc_b])
    monkeypatch.setattr(
        "app.api.main._save_upload_file",
        lambda file: Path(f"/tmp/{file.filename}"),
    )
    monkeypatch.setattr(
        "app.api.main._load_and_preprocess_document",
        lambda path: next(docs_iter),
    )

    mock_comparison_service = MagicMock()
    mock_comparison_service.compare_documents.return_value = ComparisonResult(
        document_a="contract_v1.pdf",
        document_b="contract_v2.pdf",
        fields=[
            ComparisonField(
                field_name="contract_value",
                document_a_value=5000,
                document_b_value=6000,
                match=False,
                details="Values differ: Document A contains '5000', while Document B contains '6000'.",
            )
        ],
        has_mismatches=True,
        summary="Compared 1 fields. 0 fields match and 1 fields contain mismatches.",
    )
    monkeypatch.setattr("app.api.main.get_comparison_service", lambda: mock_comparison_service)

    fake_pdf1 = BytesIO(b"%PDF-1.4 contract v1")
    fake_pdf2 = BytesIO(b"%PDF-1.4 contract v2")
    response = client.post(
        "/documents/compare",
        files={
            "file1": ("contract_v1.pdf", fake_pdf1, "application/pdf"),
            "file2": ("contract_v2.pdf", fake_pdf2, "application/pdf"),
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["document_a"] == "contract_v1.pdf"
    assert data["document_b"] == "contract_v2.pdf"
    assert data["has_mismatches"] is True
    assert len(data["fields"]) == 1
    assert data["fields"][0]["match"] is False
    assert data["fields"][0]["document_a_value"] == 5000
    assert data["fields"][0]["document_b_value"] == 6000

    mock_comparison_service.compare_documents.assert_called_once_with(
        document_a_text="Contract Amount: $5000",
        document_b_text="Contract Amount: $6000",
        document_a_id="doc-a-uuid",
        document_b_id="doc-b-uuid",
        document_a_filename="contract_v1.pdf",
        document_b_filename="contract_v2.pdf",
    )


def test_compare_service_error(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    sample_processed_document: ProcessedDocument,
) -> None:
    """Verify POST /documents/compare returns 500 when comparison service fails."""
    monkeypatch.setattr(
        "app.api.main._save_upload_file",
        lambda file: Path(f"/tmp/{file.filename}"),
    )
    monkeypatch.setattr(
        "app.api.main._load_and_preprocess_document",
        lambda path: sample_processed_document,
    )

    mock_comparison_service = MagicMock()
    mock_comparison_service.compare_documents.side_effect = RuntimeError("Comparison failed.")
    monkeypatch.setattr("app.api.main.get_comparison_service", lambda: mock_comparison_service)

    fake_pdf1 = BytesIO(b"%PDF-1.4 doc1")
    fake_pdf2 = BytesIO(b"%PDF-1.4 doc2")
    response = client.post(
        "/documents/compare",
        files={
            "file1": ("doc1.pdf", fake_pdf1, "application/pdf"),
            "file2": ("doc2.pdf", fake_pdf2, "application/pdf"),
        },
    )

    assert response.status_code == 500
    assert "Failed to compare documents" in response.json()["detail"]


# =====================================================================
# 6. Agent Query Endpoint Tests
# =====================================================================

def test_agent_success(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify POST /agent/query delegates execution to DocumentAgent and returns answer."""
    mock_agent = MagicMock()
    mock_agent.run.return_value = "The vendor is Acme Corp and the total payable is $1,200."
    monkeypatch.setattr("app.api.main.get_document_agent", lambda: mock_agent)

    payload = {
        "question": "What is the vendor name and total?",
        "document_ids": ["doc-101"],
        "document_text": "Vendor: Acme Corp. Total: $1,200",
        "document_id": "doc-101",
        "filename": "invoice_101.pdf",
        "document_type": "invoice",
    }
    response = client.post("/agent/query", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "The vendor is Acme Corp and the total payable is $1,200."

    mock_agent.run.assert_called_once_with(
        question="What is the vendor name and total?",
        document_ids=["doc-101"],
        document_text="Vendor: Acme Corp. Total: $1,200",
        document_id="doc-101",
        filename="invoice_101.pdf",
        document_type="invoice",
    )


def test_agent_validation_error_missing_question(client: TestClient) -> None:
    """Verify POST /agent/query returns 422 when required question is missing."""
    response = client.post("/agent/query", json={"filename": "sample.pdf"})
    assert response.status_code == 422


def test_agent_execution_error(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify POST /agent/query returns 500 when agent execution encounters an exception."""
    mock_agent = MagicMock()
    mock_agent.run.side_effect = RuntimeError("Agent execution failed unexpectedly.")
    monkeypatch.setattr("app.api.main.get_document_agent", lambda: mock_agent)

    payload = {"question": "Analyze the legal risks"}
    response = client.post("/agent/query", json=payload)

    assert response.status_code == 500
    assert "Failed to process agent query" in response.json()["detail"]
