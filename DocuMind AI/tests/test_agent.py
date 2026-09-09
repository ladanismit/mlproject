"""Comprehensive unit tests for the DocuMind AI DocumentAgent and Tool Callers.

This module tests DocumentAgent initialization, context formulation, input validation,
autonomous executor invocation, document text parsing helpers, and tool executions
(document_question_answering, extract_document_fields) using mocked LLM and service backends.
"""

from unittest.mock import MagicMock
import pytest

from app.agents.document_agent import DocumentAgent
from app.agents.tools import (
    _build_processed_document,
    _parse_document_text,
    document_question_answering,
    extract_document_fields,
)
from app.core.config import settings
from app.models.schemas import (
    ChatResponse,
    ExtractedField,
    ProcessedDocument,
    SourceCitation,
)
from app.services.extraction_service import StructuredExtractionResult


from langchain_core.runnables import Runnable

# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture
def mock_agent_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fixture ensuring safe offline initialization of LangChain agent components."""
    monkeypatch.setattr(settings, "LLM_PROVIDER", "gemini")
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "mock-gemini-key")
    monkeypatch.setattr("app.agents.document_agent.ChatGoogleGenerativeAI", lambda **kwargs: MagicMock())
    monkeypatch.setattr("app.agents.document_agent.create_tool_calling_agent", lambda **kwargs: MagicMock(spec=Runnable))


# =====================================================================
# 1. DocumentAgent Initialization Tests
# =====================================================================

def test_agent_unsupported_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify DocumentAgent raises ValueError if LLM_PROVIDER is not gemini."""
    monkeypatch.setattr(settings, "LLM_PROVIDER", "anthropic")
    with pytest.raises(ValueError, match="Unsupported LLM provider 'anthropic'"):
        DocumentAgent()


def test_agent_missing_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify DocumentAgent raises ValueError when GEMINI_API_KEY is not configured."""
    monkeypatch.setattr(settings, "LLM_PROVIDER", "gemini")
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "")
    monkeypatch.setattr(settings, "GOOGLE_API_KEY", "")
    with pytest.raises(ValueError, match="GEMINI_API_KEY is not configured"):
        DocumentAgent()


def test_agent_successful_initialization(mock_agent_dependencies: None) -> None:
    """Verify DocumentAgent initializes with expected model, tools, and max iterations."""
    agent = DocumentAgent(model="gemini-1.5-flash", temperature=0.0)

    assert agent.model_name == "gemini-1.5-flash"
    assert agent.temperature == 0.0
    assert len(agent.tools) == 2
    assert document_question_answering in agent.tools
    assert extract_document_fields in agent.tools
    assert agent.executor.max_iterations == 5


def test_agent_empty_custom_tools(mock_agent_dependencies: None) -> None:
    """Verify passing an empty list of custom tools raises ValueError."""
    with pytest.raises(ValueError, match="Custom tools list must not be empty"):
        DocumentAgent(tools=[])


# =====================================================================
# 2. Document Context Building Tests
# =====================================================================

def test_build_document_context_full(mock_agent_dependencies: None) -> None:
    """Verify _build_document_context formats all provided document attributes."""
    agent = DocumentAgent()
    context = agent._build_document_context(
        document_id="doc-001",
        filename="invoice.pdf",
        document_type="invoice",
        document_text="Invoice Number: INV-001\nDate: 2026-09-01\nTotal Amount: ₹50,000",
        document_ids=["doc-001", "doc-002"],
    )

    assert "Document ID: doc-001" in context
    assert "Filename: invoice.pdf" in context
    assert "Document Type Hint: invoice" in context
    assert "Available Document IDs for Retrieval: doc-001, doc-002" in context
    assert "Full Document Text:\nInvoice Number: INV-001\nDate: 2026-09-01\nTotal Amount: ₹50,000" in context


def test_build_document_context_omitted_fields(mock_agent_dependencies: None) -> None:
    """Verify _build_document_context provides clear fallback messages for omitted inputs."""
    agent = DocumentAgent()
    context = agent._build_document_context(
        document_id=None,
        filename=None,
        document_type=None,
        document_text="",
        document_ids=None,
    )

    assert "Document ID: Not provided" in context
    assert "Filename: Not provided" in context
    assert "Document Type Hint: Not provided" in context
    assert "Available Document IDs for Retrieval: Not provided (search all indexed documents)" in context
    assert "Full Document Text: Not provided" in context


# =====================================================================
# 3. DocumentAgent.run() Tests
# =====================================================================

def test_agent_run_success(mock_agent_dependencies: None) -> None:
    """Verify DocumentAgent.run invokes executor and returns expected final answer."""
    agent = DocumentAgent()

    mock_executor_instance = MagicMock()
    mock_executor_instance.invoke.return_value = {
        "output": "The invoice total is ₹50,000.",
    }
    agent.executor = mock_executor_instance

    result = agent.run(
        question="What is the total amount?",
        document_ids=["doc-001"],
        filename="invoice.pdf",
    )

    assert result == "The invoice total is ₹50,000."
    mock_executor_instance.invoke.assert_called_once()
    call_args = mock_executor_instance.invoke.call_args[0][0]
    assert call_args["input"] == "What is the total amount?"
    assert "Filename: invoice.pdf" in call_args["document_context"]


def test_agent_run_citation_question_preserved(mock_agent_dependencies: None) -> None:
    """Verify citation-oriented questions reach the agent unchanged."""
    agent = DocumentAgent()

    mock_executor_instance = MagicMock()
    mock_executor_instance.invoke.return_value = {
        "output": "Page 2 states the total amount is ₹50,000.",
    }
    agent.executor = mock_executor_instance

    query = "What is the total amount on page 2?"
    result = agent.run(question=query)

    assert result == "Page 2 states the total amount is ₹50,000."
    call_args = mock_executor_instance.invoke.call_args[0][0]
    assert call_args["input"] == query


def test_agent_run_multiple_documents(mock_agent_dependencies: None) -> None:
    """Verify multiple document IDs are represented in the generated context."""
    agent = DocumentAgent()

    mock_executor_instance = MagicMock()
    mock_executor_instance.invoke.return_value = {
        "output": "Both documents have been reviewed.",
    }
    agent.executor = mock_executor_instance

    agent.run(
        question="Compare doc-001 and doc-002",
        document_ids=["doc-001", "doc-002"],
    )

    call_args = mock_executor_instance.invoke.call_args[0][0]
    assert "Available Document IDs for Retrieval: doc-001, doc-002" in call_args["document_context"]


def test_agent_run_invalid_question(mock_agent_dependencies: None) -> None:
    """Verify DocumentAgent.run raises ValueError for empty or non-string queries."""
    agent = DocumentAgent()

    with pytest.raises(ValueError, match="Question must be a non-empty string"):
        agent.run(question="")

    with pytest.raises(ValueError, match="Question must be a non-empty string"):
        agent.run(question="   \n ")

    with pytest.raises(ValueError, match="Question must be a non-empty string"):
        agent.run(question=12345)  # type: ignore[arg-type]


def test_agent_run_invalid_document_ids(mock_agent_dependencies: None) -> None:
    """Verify DocumentAgent.run validates document_ids parameter structure."""
    agent = DocumentAgent()

    with pytest.raises(ValueError, match="document_ids must be a list of strings if provided"):
        agent.run(question="Valid question", document_ids="doc-001")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Every document_id must be a string"):
        agent.run(question="Valid question", document_ids=["doc-001", 123])  # type: ignore[list-item]


def test_agent_run_empty_output_error(mock_agent_dependencies: None) -> None:
    """Verify DocumentAgent.run raises RuntimeError when executor returns empty output."""
    agent = DocumentAgent()

    mock_executor_instance = MagicMock()
    mock_executor_instance.invoke.return_value = {"output": ""}
    agent.executor = mock_executor_instance

    with pytest.raises(RuntimeError, match="Document agent returned an empty response"):
        agent.run(question="What is the invoice amount?")


def test_agent_run_executor_failure(mock_agent_dependencies: None) -> None:
    """Verify DocumentAgent.run wraps unhandled executor failures in RuntimeError."""
    agent = DocumentAgent()

    mock_executor_instance = MagicMock()
    mock_executor_instance.invoke.side_effect = RuntimeError("LangChain executor error")
    agent.executor = mock_executor_instance

    with pytest.raises(RuntimeError, match="Document agent execution failed"):
        agent.run(question="Find contract terms")


# =====================================================================
# 4. Tool Tests: document_question_answering
# =====================================================================

def test_tool_question_answering_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify document_question_answering calls RAG pipeline and returns formatted answer with sources."""
    mock_rag = MagicMock()
    mock_rag.answer.return_value = ChatResponse(
        answer="The invoice total is ₹50,000.",
        sources=[
            SourceCitation(
                document_id="doc-001",
                filename="invoice.pdf",
                page_number=2,
                content="Total Amount: ₹50,000",
            )
        ],
    )
    monkeypatch.setattr("app.agents.tools.rag_pipeline", mock_rag)

    tool_result = document_question_answering.invoke(
        {
            "question": "What is the invoice total?",
            "document_ids": ["doc-001"],
            "top_k": 3,
        }
    )

    assert "Answer:\nThe invoice total is ₹50,000." in tool_result
    assert "Sources:" in tool_result
    assert "- invoice.pdf, Page 2" in tool_result
    assert '"Total Amount: ₹50,000"' in tool_result

    mock_rag.answer.assert_called_once_with(
        question="What is the invoice total?",
        top_k=3,
        document_ids=["doc-001"],
    )


def test_tool_question_answering_no_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify document_question_answering handles response with no sources."""
    mock_rag = MagicMock()
    mock_rag.answer.return_value = ChatResponse(
        answer="No specific details found.",
        sources=[],
    )
    monkeypatch.setattr("app.agents.tools.rag_pipeline", mock_rag)

    tool_result = document_question_answering.invoke(
        {
            "question": "What is the penalty clause?",
        }
    )

    assert "Answer:\nNo specific details found." in tool_result
    assert "Sources: None found." in tool_result


def test_tool_question_answering_validation() -> None:
    """Verify document_question_answering rejects invalid questions and invalid top_k."""
    with pytest.raises(ValueError, match="Question must be a non-empty string"):
        document_question_answering.invoke({"question": "  "})

    with pytest.raises(ValueError, match="top_k must be an integer >= 1"):
        document_question_answering.invoke({"question": "Valid query", "top_k": 0})

    with pytest.raises(ValueError, match="top_k must be an integer >= 1"):
        document_question_answering.invoke({"question": "Valid query", "top_k": -1})


def test_tool_question_answering_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify document_question_answering wraps RAG errors in RuntimeError."""
    mock_rag = MagicMock()
    mock_rag.answer.side_effect = RuntimeError("FAISS search failure")
    monkeypatch.setattr("app.agents.tools.rag_pipeline", mock_rag)

    with pytest.raises(RuntimeError, match="RAG question answering tool failed"):
        document_question_answering.invoke({"question": "Check invoice"})


# =====================================================================
# 5. Tool Tests: extract_document_fields
# =====================================================================

def test_tool_extract_document_fields_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify extract_document_fields builds ProcessedDocument, calls extraction service, and returns JSON."""
    mock_extractor = MagicMock()
    mock_extractor.extract_from_document.return_value = StructuredExtractionResult(
        document_id="doc-001",
        filename="invoice.pdf",
        document_type="invoice",
        fields=[
            ExtractedField(field_name="invoice_number", value="INV-001", confidence=1.0, source_page=1),
            ExtractedField(field_name="total_amount", value="₹50,000", confidence=0.98, source_page=2),
        ],
        summary="Invoice summary text.",
        raw_text_used=True,
    )
    monkeypatch.setattr("app.agents.tools.extraction_service", mock_extractor)

    document_content = (
        "[Page 1]\n"
        "Invoice Number: INV-001\n"
        "Vendor: Alpha Supplies\n\n"
        "[Page 2]\n"
        "Total Amount: ₹50,000"
    )
    tool_result = extract_document_fields.invoke(
        {
            "document_text": document_content,
            "document_id": "doc-001",
            "filename": "invoice.pdf",
            "document_type": "invoice",
        }
    )

    assert '"document_id": "doc-001"' in tool_result
    assert '"filename": "invoice.pdf"' in tool_result
    assert '"invoice_number"' in tool_result
    assert '"total_amount"' in tool_result

    mock_extractor.extract_from_document.assert_called_once()
    call_kwargs = mock_extractor.extract_from_document.call_args[1]
    called_doc = call_kwargs["document"]
    assert isinstance(called_doc, ProcessedDocument)
    assert called_doc.metadata.document_id == "doc-001"
    assert called_doc.metadata.filename == "invoice.pdf"
    assert call_kwargs["document_type"] == "invoice"
    assert len(called_doc.pages) == 2


def test_tool_extract_document_fields_validation() -> None:
    """Verify extract_document_fields validates required text, ID, and filename."""
    with pytest.raises(ValueError, match="document_text must be a non-empty string"):
        extract_document_fields.invoke(
            {"document_text": "", "document_id": "doc-1", "filename": "test.pdf"}
        )

    with pytest.raises(ValueError, match="document_id must be a non-empty string"):
        extract_document_fields.invoke(
            {"document_text": "Sample text", "document_id": " ", "filename": "test.pdf"}
        )

    with pytest.raises(ValueError, match="filename must be a non-empty string"):
        extract_document_fields.invoke(
            {"document_text": "Sample text", "document_id": "doc-1", "filename": ""}
        )


def test_tool_extract_document_fields_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify extract_document_fields wraps service failures in RuntimeError."""
    mock_extractor = MagicMock()
    mock_extractor.extract_from_document.side_effect = RuntimeError("LLM rate limit exceeded")
    monkeypatch.setattr("app.agents.tools.extraction_service", mock_extractor)

    with pytest.raises(RuntimeError, match="Structured extraction tool failed for 'test.pdf'"):
        extract_document_fields.invoke(
            {
                "document_text": "Valid document text content",
                "document_id": "doc-001",
                "filename": "test.pdf",
            }
        )


# =====================================================================
# 6. Page Parsing & Document Construction Helpers
# =====================================================================

def test_parse_document_text_with_page_markers() -> None:
    """Verify _parse_document_text parses [Page N] demarcations into chronological PageContent objects."""
    raw_text = (
        "[Page 1]\nInvoice Number: INV-001\nVendor: Alpha Supplies\n"
        "[Page 2]\nTotal Amount: ₹50,000\nPayment Terms: Net 30"
    )

    pages = _parse_document_text(raw_text)
    assert len(pages) == 2
    assert pages[0].page_number == 1
    assert "INV-001" in pages[0].text
    assert pages[1].page_number == 2
    assert "₹50,000" in pages[1].text


def test_parse_document_text_without_page_markers() -> None:
    """Verify _parse_document_text assigns unmarked document content to Page 1."""
    raw_text = "Invoice Number: INV-001\nTotal Amount: ₹50,000"
    pages = _parse_document_text(raw_text)

    assert len(pages) == 1
    assert pages[0].page_number == 1
    assert pages[0].text == raw_text


def test_parse_document_text_invalid_input() -> None:
    """Verify _parse_document_text raises ValueError on empty text or invalid page index."""
    with pytest.raises(ValueError, match="Document text must be a non-empty string"):
        _parse_document_text("   ")

    with pytest.raises(ValueError, match="Invalid page number 0"):
        _parse_document_text("[Page 0]\nInvalid page header")


def test_build_processed_document_helper() -> None:
    """Verify _build_processed_document constructs a valid ProcessedDocument schema."""
    doc_text = "[Page 1]\nInvoice Number: INV-001\n[Page 2]\nTotal Amount: ₹50,000"
    processed_doc = _build_processed_document(
        document_id="doc-001",
        filename="invoice.pdf",
        document_text=doc_text,
        document_type="invoice",
    )

    assert isinstance(processed_doc, ProcessedDocument)
    assert processed_doc.metadata.document_id == "doc-001"
    assert processed_doc.metadata.filename == "invoice.pdf"
    assert processed_doc.metadata.file_type == "pdf"
    assert processed_doc.metadata.source == "invoice"
    assert processed_doc.metadata.total_pages == 2
    assert len(processed_doc.pages) == 2
    assert "Invoice Number: INV-001" in processed_doc.full_text
    assert "Total Amount: ₹50,000" in processed_doc.full_text


def test_build_processed_document_validation() -> None:
    """Verify _build_processed_document validates document_id and filename."""
    with pytest.raises(ValueError, match="document_id must be a non-empty string"):
        _build_processed_document(
            document_id="",
            filename="invoice.pdf",
            document_text="Valid text",
        )

    with pytest.raises(ValueError, match="filename must be a non-empty string"):
        _build_processed_document(
            document_id="doc-001",
            filename="  ",
            document_text="Valid text",
        )


# =====================================================================
# 7. Agent-Tool Integration Test
# =====================================================================

def test_agent_tool_registration(mock_agent_dependencies: None) -> None:
    """Verify DocumentAgent exposes both document_question_answering and extract_document_fields."""
    agent = DocumentAgent()
    registered_tool_names = [getattr(t, "name", str(t)) for t in agent.tools]

    assert "document_question_answering" in registered_tool_names
    assert "extract_document_fields" in registered_tool_names


# Run this test file using:
# pytest tests/test_agent.py -v
