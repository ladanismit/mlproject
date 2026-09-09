"""Unit tests for the DocuMind AI RAG subsystem.

This test suite verifies document chunking, embedding abstractions, persistent FAISS
vector storage, semantic retrieval with metadata filtering, and full RAG pipeline
question-answering with citation synthesis, using mocked LLM and embedding backends.
"""

from pathlib import Path
from unittest.mock import MagicMock
import pytest
from langchain_core.documents import Document

from app.core.config import settings
from app.models.schemas import (
    ChatResponse,
    DocumentMetadata,
    PageContent,
    ProcessedDocument,
    SourceCitation,
)
from app.rag.chunker import DocumentChunker
from app.rag.embeddings import EmbeddingService
from app.rag.pipeline import RAGPipeline
from app.rag.retriever import DocumentRetriever
from app.rag.vector_store import VectorStoreService


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture
def sample_metadata() -> DocumentMetadata:
    """Fixture providing a sample DocumentMetadata model."""
    return DocumentMetadata(
        document_id="doc-uuid-101",
        filename="invoice_1001.pdf",
        file_path="data/raw/invoice_1001.pdf",
        file_type="pdf",
        source="invoice",
        total_pages=2,
        ocr_used=False,
    )


@pytest.fixture
def sample_processed_document(sample_metadata: DocumentMetadata) -> ProcessedDocument:
    """Fixture providing a multi-page ProcessedDocument."""
    pages = [
        PageContent(
            page_number=1,
            text="Invoice Number: INV-1001\nVendor: Global Tech Inc.\nDate: 2025-01-15",
            metadata={"page_index": 0},
        ),
        PageContent(
            page_number=2,
            text="Total Amount: $1,500.00\nPayment Terms: Net 30 Days\nDue Date: 2025-02-15",
            metadata={"page_index": 1},
        ),
    ]
    full_text = "\n\n".join(p.text for p in pages)
    return ProcessedDocument(
        metadata=sample_metadata,
        pages=pages,
        full_text=full_text,
    )


@pytest.fixture
def sample_chunks() -> list[Document]:
    """Fixture providing standard LangChain Document chunks."""
    return [
        Document(
            page_content="Invoice Number: INV-1001\nVendor: Global Tech Inc.",
            metadata={
                "document_id": "doc-uuid-101",
                "filename": "invoice_1001.pdf",
                "file_type": "pdf",
                "source": "invoice",
                "page_number": 1,
                "chunk_index": 0,
                "ocr_used": False,
            },
        ),
        Document(
            page_content="Total Amount: $1,500.00\nPayment Terms: Net 30 Days",
            metadata={
                "document_id": "doc-uuid-101",
                "filename": "invoice_1001.pdf",
                "file_type": "pdf",
                "source": "invoice",
                "page_number": 2,
                "chunk_index": 1,
                "ocr_used": False,
            },
        ),
    ]


@pytest.fixture
def mock_embedding_service() -> MagicMock:
    """Fixture providing a mock EmbeddingService."""
    mock_service = MagicMock(spec=EmbeddingService)
    mock_embeddings = MagicMock()
    mock_service.get_embeddings.return_value = mock_embeddings
    mock_service.embed_documents.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    mock_service.embed_query.return_value = [0.1, 0.2, 0.3]
    return mock_service


# =====================================================================
# 1. Document Chunker Tests
# =====================================================================

def test_chunker_initialization_defaults() -> None:
    """Verify DocumentChunker initializes with settings defaults."""
    chunker = DocumentChunker()
    assert chunker.chunk_size == settings.CHUNK_SIZE
    assert chunker.chunk_overlap == settings.CHUNK_OVERLAP


def test_chunker_invalid_parameters() -> None:
    """Verify DocumentChunker raises ValueError for invalid size/overlap configurations."""
    with pytest.raises(ValueError, match="chunk_size must be greater than 0"):
        DocumentChunker(chunk_size=0)

    with pytest.raises(ValueError, match="chunk_overlap .* must satisfy"):
        DocumentChunker(chunk_size=500, chunk_overlap=500)

    with pytest.raises(ValueError, match="chunk_overlap .* must satisfy"):
        DocumentChunker(chunk_size=500, chunk_overlap=-10)


def test_chunker_chunks_processed_document(sample_processed_document: ProcessedDocument) -> None:
    """Verify DocumentChunker generates chunks with preserved page-level metadata."""
    chunker = DocumentChunker(chunk_size=1000, chunk_overlap=100)
    chunks = chunker.chunk(sample_processed_document)

    assert len(chunks) == 2
    # Verify Page 1 chunk
    assert "INV-1001" in chunks[0].page_content
    assert chunks[0].metadata["document_id"] == "doc-uuid-101"
    assert chunks[0].metadata["filename"] == "invoice_1001.pdf"
    assert chunks[0].metadata["page_number"] == 1
    assert chunks[0].metadata["chunk_index"] == 0

    # Verify Page 2 chunk
    assert "1,500.00" in chunks[1].page_content
    assert chunks[1].metadata["page_number"] == 2
    assert chunks[1].metadata["chunk_index"] == 1


def test_chunker_skips_empty_pages(sample_metadata: DocumentMetadata) -> None:
    """Verify DocumentChunker gracefully ignores empty or whitespace-only pages."""
    doc = ProcessedDocument(
        metadata=sample_metadata,
        pages=[
            PageContent(page_number=1, text=""),
            PageContent(page_number=2, text="   \n  "),
            PageContent(page_number=3, text="Valid content on page 3."),
        ],
        full_text="Valid content on page 3.",
    )
    chunker = DocumentChunker()
    chunks = chunker.chunk(doc)

    assert len(chunks) == 1
    assert chunks[0].metadata["page_number"] == 3
    assert chunks[0].page_content == "Valid content on page 3."


def test_chunker_invalid_input_type() -> None:
    """Verify DocumentChunker raises ValueError if input is not a ProcessedDocument."""
    chunker = DocumentChunker()
    with pytest.raises(ValueError, match="Input must be an instance of ProcessedDocument"):
        chunker.chunk("invalid string input")  # type: ignore[arg-type]


# =====================================================================
# 2. Embedding Service Tests
# =====================================================================

def test_embeddings_unsupported_provider() -> None:
    """Verify EmbeddingService rejects unsupported providers."""
    with pytest.raises(ValueError, match="Unsupported embedding provider"):
        EmbeddingService(provider="unsupported_provider")


def test_embeddings_missing_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify EmbeddingService raises ValueError when GEMINI_API_KEY is not configured."""
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "")
    monkeypatch.setattr(settings, "GOOGLE_API_KEY", "")
    with pytest.raises(ValueError, match="GEMINI_API_KEY is not configured"):
        EmbeddingService(provider="gemini")


def test_embeddings_initialization_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify EmbeddingService instantiates GoogleGenerativeAIEmbeddings when key is provided."""
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "mock-test-key")
    mock_gemini_cls = MagicMock()
    monkeypatch.setattr("app.rag.embeddings.GoogleGenerativeAIEmbeddings", mock_gemini_cls)

    service = EmbeddingService(provider="gemini", model_name="models/text-embedding-004")
    assert service.provider == "gemini"
    assert service.model_name == "models/text-embedding-004"
    mock_gemini_cls.assert_called_once_with(
        model="models/text-embedding-004",
        google_api_key="mock-test-key",
    )


def test_embed_documents_and_query(
    monkeypatch: pytest.MonkeyPatch, sample_chunks: list[Document]
) -> None:
    """Verify embed_documents and embed_query delegate correctly to underlying embeddings."""
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "mock-test-key")
    mock_gemini_instance = MagicMock()
    mock_gemini_instance.embed_documents.return_value = [[0.1, 0.2], [0.3, 0.4]]
    mock_gemini_instance.embed_query.return_value = [0.1, 0.2]
    monkeypatch.setattr(
        "app.rag.embeddings.GoogleGenerativeAIEmbeddings",
        lambda **kwargs: mock_gemini_instance,
    )

    service = EmbeddingService(provider="gemini")

    # Test embed_documents
    vectors = service.embed_documents(sample_chunks)
    assert len(vectors) == 2
    mock_gemini_instance.embed_documents.assert_called_once_with(
        [chunk.page_content for chunk in sample_chunks]
    )

    # Test embed_query
    query_vec = service.embed_query("Invoice total?")
    assert query_vec == [0.1, 0.2]
    mock_gemini_instance.embed_query.assert_called_once_with("Invoice total?")


def test_embed_query_validation() -> None:
    """Verify embed_query rejects empty strings."""
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "mock-test-key")
    monkeypatch.setattr("app.rag.embeddings.GoogleGenerativeAIEmbeddings", lambda **kwargs: MagicMock())

    service = EmbeddingService(provider="gemini")
    with pytest.raises(ValueError, match="Query must be a non-empty string"):
        service.embed_query("   ")


# =====================================================================
# 3. Vector Store Service Tests
# =====================================================================

def test_vector_store_exists_check(tmp_path: Path, mock_embedding_service: MagicMock) -> None:
    """Verify exists() correctly detects presence of FAISS files on disk."""
    service = VectorStoreService(
        embedding_service=mock_embedding_service,
        persist_directory=tmp_path,
        index_name="test_index",
    )
    assert service.exists() is False

    # Create fake .faiss and .pkl files
    (tmp_path / "test_index.faiss").write_bytes(b"dummy faiss data")
    assert service.exists() is False  # Missing .pkl

    (tmp_path / "test_index.pkl").write_bytes(b"dummy pkl data")
    assert service.exists() is True


def test_vector_store_create(
    tmp_path: Path,
    mock_embedding_service: MagicMock,
    sample_chunks: list[Document],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify create() initializes FAISS index and persists to disk."""
    mock_faiss_instance = MagicMock()
    mock_faiss_cls = MagicMock()
    mock_faiss_cls.from_documents.return_value = mock_faiss_instance
    monkeypatch.setattr("app.rag.vector_store.FAISS", mock_faiss_cls)

    service = VectorStoreService(
        embedding_service=mock_embedding_service,
        persist_directory=tmp_path,
        index_name="test_index",
    )

    result = service.create(sample_chunks)
    assert result == mock_faiss_instance
    mock_faiss_cls.from_documents.assert_called_once_with(
        sample_chunks,
        mock_embedding_service.get_embeddings(),
    )
    mock_faiss_instance.save_local.assert_called_once_with(
        str(tmp_path.resolve()),
        index_name="test_index",
    )


def test_vector_store_load_not_found(tmp_path: Path, mock_embedding_service: MagicMock) -> None:
    """Verify load() raises FileNotFoundError if index files do not exist."""
    service = VectorStoreService(
        embedding_service=mock_embedding_service,
        persist_directory=tmp_path,
        index_name="test_index",
    )
    with pytest.raises(FileNotFoundError, match="FAISS index 'test_index' not found"):
        service.load()


def test_vector_store_add_documents(
    tmp_path: Path,
    mock_embedding_service: MagicMock,
    sample_chunks: list[Document],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify add_documents() loads existing store, appends chunks, and saves index."""
    (tmp_path / "test_index.faiss").write_bytes(b"dummy")
    (tmp_path / "test_index.pkl").write_bytes(b"dummy")

    mock_loaded_faiss = MagicMock()
    mock_faiss_cls = MagicMock()
    mock_faiss_cls.load_local.return_value = mock_loaded_faiss
    monkeypatch.setattr("app.rag.vector_store.FAISS", mock_faiss_cls)

    service = VectorStoreService(
        embedding_service=mock_embedding_service,
        persist_directory=tmp_path,
        index_name="test_index",
    )

    service.add_documents(sample_chunks)
    mock_loaded_faiss.add_documents.assert_called_once_with(sample_chunks)
    mock_loaded_faiss.save_local.assert_called_once_with(
        str(tmp_path.resolve()),
        index_name="test_index",
    )


def test_vector_store_rebuild(
    tmp_path: Path,
    mock_embedding_service: MagicMock,
    sample_chunks: list[Document],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify rebuild() delegates to create() to overwrite existing index."""
    mock_faiss_instance = MagicMock()
    mock_faiss_cls = MagicMock()
    mock_faiss_cls.from_documents.return_value = mock_faiss_instance
    monkeypatch.setattr("app.rag.vector_store.FAISS", mock_faiss_cls)

    service = VectorStoreService(
        embedding_service=mock_embedding_service,
        persist_directory=tmp_path,
        index_name="test_index",
    )

    result = service.rebuild(sample_chunks)
    assert result == mock_faiss_instance
    mock_faiss_cls.from_documents.assert_called_once()


# =====================================================================
# 4. Document Retriever Tests
# =====================================================================

def test_retriever_initialization_and_validation() -> None:
    """Verify DocumentRetriever top_k validation."""
    with pytest.raises(ValueError, match="default_top_k must be at least 1"):
        DocumentRetriever(default_top_k=0)


def test_retriever_similarity_search(sample_chunks: list[Document]) -> None:
    """Verify retrieve_with_scores performs similarity search over vector store."""
    mock_store = MagicMock()
    # FAISS distance scores: lower score indicates closer distance/match
    mock_scored_results = [(sample_chunks[0], 0.25), (sample_chunks[1], 0.85)]
    mock_store.similarity_search_with_score.return_value = mock_scored_results

    mock_vector_service = MagicMock(spec=VectorStoreService)
    mock_vector_service.get_store.return_value = mock_store

    retriever = DocumentRetriever(vector_store_service=mock_vector_service, default_top_k=4)
    results = retriever.retrieve_with_scores("What is the invoice number?", top_k=2)

    assert len(results) == 2
    assert results[0][0].metadata["page_number"] == 1
    assert results[0][1] == 0.25
    mock_store.similarity_search_with_score.assert_called_once_with(
        "What is the invoice number?",
        k=2,
    )


def test_retriever_document_id_filtering(sample_chunks: list[Document]) -> None:
    """Verify retriever filters results to only matching target document IDs."""
    doc_target = sample_chunks[0]  # doc_id: doc-uuid-101
    doc_other = Document(
        page_content="Other contract text",
        metadata={"document_id": "doc-other-999", "page_number": 1},
    )

    mock_store = MagicMock()
    mock_store.similarity_search_with_score.return_value = [
        (doc_target, 0.20),
        (doc_other, 0.15),
    ]

    mock_vector_service = MagicMock(spec=VectorStoreService)
    mock_vector_service.get_store.return_value = mock_store

    retriever = DocumentRetriever(vector_store_service=mock_vector_service)
    results = retriever.retrieve(
        query="Find invoice",
        top_k=2,
        document_ids=["doc-uuid-101"],
    )

    assert len(results) == 1
    assert results[0].metadata["document_id"] == "doc-uuid-101"


def test_retriever_empty_query_validation() -> None:
    """Verify retriever raises ValueError for empty search query."""
    retriever = DocumentRetriever(vector_store_service=MagicMock())
    with pytest.raises(ValueError, match="Search query must be a non-empty string"):
        retriever.retrieve("   ")


# =====================================================================
# 5. RAG Pipeline Tests
# =====================================================================

def test_rag_pipeline_successful_answer(
    monkeypatch: pytest.MonkeyPatch, sample_chunks: list[Document]
) -> None:
    """Verify RAGPipeline builds grounded context, queries LLM, and formats citations."""
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "mock-test-key")

    mock_retriever = MagicMock(spec=DocumentRetriever)
    mock_retriever.retrieve_with_scores.return_value = [
        (sample_chunks[0], 0.20),
        (sample_chunks[1], 0.35),
    ]

    mock_llm_response = MagicMock()
    mock_llm_response.content = "The invoice number is INV-1001 and total is $1,500.00."

    mock_chat_model = MagicMock()
    mock_chat_model.invoke.return_value = mock_llm_response
    monkeypatch.setattr("app.rag.pipeline.ChatGoogleGenerativeAI", lambda **kwargs: mock_chat_model)

    pipeline = RAGPipeline(retriever=mock_retriever)
    # Mock chain invocation directly
    monkeypatch.setattr(
        pipeline.prompt_template.__class__,
        "__or__",
        lambda self, other: MagicMock(invoke=lambda x: mock_llm_response),
    )

    response: ChatResponse = pipeline.answer(
        question="What is the invoice number and total?",
        top_k=2,
        document_ids=["doc-uuid-101"],
    )

    assert response.answer == "The invoice number is INV-1001 and total is $1,500.00."
    assert len(response.sources) == 2
    assert response.sources[0].filename == "invoice_1001.pdf"
    assert response.sources[0].page_number == 1
    assert "INV-1001" in response.sources[0].content
    assert response.sources[1].page_number == 2
    assert "1,500.00" in response.sources[1].content


def test_rag_pipeline_deduplicates_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify RAGPipeline deduplicates identical citation snippets from the same page."""
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "mock-test-key")
    monkeypatch.setattr("app.rag.pipeline.ChatGoogleGenerativeAI", lambda **kwargs: MagicMock())

    duplicate_chunk = Document(
        page_content="Identical snippet content",
        metadata={"document_id": "doc-1", "filename": "doc.pdf", "page_number": 1},
    )

    mock_retriever = MagicMock(spec=DocumentRetriever)
    mock_retriever.retrieve_with_scores.return_value = [
        (duplicate_chunk, 0.10),
        (duplicate_chunk, 0.12),
    ]

    mock_llm_response = MagicMock(content="Answer text.")
    pipeline = RAGPipeline(retriever=mock_retriever)
    monkeypatch.setattr(
        pipeline.prompt_template.__class__,
        "__or__",
        lambda self, other: MagicMock(invoke=lambda x: mock_llm_response),
    )

    response = pipeline.answer("Sample question")
    assert len(response.sources) == 1
    assert response.sources[0].content == "Identical snippet content"


def test_rag_pipeline_empty_retrieval_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify RAGPipeline returns fallback response when no relevant chunks are found."""
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "mock-test-key")
    monkeypatch.setattr("app.rag.pipeline.ChatGoogleGenerativeAI", lambda **kwargs: MagicMock())

    mock_retriever = MagicMock(spec=DocumentRetriever)
    mock_retriever.retrieve_with_scores.return_value = []

    pipeline = RAGPipeline(retriever=mock_retriever)
    response = pipeline.answer("Unknown topic question")

    assert "could not find any relevant information" in response.answer
    assert response.sources == []
