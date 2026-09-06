"""Pydantic schemas and data models for DocuMind AI.

This module defines the structured data contracts used across document ingestion,
OCR, text extraction, chunking, RAG retrieval, agent operations, document comparison,
and FastAPI request/response endpoints.
"""

from typing import Any
from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    """Metadata describing an uploaded or ingested document."""

    document_id: str = Field(..., description="Unique identifier for the document")
    filename: str = Field(..., min_length=1, description="Original name of the uploaded file")
    file_path: str = Field(..., description="Filesystem or storage path of the document")
    file_type: str = Field(..., description="Document extension/MIME type (e.g. pdf, png, jpg)")
    source: str | None = Field(default=None, description="Document category or origin (e.g. invoice, contract, form)")
    total_pages: int = Field(default=1, ge=1, description="Total number of pages in the document")
    ocr_used: bool = Field(default=False, description="Whether OCR was performed during text extraction")


class PageContent(BaseModel):
    """Extracted text and metadata for a single page of a document."""

    page_number: int = Field(..., ge=1, description="Page index (1-based)")
    text: str = Field(..., description="Extracted textual content of the page")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional page-level attributes")


class ProcessedDocument(BaseModel):
    """Aggregated output of the document ingestion and extraction pipeline."""

    metadata: DocumentMetadata = Field(..., description="Document metadata")
    pages: list[PageContent] = Field(default_factory=list, description="List of extracted individual pages")
    full_text: str = Field(..., description="Consolidated plain text across all document pages")


class ExtractedField(BaseModel):
    """A single structured entity or key-value field extracted from a document."""

    field_name: str = Field(..., min_length=1, description="Name of the extracted entity or field")
    value: Any = Field(default=None, description="Extracted value (string, numeric, date, etc.)")
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Extraction confidence score between 0.0 and 1.0",
    )
    source_page: int | None = Field(
        default=None,
        ge=1,
        description="Page number where the field was located",
    )


class SourceCitation(BaseModel):
    """Attribution citation linking generated answers to source document context."""

    document_id: str = Field(..., description="ID of the referenced document")
    filename: str = Field(..., description="Filename of the referenced document")
    page_number: int | None = Field(
        default=None,
        ge=1,
        description="Specific page number providing the context",
    )
    content: str = Field(..., description="Relevant text snippet or chunk used as evidence")


class ChatRequest(BaseModel):
    """Payload for conversational Q&A and RAG search queries."""

    question: str = Field(..., min_length=1, description="User question or prompt")
    document_ids: list[str] = Field(
        default_factory=list,
        description="Optional list of target document IDs to restrict retrieval scope",
    )
    top_k: int = Field(
        default=4,
        ge=1,
        le=50,
        description="Number of most relevant context chunks to retrieve",
    )


class ChatResponse(BaseModel):
    """Response returned by RAG and agent question-answering workflows."""

    answer: str = Field(..., description="Generated answer from LLM/agent")
    sources: list[SourceCitation] = Field(
        default_factory=list,
        description="List of document citations supporting the generated answer",
    )


class ComparisonField(BaseModel):
    """Comparison evaluation of a specific field between two documents."""

    field_name: str = Field(..., min_length=1, description="Name of the field being compared")
    document_a_value: Any = Field(default=None, description="Value extracted from Document A")
    document_b_value: Any = Field(default=None, description="Value extracted from Document B")
    match: bool = Field(..., description="Whether both values match semantically or strictly")
    details: str | None = Field(default=None, description="Explanation or discrepancy notes")


class ComparisonResult(BaseModel):
    """Consolidated report comparing two documents for discrepancies or alignment."""

    document_a: str = Field(..., description="Identifier or filename for Document A")
    document_b: str = Field(..., description="Identifier or filename for Document B")
    fields: list[ComparisonField] = Field(
        default_factory=list,
        description="Detailed field-by-field comparison list",
    )
    has_mismatches: bool = Field(
        default=False,
        description="Flag indicating if any discrepancies were detected",
    )
    summary: str = Field(
        default="",
        description="Executive summary of differences or alignment between documents",
    )
