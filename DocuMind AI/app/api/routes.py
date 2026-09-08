"""FastAPI API routes module for DocuMind AI.

This module defines the REST API router exposing health checks, RAG question answering,
and structured document information extraction endpoints.
"""

from functools import lru_cache
from pathlib import Path
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.core.logger import get_logger
from app.models.schemas import (
    ChatRequest,
    ChatResponse,
    DocumentMetadata,
    PageContent,
    ProcessedDocument,
)
from app.rag.pipeline import RAGPipeline
from app.services.extraction_service import (
    ExtractionService,
    StructuredExtractionResult,
)

logger = get_logger(__name__)

router = APIRouter()


# =====================================================================
# Request & Response Schemas
# =====================================================================

class HealthResponse(BaseModel):
    """Health check response schema."""

    status: str = Field(default="ok", description="Operational status of the API")


class ExtractionRequest(BaseModel):
    """Request payload for structured document information extraction."""

    document_text: str = Field(
        ...,
        min_length=1,
        description="Text content of the document to extract structured fields from",
    )
    document_id: str = Field(
        ...,
        min_length=1,
        description="Unique identifier for the document",
    )
    filename: str = Field(
        ...,
        min_length=1,
        description="Original name of the document file",
    )
    document_type: str | None = Field(
        default=None,
        description="Optional category or classification hint (e.g., invoice, contract, form)",
    )


# =====================================================================
# Dependency Providers
# =====================================================================

@lru_cache
def get_rag_pipeline() -> RAGPipeline:
    """Return a cached RAGPipeline service instance."""
    return RAGPipeline()


@lru_cache
def get_extraction_service() -> ExtractionService:
    """Return a cached ExtractionService instance."""
    return ExtractionService()


# =====================================================================
# API Endpoints
# =====================================================================

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    tags=["System"],
)
async def health_check() -> HealthResponse:
    """Check the health status of the DocuMind AI system."""
    logger.info("Handling health check request.")
    return HealthResponse(status="ok")


@router.post(
    "/ask",
    response_model=ChatResponse,
    summary="RAG Question Answering",
    tags=["RAG"],
)
async def ask_question(request: ChatRequest) -> ChatResponse:
    """Answer factual user questions using grounded document retrieval and citation synthesis."""
    logger.info("Received RAG /ask query: '%s'", request.question[:80])

    rag_pipeline = get_rag_pipeline()

    try:
        response = rag_pipeline.answer(
            question=request.question,
            top_k=request.top_k,
            document_ids=request.document_ids or None,
        )
        logger.info("Successfully generated answer for /ask query.")
        return response
    except FileNotFoundError as exc:
        logger.warning("Vector store index missing during /ask query: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Vector store index not found: {exc}",
        ) from exc
    except ValueError as exc:
        logger.warning("Validation error during /ask query: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.error("Unexpected error during RAG /ask execution: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate answer from documents.",
        ) from exc


@router.post(
    "/extract",
    response_model=StructuredExtractionResult,
    summary="Structured Document Extraction",
    tags=["Extraction"],
)
async def extract_document(request: ExtractionRequest) -> StructuredExtractionResult:
    """Extract structured key-value entities, classifications, and executive summaries from text."""
    logger.info(
        "Received /extract request for document '%s' (id: %s, type_hint: %s)",
        request.filename,
        request.document_id,
        request.document_type,
    )

    extraction_service = get_extraction_service()

    try:
        file_suffix = Path(request.filename).suffix.lower().lstrip(".")
        inferred_type = file_suffix if file_suffix else "txt"

        metadata = DocumentMetadata(
            document_id=request.document_id.strip(),
            filename=request.filename.strip(),
            file_path=request.filename.strip(),
            file_type=inferred_type,
            source=request.document_type.strip() if request.document_type else None,
            total_pages=1,
            ocr_used=False,
        )

        processed_doc = ProcessedDocument(
            metadata=metadata,
            pages=[PageContent(page_number=1, text=request.document_text.strip())],
            full_text=request.document_text.strip(),
        )

        result = extraction_service.extract_from_document(
            document=processed_doc,
            document_type=request.document_type,
        )

        logger.info(
            "Successfully extracted %d field(s) from document '%s'.",
            len(result.fields),
            request.filename,
        )
        return result
    except FileNotFoundError as exc:
        logger.warning("Resource not found during /extract execution: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        logger.warning("Validation error during /extract execution: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.error("Unexpected error during structured extraction for '%s': %s", request.filename, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to extract structured information from document.",
        ) from exc
