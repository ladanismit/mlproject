"""LangChain agent tools for DocuMind AI.

This module exposes core document intelligence capabilities (grounded RAG question
answering and structured entity extraction) as standard LangChain tools for autonomous
agents to invoke during multi-step document analysis workflows.
"""

from pathlib import Path
import re
from langchain_core.tools import tool

from app.core.logger import get_logger
from app.models.schemas import DocumentMetadata, PageContent, ProcessedDocument
from app.rag.pipeline import RAGPipeline
from app.services.extraction_service import ExtractionService

logger = get_logger(__name__)

# Module-level reusable service singletons
rag_pipeline = RAGPipeline()
extraction_service = ExtractionService()


def _parse_document_text(document_text: str) -> list[PageContent]:
    """Parse text containing optional [Page N] demarcations into structured PageContent objects.

    Args:
        document_text: Raw or page-demarcated document text string.

    Returns:
        list[PageContent]: List of page content instances in chronological order.

    Raises:
        ValueError: If the text is empty or page markers are malformed.
    """
    if not isinstance(document_text, str) or not document_text.strip():
        raise ValueError("Document text must be a non-empty string.")

    # Match lines that define a page boundary: e.g., "[Page 1]" or "[Page 2]"
    page_marker_pattern = re.compile(r"^\[Page\s+(\d+)\]\s*$", re.MULTILINE | re.IGNORECASE)
    matches = list(page_marker_pattern.finditer(document_text))

    if not matches:
        # No explicit page markers found; treat the whole text as Page 1
        return [
            PageContent(
                page_number=1,
                text=document_text.strip(),
                metadata={"parsed_page": 1},
            )
        ]

    pages: list[PageContent] = []
    for i, match in enumerate(matches):
        page_num = int(match.group(1))
        if page_num < 1:
            raise ValueError(
                f"Invalid page number {page_num}. Page numbers must be >= 1."
            )
        start_idx = match.end()
        end_idx = matches[i + 1].start() if i + 1 < len(matches) else len(document_text)

        page_body = document_text[start_idx:end_idx].strip()
        pages.append(
            PageContent(
                page_number=page_num,
                text=page_body,
                metadata={"parsed_page": page_num},
            )
        )

    if not pages:
        raise ValueError("Failed to parse pages from the provided document text.")

    return pages


def _build_processed_document(
    document_id: str,
    filename: str,
    document_text: str,
    document_type: str | None = None,
) -> ProcessedDocument:
    """Construct a complete ProcessedDocument schema from raw text inputs.

    Args:
        document_id: Unique identifier for the document.
        filename: Original file name.
        document_text: Text content of the document with optional page tags.
        document_type: Optional classification type or category hint.

    Returns:
        ProcessedDocument: Standardized document schema ready for extraction.

    Raises:
        ValueError: If required metadata or text fields are missing or empty.
    """
    if not isinstance(document_id, str) or not document_id.strip():
        raise ValueError("document_id must be a non-empty string.")

    if not isinstance(filename, str) or not filename.strip():
        raise ValueError("filename must be a non-empty string.")

    pages = _parse_document_text(document_text)
    full_text = "\n\n".join(p.text for p in pages if p.text.strip())

    file_suffix = Path(filename).suffix.lower().lstrip(".")
    inferred_type = file_suffix if file_suffix else "txt"

    metadata = DocumentMetadata(
        document_id=document_id.strip(),
        filename=filename.strip(),
        file_path=filename.strip(),
        file_type=inferred_type,
        source=document_type.strip() if document_type else None,
        total_pages=len(pages),
        ocr_used=False,
    )

    return ProcessedDocument(
        metadata=metadata,
        pages=pages,
        full_text=full_text,
    )


@tool
def document_question_answering(
    question: str,
    document_ids: list[str] | None = None,
    top_k: int = 4,
) -> str:
    """Answers factual questions about indexed document collections using grounded semantic search.

    Use this tool when:
    - The user asks specific questions about content inside indexed documents (e.g. contracts, invoices, forms).
    - You need to locate exact figures, dates, parties, terms, clauses, or facts grounded in document context.
    - You require verified page citations and source snippets to back up an answer.

    Do NOT use this tool when:
    - You need complete structured JSON entity extraction for an entire document (use extract_document_fields instead).
    - The user is asking generic conversational questions unrelated to document collections.

    Args:
        question: The clear natural-language query to retrieve and answer.
        document_ids: Optional list of target document IDs to restrict retrieval scope.
        top_k: Number of relevant document context chunks to retrieve (must be >= 1, default 4).

    Returns:
        str: A structured text response containing the grounded answer and detailed page-level citations.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must be a non-empty string.")

    if not isinstance(top_k, int) or top_k < 1:
        raise ValueError(f"top_k must be an integer >= 1, got {top_k}")

    logger.info(
        "Invoking document_question_answering tool (top_k=%d, doc_filter=%s)",
        top_k,
        bool(document_ids),
    )

    try:
        response = rag_pipeline.answer(
            question=question.strip(),
            top_k=top_k,
            document_ids=document_ids,
        )

        formatted_output = [f"Answer:\n{response.answer}"]

        if response.sources:
            formatted_output.append("\nSources:")
            for src in response.sources:
                page_str = f", Page {src.page_number}" if src.page_number is not None else ""
                formatted_output.append(f"- {src.filename}{page_str}\n  \"{src.content}\"")
        else:
            formatted_output.append("\nSources: None found.")

        logger.info("document_question_answering completed successfully.")
        return "\n".join(formatted_output)

    except (ValueError, FileNotFoundError):
        raise
    except Exception as exc:
        logger.error("Error executing document_question_answering tool: %s", exc)
        raise RuntimeError(f"RAG question answering tool failed: {exc}") from exc


@tool
def extract_document_fields(
    document_text: str,
    document_id: str,
    filename: str,
    document_type: str | None = None,
) -> str:
    """Extracts structured key-value entities, classifications, and summaries from a document.

    Use this tool when:
    - You need to extract structured fields and metadata from an invoice, contract, form, or receipt.
    - You need to identify invoice numbers, dates, line items, totals, parties, governing laws, or survey numbers.
    - You need a machine-readable JSON representation of the document's structured attributes.

    Do NOT use this tool when:
    - You only want to ask a targeted question or find a single fact from indexed documents (use document_question_answering instead).

    Args:
        document_text: Complete text of the document, optionally with '[Page N]' headers.
        document_id: Unique identifier for the document.
        filename: Original name of the document file.
        document_type: Optional category hint (e.g. 'invoice', 'contract', 'form', 'receipt').

    Returns:
        str: JSON string containing document classification, extracted fields with confidence & source pages, and summary.
    """
    if not isinstance(document_text, str) or not document_text.strip():
        raise ValueError("document_text must be a non-empty string.")

    if not isinstance(document_id, str) or not document_id.strip():
        raise ValueError("document_id must be a non-empty string.")

    if not isinstance(filename, str) or not filename.strip():
        raise ValueError("filename must be a non-empty string.")

    logger.info(
        "Invoking extract_document_fields tool for '%s' (id: %s, type_hint: %s)",
        filename,
        document_id,
        document_type,
    )

    try:
        processed_doc = _build_processed_document(
            document_id=document_id,
            filename=filename,
            document_text=document_text,
            document_type=document_type,
        )

        extraction_result = extraction_service.extract_from_document(
            document=processed_doc,
            document_type=document_type,
        )

        logger.info(
            "extract_document_fields completed for '%s' (%d fields extracted).",
            filename,
            len(extraction_result.fields),
        )

        return extraction_result.model_dump_json(indent=2)

    except ValueError:
        raise
    except Exception as exc:
        logger.error("Error executing extract_document_fields tool for '%s': %s", filename, exc)
        raise RuntimeError(f"Structured extraction tool failed for '{filename}': {exc}") from exc


__all__ = [
    "document_question_answering",
    "extract_document_fields",
]
