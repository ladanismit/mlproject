"""Structured document information extraction service for DocuMind AI.

This module provides generic, page-aware entity and key-value extraction from
processed documents (invoices, contracts, forms, and general business documents)
using LLM structured outputs with strict schema validation.
"""

from typing import Any
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from app.core.config import settings
from app.core.logger import get_logger
from app.models.schemas import ExtractedField, ProcessedDocument

logger = get_logger(__name__)

SUPPORTED_LLM_PROVIDERS = {"openai"}
MAX_EXTRACTION_TEXT_LENGTH = 30000

EXTRACTION_SYSTEM_PROMPT = """You are DocuMind AI, a structured document information extraction engine.

Your task is to analyze the provided document text and extract all meaningful entities, key-value fields, and an executive summary into a strict structured schema.

Extraction Rules and Constraints:
1. Explicit Grounding: Extract ONLY values explicitly stated in the text. NEVER infer, guess, extrapolate, or invent missing information.
2. Missing Fields: If a specific field or value is not present in the text, omit it or set its value to null.
3. Fact Precision: Preserve all numbers, identifiers, reference codes, invoice numbers, dates, currency amounts, percentages, tax rates, party names, addresses, survey numbers, and legal terms exactly as written.
4. Granular Disambiguation: Carefully distinguish between semantically similar roles and concepts:
   - Invoice Number vs. Purchase Order Number vs. Customer ID
   - Seller / Vendor vs. Buyer / Customer
   - Contract Execution Date vs. Effective Date vs. Termination Date
   - Subtotal vs. Tax vs. Total Payable Amount
5. Confidence Scoring: Assign each extracted field a confidence score between 0.0 and 1.0 representing how clearly and unambiguously the value is directly supported by the text (1.0 = unambiguous explicit mention).
6. Page Attribution: Associate each extracted field with its exact source page (1-based index) as indicated by the [Page N] headers in the text. If the source page cannot be determined, use null.
7. Document Type Classification: Identify the specific document category (e.g. 'invoice', 'receipt', 'contract', 'employment_agreement', 'real_estate_deed', 'application_form', or 'financial_statement'). If a document type hint is provided, consider it, but let the text decide. If indeterminate, return null.
8. Executive Summary: Provide a concise, factual summary (2-4 sentences) outlining the core subject, parties, key numbers, and purpose based strictly on the text."""

EXTRACTION_HUMAN_PROMPT = """Document Information:
- Document ID: {document_id}
- Filename: {filename}
- Document Type Hint: {document_type_hint}

Document Content (Page by Page):
{document_content}

Extract all structured fields, identify the document type, and produce an executive summary following the schema."""


class StructuredExtractionResult(BaseModel):
    """Structured extraction output schema containing identified fields, classification, and summary."""

    document_id: str = Field(..., description="Unique ID of the analyzed document")
    filename: str = Field(..., description="Original filename of the analyzed document")
    document_type: str | None = Field(
        default=None,
        description="Classified document category (e.g., invoice, contract, form, receipt)",
    )
    fields: list[ExtractedField] = Field(
        default_factory=list,
        description="List of structured key-value entities extracted from the text",
    )
    summary: str = Field(
        ...,
        description="Concise factual summary of the document contents and key figures",
    )
    raw_text_used: bool = Field(
        default=True,
        description="Indicates whether raw document text was utilized for the extraction",
    )


class ExtractionService:
    """Extracts structured entities and summaries from processed documents using LLM structured outputs."""

    def __init__(
        self,
        provider: str | None = None,
        model_name: str | None = None,
        temperature: float | None = None,
    ) -> None:
        """Initialize the structured extraction service.

        Args:
            provider: LLM provider name (defaults to settings.LLM_PROVIDER).
            model_name: Name of the LLM model (defaults to settings.LLM_MODEL).
            temperature: Sampling temperature (defaults to 0.0 for deterministic extraction).

        Raises:
            ValueError: If the provider is unsupported or required API keys are missing.
        """
        self.provider = (provider or settings.LLM_PROVIDER).lower()
        self.model_name = model_name or settings.LLM_MODEL
        self.temperature = (
            temperature if temperature is not None else 0.0
        )

        if self.provider not in SUPPORTED_LLM_PROVIDERS:
            raise ValueError(
                f"Unsupported LLM provider '{self.provider}'. "
                f"Supported providers: {sorted(SUPPORTED_LLM_PROVIDERS)}"
            )

        self.llm = self._initialize_llm()
        self.structured_llm = self.llm.with_structured_output(StructuredExtractionResult)
        self.prompt_template = self._build_prompt()

        logger.info(
            "ExtractionService initialized (provider=%s, model=%s, temperature=%.2f)",
            self.provider,
            self.model_name,
            self.temperature,
        )

    def _initialize_llm(self) -> ChatOpenAI:
        """Configure and instantiate the underlying chat LLM.

        Returns:
            ChatOpenAI: Initialized LangChain chat model.

        Raises:
            ValueError: If API credentials are not set.
        """
        if self.provider == "openai":
            api_key = settings.OPENAI_API_KEY
            if not api_key or not api_key.strip():
                raise ValueError(
                    "OPENAI_API_KEY is not configured. Please configure OPENAI_API_KEY "
                    "in your environment or .env file."
                )

            return ChatOpenAI(
                model=self.model_name,
                temperature=self.temperature,
                api_key=api_key,
            )

        raise ValueError(f"Unhandled LLM provider: {self.provider}")

    def _build_prompt(self) -> ChatPromptTemplate:
        """Construct the prompt template for structured extraction.

        Returns:
            ChatPromptTemplate: Formatted chat prompt template.
        """
        return ChatPromptTemplate.from_messages(
            [
                ("system", EXTRACTION_SYSTEM_PROMPT),
                ("human", EXTRACTION_HUMAN_PROMPT),
            ]
        )

    def _prepare_document_text(self, document: ProcessedDocument) -> str:
        """Format document text page-by-page and apply length limits if necessary.

        Args:
            document: ProcessedDocument containing page contents.

        Returns:
            str: Page-delimited document text prepared for LLM analysis.

        Raises:
            ValueError: If the document does not contain any usable text.
        """
        page_sections: list[str] = []

        for page in document.pages:
            text = page.text.strip() if page.text else ""
            if text:
                page_sections.append(f"[Page {page.page_number}]\n{text}")

        combined_text = "\n\n".join(page_sections).strip()

        if not combined_text:
            raise ValueError(
                f"Document '{document.metadata.filename}' contains no readable text for extraction."
            )

        # Apply maximum length limit while preserving header and footer sections
        if len(combined_text) > MAX_EXTRACTION_TEXT_LENGTH:
            head_len = MAX_EXTRACTION_TEXT_LENGTH // 2
            tail_len = MAX_EXTRACTION_TEXT_LENGTH // 2
            omitted_count = len(combined_text) - (head_len + tail_len)

            logger.warning(
                "Document '%s' exceeds max text length (%d chars). Truncating middle section (%d chars omitted).",
                document.metadata.filename,
                len(combined_text),
                omitted_count,
            )

            combined_text = (
                combined_text[:head_len]
                + f"\n\n[... DOCUMENT TEXT TRUNCATED: {omitted_count} CHARACTERS OMITTED ...]\n\n"
                + combined_text[-tail_len:]
            )

        return combined_text

    def _validate_result(
        self,
        result: Any,
        expected_document_id: str,
        expected_filename: str,
    ) -> StructuredExtractionResult:
        """Validate and harmonize the structured result returned by the LLM.

        Args:
            result: LLM response parsed into StructuredExtractionResult or dict.
            expected_document_id: ID of the input document.
            expected_filename: Filename of the input document.

        Returns:
            StructuredExtractionResult: Verified and consistent extraction result.

        Raises:
            ValueError: If the result cannot be validated against the schema.
        """
        if isinstance(result, dict):
            extraction_result = StructuredExtractionResult(**result)
        elif isinstance(result, StructuredExtractionResult):
            extraction_result = result
        else:
            raise ValueError(f"Unexpected extraction result format: {type(result)}")

        # Enforce exact document ID and filename alignment
        extraction_result.document_id = expected_document_id
        extraction_result.filename = expected_filename
        extraction_result.raw_text_used = True

        # Verify field confidence bounds and page constraints
        for field in extraction_result.fields:
            if not 0.0 <= field.confidence <= 1.0:
                raise ValueError(
                    f"Invalid confidence for field '{field.field_name}': "
                    f"{field.confidence}. Expected a value between 0.0 and 1.0."
                )

        return extraction_result

    def extract_from_document(
        self,
        document: ProcessedDocument,
        document_type: str | None = None,
    ) -> StructuredExtractionResult:
        """Extract structured entities, key-value data, and a summary from a ProcessedDocument.

        Args:
            document: ProcessedDocument instance with extracted text.
            document_type: Optional hint regarding the document category.

        Returns:
            StructuredExtractionResult: Pydantic model populated with extracted fields and metadata.

        Raises:
            ValueError: If input is invalid or contains no readable text.
            RuntimeError: If the LLM call or extraction pipeline fails.
        """
        if not isinstance(document, ProcessedDocument):
            raise ValueError("Input 'document' must be an instance of ProcessedDocument")

        doc_id = document.metadata.document_id
        filename = document.metadata.filename
        type_hint_str = document_type or document.metadata.source or "Not specified (determine from text)"

        logger.info(
            "Starting structured extraction for document '%s' (id: %s, pages: %d, type_hint: %s)",
            filename,
            doc_id,
            len(document.pages),
            type_hint_str,
        )

        try:
            prepared_text = self._prepare_document_text(document)

            extraction_chain = self.prompt_template | self.structured_llm

            raw_result = extraction_chain.invoke(
                {
                    "document_id": doc_id,
                    "filename": filename,
                    "document_type_hint": type_hint_str,
                    "document_content": prepared_text,
                }
            )

            validated_result = self._validate_result(
                result=raw_result,
                expected_document_id=doc_id,
                expected_filename=filename,
            )

            logger.info(
                "Structured extraction completed for '%s': detected_type='%s', %d fields extracted.",
                filename,
                validated_result.document_type,
                len(validated_result.fields),
            )

            return validated_result

        except ValueError:
            raise
        except Exception as exc:
            logger.error("Structured extraction failed for document '%s': %s", filename, exc)
            raise RuntimeError(f"Structured extraction failed for '{filename}': {exc}") from exc
