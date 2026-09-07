"""Document comparison and inconsistency detection service for DocuMind AI.

This module provides deterministic field-by-field comparison and discrepancy analysis
between two documents (e.g., invoices, contracts, forms, or agreements) by utilizing
structured extraction and normalized semantic attribute matching.
"""

from datetime import datetime
from pathlib import Path
import re
from typing import Any

from app.core.logger import get_logger
from app.models.schemas import (
    ComparisonField,
    ComparisonResult,
    DocumentMetadata,
    ExtractedField,
    PageContent,
    ProcessedDocument,
)
from app.services.extraction_service import ExtractionService

logger = get_logger(__name__)

# Canonical field name synonyms for standard document types
SYNONYM_MAPPING: dict[str, str] = {
    # Invoice numbers & IDs
    "invoice_no": "invoice_number",
    "invoice_num": "invoice_number",
    "inv_no": "invoice_number",
    "inv_num": "invoice_number",
    "invoice_#": "invoice_number",
    "bill_no": "invoice_number",
    "bill_number": "invoice_number",
    # Dates
    "date": "invoice_date",
    "bill_date": "invoice_date",
    "dated": "invoice_date",
    "execution_date": "contract_date",
    "agreement_date": "contract_date",
    "effective_start_date": "effective_date",
    "start_date": "effective_date",
    "end_date": "termination_date",
    "expiry_date": "termination_date",
    # Parties
    "vendor": "vendor_name",
    "supplier": "vendor_name",
    "seller": "vendor_name",
    "merchant": "vendor_name",
    "service_provider": "vendor_name",
    "buyer": "buyer_name",
    "customer": "buyer_name",
    "client": "buyer_name",
    "purchaser": "buyer_name",
    # Financial amounts
    "total": "total_amount",
    "grand_total": "total_amount",
    "total_payable": "total_amount",
    "amount_due": "total_amount",
    "final_amount": "total_amount",
    "sub_total": "subtotal",
    "tax_amount": "tax",
    "gst": "tax",
    "vat": "tax",
    "sales_tax": "tax",
    "contract_amount": "contract_value",
    "consideration": "consideration_amount",
    # Legal & property
    "jurisdiction": "governing_law",
    "governing_jurisdiction": "governing_law",
    "survey_no": "survey_number",
    "plot_no": "survey_number",
}


class ComparisonService:
    """Performs deterministic comparison of structured entities between two documents."""

    def __init__(self, extraction_service: ExtractionService | None = None) -> None:
        """Initialize the ComparisonService with an ExtractionService instance.

        Args:
            extraction_service: Optional ExtractionService instance. Defaults to a new instance.
        """
        self.extraction_service = extraction_service or ExtractionService()
        logger.info("ComparisonService initialized.")

    def _normalize_field_name(self, field_name: str) -> str:
        """Normalize a field name string into a canonical snake_case identifier.

        Args:
            field_name: Raw field name (e.g., 'Invoice No.', 'Vendor Name').

        Returns:
            str: Canonical normalized field name.
        """
        if not field_name:
            return ""

        # Convert to lowercase and trim
        cleaned = field_name.strip().lower()

        # Replace non-alphanumeric characters (except underscores and hashes) with spaces
        cleaned = re.sub(r"[^\w\s#]", " ", cleaned)

        # Collapse whitespace into single underscores
        cleaned = re.sub(r"[\s_]+", "_", cleaned).strip("_")

        # Map through canonical synonyms if available
        return SYNONYM_MAPPING.get(cleaned, cleaned)

    def _normalize_value(self, value: Any) -> Any:
        """Normalize field values to enable reliable equality comparison across formatting variations.

        Args:
            value: Raw field value (string, numeric, date, etc.).

        Returns:
            Any: Normalized value representation.
        """
        if value is None:
            return None

        # Handle boolean or direct numeric types
        if isinstance(value, (int, float, bool)):
            return value

        val_str = str(value).strip()
        if not val_str:
            return ""

        # 1. Attempt numeric/currency normalization
        # Remove currency symbols (₹, $, €, £, Rs, INR, USD) and commas
        currency_cleaned = re.sub(
            r"(?i)^(₹|\$|€|£|rs\.?|inr|usd)\s*|\s*(inr|usd)$", "", val_str
        ).strip()
        currency_cleaned = currency_cleaned.replace(",", "")

        # Try parsing as float or int
        try:
            num = float(currency_cleaned)
            # If it's a whole integer, convert to int for consistent comparison
            return int(num) if num.is_integer() else round(num, 4)
        except ValueError:
            pass

        # 2. Attempt Date normalization (e.g. YYYY-MM-DD, DD/MM/YYYY, MM/DD/YYYY, DD-MM-YYYY)
        date_formats = [
            "%Y-%m-%d",
            "%d/%m/%Y",
            "%m/%d/%Y",
            "%d-%m-%Y",
            "%d %B %Y",
            "%d %b %Y",
            "%B %d, %Y",
            "%b %d, %Y",
        ]
        for fmt in date_formats:
            try:
                parsed_dt = datetime.strptime(val_str, fmt)
                return parsed_dt.strftime("%Y-%m-%d")
            except ValueError:
                continue

        # 3. Text normalization: lowercased, whitespace collapsed, standard corporate suffixes
        norm_text = val_str.lower()
        # Normalize punctuation in corporate abbreviations (e.g., pvt. ltd. -> pvt ltd)
        norm_text = norm_text.replace(".", "").replace(",", "")
        norm_text = re.sub(r"\s+", " ", norm_text).strip()

        return norm_text

    def _build_field_map(
        self, fields: list[ExtractedField]
    ) -> dict[str, ExtractedField]:
        """Index a list of extracted fields by their canonical normalized field names.

        Args:
            fields: List of ExtractedField objects.

        Returns:
            dict[str, ExtractedField]: Mapping of normalized field name to ExtractedField.
        """
        field_map: dict[str, ExtractedField] = {}
        for field in fields:
            canonical_name = self._normalize_field_name(field.field_name)
            if canonical_name:
                field_map[canonical_name] = field
        return field_map

    def _compare_fields(
        self,
        map_a: dict[str, ExtractedField],
        map_b: dict[str, ExtractedField],
    ) -> list[ComparisonField]:
        """Compare two field maps and produce a detailed list of ComparisonField results.

        Args:
            map_a: Field map for Document A.
            map_b: Field map for Document B.

        Returns:
            list[ComparisonField]: Complete list of comparison evaluations across all fields.
        """
        all_field_names = sorted(set(map_a.keys()) | set(map_b.keys()))
        comparison_results: list[ComparisonField] = []

        for name in all_field_names:
            field_a = map_a.get(name)
            field_b = map_b.get(name)

            val_a = field_a.value if field_a is not None else None
            val_b = field_b.value if field_b is not None else None

            # Case 1: Present in both documents
            if field_a is not None and field_b is not None:
                norm_a = self._normalize_value(val_a)
                norm_b = self._normalize_value(val_b)
                is_match = norm_a == norm_b

                if is_match:
                    details = "Values match."
                else:
                    details = (
                        f"Values differ: Document A contains '{val_a}', "
                        f"while Document B contains '{val_b}'."
                    )

            # Case 2: Present in Document A only
            elif field_a is not None and field_b is None:
                is_match = False
                details = "Field is missing from Document B."

            # Case 3: Present in Document B only
            else:
                is_match = False
                details = "Field is missing from Document A."

            comparison_results.append(
                ComparisonField(
                    field_name=name,
                    document_a_value=val_a,
                    document_b_value=val_b,
                    match=is_match,
                    details=details,
                )
            )

        return comparison_results

    def _build_summary(self, comparison_fields: list[ComparisonField]) -> str:
        """Construct a deterministic comparison summary.

        Args:
            comparison_fields: List of evaluated ComparisonField results.

        Returns:
            str: Concise summary of matches and mismatches.
        """
        total_fields = len(comparison_fields)
        if total_fields == 0:
            return "No structured fields were available for comparison."

        matches = [f for f in comparison_fields if f.match]
        mismatches = [f for f in comparison_fields if not f.match]
        match_count = len(matches)
        mismatch_count = len(mismatches)

        if mismatch_count == 0:
            return f"All {total_fields} comparable fields match between the two documents."

        mismatched_names = [f.field_name for f in mismatches[:4]]
        mismatch_preview = ", ".join(mismatched_names)
        if mismatch_count > 4:
            mismatch_preview += f", and {mismatch_count - 4} more"

        return (
            f"Compared {total_fields} fields. {match_count} fields match and "
            f"{mismatch_count} fields contain mismatches ({mismatch_preview})."
        )

    def _build_processed_doc(
        self,
        document_id: str,
        filename: str,
        document_text: str,
        document_type: str | None,
    ) -> ProcessedDocument:
        """Create a minimal ProcessedDocument representation for extraction.

        Args:
            document_id: Unique document ID.
            filename: Original filename.
            document_text: Text content of the document.
            document_type: Optional document category hint.

        Returns:
            ProcessedDocument: Constructed document schema.
        """
        file_suffix = Path(filename).suffix.lower().lstrip(".")
        metadata = DocumentMetadata(
            document_id=document_id,
            filename=filename,
            file_path=filename,
            file_type=file_suffix if file_suffix else "txt",
            source=document_type,
            total_pages=1,
            ocr_used=False,
        )
        pages = [PageContent(page_number=1, text=document_text)]
        return ProcessedDocument(
            metadata=metadata,
            pages=pages,
            full_text=document_text,
        )

    def compare_documents(
        self,
        document_a_text: str,
        document_b_text: str,
        document_a_id: str,
        document_b_id: str,
        document_a_filename: str,
        document_b_filename: str,
        document_a_type: str | None = None,
        document_b_type: str | None = None,
    ) -> ComparisonResult:
        """Compare two documents by extracting structured fields and detecting discrepancies.

        Args:
            document_a_text: Full text content of Document A.
            document_b_text: Full text content of Document B.
            document_a_id: Unique identifier for Document A.
            document_b_id: Unique identifier for Document B.
            document_a_filename: Original filename for Document A.
            document_b_filename: Original filename for Document B.
            document_a_type: Optional document type hint for Document A.
            document_b_type: Optional document type hint for Document B.

        Returns:
            ComparisonResult: Comprehensive field-by-field comparison result.

        Raises:
            ValueError: If inputs are empty or both documents share the same ID.
            RuntimeError: If extraction or comparison processing fails.
        """
        # 1. Input Validation
        if not isinstance(document_a_text, str) or not document_a_text.strip():
            raise ValueError("Document A text must be a non-empty string.")

        if not isinstance(document_b_text, str) or not document_b_text.strip():
            raise ValueError("Document B text must be a non-empty string.")

        if not isinstance(document_a_id, str) or not document_a_id.strip():
            raise ValueError("Document A ID must be a non-empty string.")

        if not isinstance(document_b_id, str) or not document_b_id.strip():
            raise ValueError("Document B ID must be a non-empty string.")

        if not isinstance(document_a_filename, str) or not document_a_filename.strip():
            raise ValueError("Document A filename must be a non-empty string.")

        if not isinstance(document_b_filename, str) or not document_b_filename.strip():
            raise ValueError("Document B filename must be a non-empty string.")

        if document_a_id.strip() == document_b_id.strip():
            raise ValueError("Document IDs must be different for comparison.")

        doc_a_id_clean = document_a_id.strip()
        doc_b_id_clean = document_b_id.strip()
        doc_a_fn_clean = document_a_filename.strip()
        doc_b_fn_clean = document_b_filename.strip()

        logger.info(
            "Starting document comparison (doc_a=%s, doc_b=%s)",
            doc_a_id_clean,
            doc_b_id_clean,
        )

        # 2. Extract structured fields from both documents
        try:
            doc_a = self._build_processed_doc(
                document_id=doc_a_id_clean,
                filename=doc_a_fn_clean,
                document_text=document_a_text.strip(),
                document_type=document_a_type,
            )
            doc_b = self._build_processed_doc(
                document_id=doc_b_id_clean,
                filename=doc_b_fn_clean,
                document_text=document_b_text.strip(),
                document_type=document_b_type,
            )

            result_a = self.extraction_service.extract_from_document(
                document=doc_a,
                document_type=document_a_type,
            )
            result_b = self.extraction_service.extract_from_document(
                document=doc_b,
                document_type=document_b_type,
            )
        except Exception as exc:
            logger.exception("Document extraction failed during comparison.")
            raise RuntimeError("Document comparison failed during field extraction.") from exc

        # 3. Normalize fields and execute deterministic comparison
        try:
            map_a = self._build_field_map(result_a.fields)
            map_b = self._build_field_map(result_b.fields)

            comparison_fields = self._compare_fields(map_a, map_b)
            has_mismatches = any(not field.match for field in comparison_fields)
            summary_text = self._build_summary(comparison_fields)

            logger.info(
                "Document comparison completed: %d fields evaluated, has_mismatches=%s",
                len(comparison_fields),
                has_mismatches,
            )

            return ComparisonResult(
                document_a=doc_a_fn_clean,
                document_b=doc_b_fn_clean,
                fields=comparison_fields,
                has_mismatches=has_mismatches,
                summary=summary_text,
            )
        except Exception as exc:
            logger.exception("Field comparison logic encountered an unexpected failure.")
            raise RuntimeError("Document comparison failed.") from exc


__all__ = ["ComparisonService"]
