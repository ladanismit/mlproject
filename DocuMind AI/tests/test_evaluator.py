"""Comprehensive offline unit tests for the DocuMind AI evaluation engine.

This test module validates question dataset loading, schema validation, answer extraction,
page citation detection, category-specific evaluation logic, error isolation, summary metric
calculations, report serialization, and citation compliance score accuracy.
"""

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any
import pytest

from app.models.schemas import ChatResponse, SourceCitation
from evaluation.evaluator import (
    REQUIRED_QUESTION_KEYS,
    UNCERTAINTY_PHRASES,
    VALID_CATEGORIES,
    VALID_DIFFICULTIES,
    DocumentEvaluator,
    EvaluationResult,
    EvaluationSummary,
    load_questions,
    validate_questions,
)


# =====================================================================
# Fixtures & Helper Utilities
# =====================================================================

def create_sample_question(
    qid: str = "Q001",
    category: str = "rag",
    question: str = "What is the invoice number?",
    document_type: str = "invoice",
    difficulty: str = "easy",
    requires_citation: bool = True,
    expected_behavior: str = "Extract the invoice number accurately from context.",
) -> dict[str, Any]:
    """Helper to generate a valid individual question dictionary."""
    return {
        "id": qid,
        "category": category,
        "question": question,
        "document_type": document_type,
        "difficulty": difficulty,
        "requires_citation": requires_citation,
        "expected_behavior": expected_behavior,
    }


@pytest.fixture
def sample_questions_file(tmp_path: Path) -> Path:
    """Fixture providing a temporary questions.json file with diverse questions."""
    questions = [
        create_sample_question(
            qid="Q001",
            category="rag",
            question="What is the invoice amount?",
            document_type="invoice",
            difficulty="easy",
            requires_citation=True,
        ),
        create_sample_question(
            qid="Q002",
            category="citation",
            question="On which page is the contract duration stated?",
            document_type="contract",
            difficulty="medium",
            requires_citation=True,
        ),
        create_sample_question(
            qid="Q003",
            category="missing_information",
            question="What is the bank account number?",
            document_type="invoice",
            difficulty="medium",
            requires_citation=False,
            expected_behavior="State that bank account details are not provided.",
        ),
        create_sample_question(
            qid="Q004",
            category="comparison",
            question="What differences exist between both invoices?",
            document_type="multi_document",
            difficulty="hard",
            requires_citation=True,
        ),
    ]
    file_path = tmp_path / "questions.json"
    file_path.write_text(json.dumps({"questions": questions}, indent=2), encoding="utf-8")
    return file_path


# =====================================================================
# 1. Question Loading Tests
# =====================================================================

def test_load_questions_success(tmp_path: Path) -> None:
    """Verify load_questions successfully parses and validates a valid JSON dataset."""
    raw_data = {
        "dataset_name": "Test Dataset",
        "questions": [
            create_sample_question("Q001"),
            create_sample_question("Q002", category="extraction", difficulty="medium"),
        ],
    }
    file_path = tmp_path / "valid_questions.json"
    file_path.write_text(json.dumps(raw_data), encoding="utf-8")

    loaded = load_questions(file_path)
    assert len(loaded) == 2
    assert loaded[0]["id"] == "Q001"
    assert loaded[1]["id"] == "Q002"


def test_load_questions_missing_file(tmp_path: Path) -> None:
    """Verify load_questions raises FileNotFoundError for nonexistent paths."""
    missing_path = tmp_path / "nonexistent.json"
    with pytest.raises(FileNotFoundError, match="Evaluation questions file not found"):
        load_questions(missing_path)


def test_load_questions_invalid_json_syntax(tmp_path: Path) -> None:
    """Verify load_questions raises ValueError when JSON syntax is corrupted."""
    bad_json_file = tmp_path / "corrupted.json"
    bad_json_file.write_text("{ unclosed json: invalid ", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid JSON format"):
        load_questions(bad_json_file)


def test_load_questions_root_not_dict(tmp_path: Path) -> None:
    """Verify load_questions raises ValueError when JSON root is a list or scalar."""
    list_file = tmp_path / "list_root.json"
    list_file.write_text(json.dumps([{"id": "Q001"}]), encoding="utf-8")

    with pytest.raises(ValueError, match="Questions dataset root must be a JSON object"):
        load_questions(list_file)


def test_load_questions_missing_questions_array(tmp_path: Path) -> None:
    """Verify load_questions raises ValueError when 'questions' key is missing or not a list."""
    no_arr_file = tmp_path / "no_array.json"
    no_arr_file.write_text(json.dumps({"dataset_name": "Demo"}), encoding="utf-8")

    with pytest.raises(ValueError, match="Dataset JSON must contain a top-level 'questions' list"):
        load_questions(no_arr_file)


# =====================================================================
# 2. Question Schema Validation Tests
# =====================================================================

def test_validate_questions_valid_dataset() -> None:
    """Verify validate_questions passes without error on correctly formatted questions."""
    valid_items = [
        create_sample_question("Q001", category="rag", difficulty="easy"),
        create_sample_question("Q002", category="comparison", difficulty="hard"),
    ]
    # Should complete with no exception
    validate_questions(valid_items)


def test_validate_questions_rejects_non_dict_element() -> None:
    """Verify validate_questions rejects non-dictionary question items."""
    with pytest.raises(ValueError, match="Question at index 0 must be a JSON object"):
        validate_questions(["not-a-dict"])  # type: ignore[list-item]


@pytest.mark.parametrize("missing_key", list(REQUIRED_QUESTION_KEYS))
def test_validate_questions_missing_required_keys(missing_key: str) -> None:
    """Verify validate_questions detects every required key omission."""
    item = create_sample_question()
    del item[missing_key]

    with pytest.raises(ValueError, match="is missing keys"):
        validate_questions([item])


def test_validate_questions_duplicate_ids() -> None:
    """Verify validate_questions detects duplicate question identifiers."""
    items = [
        create_sample_question("Q001"),
        create_sample_question("Q001"),
    ]
    with pytest.raises(ValueError, match="Duplicate question ID detected: 'Q001'"):
        validate_questions(items)


def test_validate_questions_invalid_category() -> None:
    """Verify validate_questions rejects unapproved category names."""
    item = create_sample_question(category="unsupported_category")
    with pytest.raises(ValueError, match="invalid category 'unsupported_category'"):
        validate_questions([item])


def test_validate_questions_invalid_difficulty() -> None:
    """Verify validate_questions rejects unapproved difficulty levels."""
    item = create_sample_question(difficulty="super_hard")
    with pytest.raises(ValueError, match="invalid difficulty 'super_hard'"):
        validate_questions([item])


def test_validate_questions_empty_question_or_expected_behavior() -> None:
    """Verify validate_questions rejects empty or whitespace-only questions and behaviors."""
    item_empty_q = create_sample_question(question="   ")
    with pytest.raises(ValueError, match="must have a non-empty string 'question'"):
        validate_questions([item_empty_q])

    item_empty_exp = create_sample_question(expected_behavior="")
    with pytest.raises(ValueError, match="must have a non-empty string 'expected_behavior'"):
        validate_questions([item_empty_exp])


def test_validate_questions_invalid_requires_citation_type() -> None:
    """Verify validate_questions enforces boolean type for requires_citation."""
    item = create_sample_question()
    item["requires_citation"] = "yes"  # type: ignore[typeddict-item]
    with pytest.raises(ValueError, match="'requires_citation' must be a boolean"):
        validate_questions([item])


# =====================================================================
# 3. DocumentEvaluator Initialization Tests
# =====================================================================

def test_evaluator_initialization_defaults(sample_questions_file: Path) -> None:
    """Verify DocumentEvaluator initializes with loaded questions and default overrides."""
    evaluator = DocumentEvaluator(questions_path=sample_questions_file)
    assert len(evaluator.questions) == 4
    assert evaluator.citation_required_override is None


def test_evaluator_initialization_with_overrides(sample_questions_file: Path) -> None:
    """Verify DocumentEvaluator respects citation_required boolean overrides."""
    eval_true = DocumentEvaluator(questions_path=sample_questions_file, citation_required=True)
    assert eval_true.citation_required_override is True

    eval_false = DocumentEvaluator(questions_path=sample_questions_file, citation_required=False)
    assert eval_false.citation_required_override is False


def test_evaluator_initialization_invalid_path(tmp_path: Path) -> None:
    """Verify DocumentEvaluator raises FileNotFoundError if target file does not exist."""
    with pytest.raises(FileNotFoundError):
        DocumentEvaluator(questions_path=tmp_path / "missing.json")


# =====================================================================
# 4. Answer Extraction Tests (_extract_answer)
# =====================================================================

def test_extract_answer_from_string(sample_questions_file: Path) -> None:
    """Verify extraction of plain string responses."""
    evaluator = DocumentEvaluator(questions_path=sample_questions_file)
    ans, sources = evaluator._extract_answer("The invoice total is ₹50,000.")
    assert ans == "The invoice total is ₹50,000."
    assert sources == []


def test_extract_answer_from_chat_response(sample_questions_file: Path) -> None:
    """Verify extraction from Pydantic ChatResponse instance."""
    evaluator = DocumentEvaluator(questions_path=sample_questions_file)
    citation = SourceCitation(
        document_id="doc-001",
        filename="invoice.pdf",
        page_number=2,
        content="Total Amount: ₹50,000",
    )
    response_obj = ChatResponse(
        answer="The total is ₹50,000.",
        sources=[citation],
    )

    ans, sources = evaluator._extract_answer(response_obj)
    assert ans == "The total is ₹50,000."
    assert len(sources) == 1
    assert sources[0].page_number == 2


def test_extract_answer_from_dict_variants(sample_questions_file: Path) -> None:
    """Verify extraction from dict payloads with 'answer', 'output', or 'response' keys."""
    evaluator = DocumentEvaluator(questions_path=sample_questions_file)

    # Variant 1: answer + sources
    ans1, src1 = evaluator._extract_answer({"answer": "Amount is ₹50k", "sources": [{"page": 1}]})
    assert ans1 == "Amount is ₹50k"
    assert len(src1) == 1

    # Variant 2: output
    ans2, src2 = evaluator._extract_answer({"output": "Contract executed on Jan 1st"})
    assert ans2 == "Contract executed on Jan 1st"
    assert src2 == []

    # Variant 3: response
    ans3, src3 = evaluator._extract_answer({"response": "Summary text"})
    assert ans3 == "Summary text"
    assert src3 == []


def test_extract_answer_from_custom_objects(sample_questions_file: Path) -> None:
    """Verify extraction from custom objects exposing .answer or .output attributes."""
    evaluator = DocumentEvaluator(questions_path=sample_questions_file)

    class CustomAnswerObj:
        answer = "Answer from attribute"
        sources = ["source_a"]

    class CustomOutputObj:
        output = "Output from attribute"

    ans1, src1 = evaluator._extract_answer(CustomAnswerObj())
    assert ans1 == "Answer from attribute"
    assert src1 == ["source_a"]

    ans2, src2 = evaluator._extract_answer(CustomOutputObj())
    assert ans2 == "Output from attribute"
    assert src2 == []


def test_extract_answer_from_none(sample_questions_file: Path) -> None:
    """Verify extraction returns empty string and empty list when response is None."""
    evaluator = DocumentEvaluator(questions_path=sample_questions_file)
    ans, sources = evaluator._extract_answer(None)
    assert ans == ""
    assert sources == []


def test_extract_answer_unsupported_type(sample_questions_file: Path) -> None:
    """Verify extraction raises ValueError for unsupported response types."""
    evaluator = DocumentEvaluator(questions_path=sample_questions_file)
    with pytest.raises(ValueError, match="Unsupported response format"):
        evaluator._extract_answer(12345)


# =====================================================================
# 5. Citation Detection Tests (check_citation)
# =====================================================================

def test_check_citation_not_required(sample_questions_file: Path) -> None:
    """Verify citation absence is valid when requires_citation is False."""
    evaluator = DocumentEvaluator(questions_path=sample_questions_file)
    found, valid = evaluator.check_citation(
        answer="Plain answer without citations.",
        sources=[],
        requires_citation=False,
    )
    assert found is False
    assert valid is True


def test_check_citation_with_source_citation_model(sample_questions_file: Path) -> None:
    """Verify detection and validation using structured SourceCitation instances."""
    evaluator = DocumentEvaluator(questions_path=sample_questions_file)
    citation = SourceCitation(
        document_id="doc-01",
        filename="contract.pdf",
        page_number=3,
        content="Clause 14",
    )
    found, valid = evaluator.check_citation(
        answer="Contract clause stated here.",
        sources=[citation],
        requires_citation=True,
    )
    assert found is True
    assert valid is True


def test_check_citation_with_source_dict(sample_questions_file: Path) -> None:
    """Verify detection and validation using dictionary-based sources."""
    evaluator = DocumentEvaluator(questions_path=sample_questions_file)
    dict_source = {"filename": "invoice.pdf", "page_number": 2, "content": "Total: 100"}
    found, valid = evaluator.check_citation(
        answer="Here is the total.",
        sources=[dict_source],
        requires_citation=True,
    )
    assert found is True
    assert valid is True


def test_check_citation_with_inline_text_patterns(sample_questions_file: Path) -> None:
    """Verify inline regex matches for 'Page 2', 'Page: 4', 'page #1'."""
    evaluator = DocumentEvaluator(questions_path=sample_questions_file)

    found1, valid1 = evaluator.check_citation("Total amount is ₹50,000 [Page 2].", [], requires_citation=True)
    assert found1 is True
    assert valid1 is True

    found2, valid2 = evaluator.check_citation("Refer to page: 5 for terms.", [], requires_citation=True)
    assert found2 is True
    assert valid2 is True


def test_check_citation_missing_when_required(sample_questions_file: Path) -> None:
    """Verify check_citation returns (False, False) when citation is required but absent."""
    evaluator = DocumentEvaluator(questions_path=sample_questions_file)
    found, valid = evaluator.check_citation(
        answer="Total amount is ₹50,000.",
        sources=[],
        requires_citation=True,
    )
    assert found is False
    assert valid is False


def test_check_citation_invalid_page_index(sample_questions_file: Path) -> None:
    """Verify check_citation flags page numbers < 1 as invalid."""
    evaluator = DocumentEvaluator(questions_path=sample_questions_file)
    found, valid = evaluator.check_citation(
        answer="Found on Page 0.",
        sources=[],
        requires_citation=True,
    )
    assert found is True
    assert valid is False


# =====================================================================
# 6. evaluate_answer Tests
# =====================================================================

def test_evaluate_answer_empty_text(sample_questions_file: Path) -> None:
    """Verify empty or whitespace-only answers fail with 0.0 score."""
    evaluator = DocumentEvaluator(questions_path=sample_questions_file)
    q = create_sample_question()

    passed, score, _, _, err = evaluator.evaluate_answer(q, "", [])
    assert passed is False
    assert score == 0.0
    assert err == "Empty answer returned."


def test_evaluate_answer_rag_with_citation(sample_questions_file: Path) -> None:
    """Verify standard RAG answer with required citation gets full score."""
    evaluator = DocumentEvaluator(questions_path=sample_questions_file)
    q = create_sample_question(category="rag", requires_citation=True)

    passed, score, cit_found, cit_valid, err = evaluator.evaluate_answer(
        question_item=q,
        answer="The total amount is ₹50,000 on Page 2.",
        sources=[],
    )
    assert passed is True
    assert score == 1.0  # 1.0 * 0.6 + 1.0 * 0.4
    assert cit_found is True
    assert cit_valid is True
    assert err is None


def test_evaluate_answer_rag_missing_required_citation(sample_questions_file: Path) -> None:
    """Verify RAG answer fails citation requirement with weighted score calculation."""
    evaluator = DocumentEvaluator(questions_path=sample_questions_file)
    q = create_sample_question(category="rag", requires_citation=True)

    passed, score, cit_found, cit_valid, err = evaluator.evaluate_answer(
        question_item=q,
        answer="The total amount is ₹50,000.",
        sources=[],
    )
    assert passed is False
    assert score == 0.6  # 1.0 * 0.6 + 0.0 * 0.4
    assert cit_found is False
    assert cit_valid is False
    assert err == "Required citation missing or structurally invalid."


def test_evaluate_answer_missing_information_success(sample_questions_file: Path) -> None:
    """Verify missing_information question succeeds when uncertainty phrase is present."""
    evaluator = DocumentEvaluator(questions_path=sample_questions_file)
    q = create_sample_question(category="missing_information", requires_citation=False)

    passed, score, _, _, err = evaluator.evaluate_answer(
        question_item=q,
        answer="The bank account number was not found in the provided document.",
        sources=[],
    )
    assert passed is True
    assert score == 1.0
    assert err is None


def test_evaluate_answer_missing_information_fabrication(sample_questions_file: Path) -> None:
    """Verify missing_information question fails when answer hallucinates a value."""
    evaluator = DocumentEvaluator(questions_path=sample_questions_file)
    q = create_sample_question(category="missing_information", requires_citation=False)

    passed, score, _, _, err = evaluator.evaluate_answer(
        question_item=q,
        answer="The bank account number is ACCT-987654321.",
        sources=[],
    )
    assert passed is False
    assert score == 0.0
    assert "failed to indicate that information was missing" in (err or "")


def test_evaluate_answer_comparison_category(sample_questions_file: Path) -> None:
    """Verify comparison evaluation checks for comparative terminology."""
    evaluator = DocumentEvaluator(questions_path=sample_questions_file)
    q = create_sample_question(category="comparison", requires_citation=False)

    # Answer containing comparison keyword "differ"
    passed1, score1, _, _, _ = evaluator.evaluate_answer(
        question_item=q,
        answer="The dates in Document A and Document B differ.",
        sources=[],
    )
    assert passed1 is True
    assert score1 == 1.0

    # Answer lacking comparison keyword gets partial score 0.5
    passed2, score2, _, _, _ = evaluator.evaluate_answer(
        question_item=q,
        answer="Document A has an invoice number.",
        sources=[],
    )
    assert passed2 is True
    assert score2 == 0.5


# =====================================================================
# 7. Citation Override Tests
# =====================================================================

def test_citation_required_override_behavior(sample_questions_file: Path) -> None:
    """Verify global citation_required_override takes precedence over question schema."""
    # Question has requires_citation=False
    q_no_cit = create_sample_question(requires_citation=False)

    # Evaluator enforces citation_required=True
    eval_enforce = DocumentEvaluator(sample_questions_file, citation_required=True)
    passed_no_cit, _, _, _, err = eval_enforce.evaluate_answer(q_no_cit, "Answer without citation", [])
    assert passed_no_cit is False
    assert err == "Required citation missing or structurally invalid."

    # Question has requires_citation=True
    q_has_cit = create_sample_question(requires_citation=True)

    # Evaluator disables citation_required=False
    eval_disable = DocumentEvaluator(sample_questions_file, citation_required=False)
    passed_cit_disabled, score, _, _, _ = eval_disable.evaluate_answer(q_has_cit, "Answer without citation", [])
    assert passed_cit_disabled is True
    assert score == 1.0


# =====================================================================
# 8. Full Evaluation Flow Tests (evaluate)
# =====================================================================

def test_evaluate_runner_success(sample_questions_file: Path) -> None:
    """Verify evaluate() executes all questions and produces structured summary metrics."""
    evaluator = DocumentEvaluator(questions_path=sample_questions_file)

    def mock_answer_fn(question_item: dict[str, Any]) -> str:
        cat = question_item["category"]
        if cat == "missing_information":
            return "This information is not available in the document."
        if cat == "comparison":
            return "Both documents are identical in amount. [Page 1]"
        return "The requested information is ₹50,000 on Page 1."

    summary = evaluator.evaluate(mock_answer_fn)

    assert isinstance(summary, EvaluationSummary)
    assert summary.total_questions == 4
    assert summary.evaluated_questions == 4
    assert summary.passed_questions == 4
    assert summary.failed_questions == 0
    assert summary.errors == 0
    assert summary.overall_score == 1.0
    assert summary.citation_score == 1.0
    assert len(summary.results) == 4

    # Verify category metrics
    assert "rag" in summary.category_scores
    assert "citation" in summary.category_scores
    assert "missing_information" in summary.category_scores
    assert "comparison" in summary.category_scores
    assert summary.category_scores["rag"]["passed"] == 1

    # Verify difficulty metrics
    assert "easy" in summary.difficulty_scores
    assert "medium" in summary.difficulty_scores
    assert "hard" in summary.difficulty_scores


# =====================================================================
# 9. Error Isolation Tests
# =====================================================================

def test_evaluate_continues_after_question_error(sample_questions_file: Path) -> None:
    """Verify an exception in answer_fn for one question does not halt evaluation of others."""
    evaluator = DocumentEvaluator(questions_path=sample_questions_file)

    def failing_answer_fn(question_item: dict[str, Any]) -> str:
        if question_item["id"] == "Q002":
            raise RuntimeError("Simulated internal LLM service crash")
        if question_item["category"] == "missing_information":
            return "Information is not provided."
        return "Valid response on Page 1."

    summary = evaluator.evaluate(failing_answer_fn)

    assert summary.total_questions == 4
    assert summary.evaluated_questions == 4
    assert summary.passed_questions == 3
    assert summary.failed_questions == 1
    assert summary.errors == 1

    # Locate failed question result
    failed_result = next(r for r in summary.results if r.question_id == "Q002")
    assert failed_result.passed is False
    assert failed_result.score == 0.0
    assert "Simulated internal LLM service crash" in (failed_result.error or "")


# =====================================================================
# 10. Corrected Citation Score Calculation Test (Bug Regression Guard)
# =====================================================================

def test_citation_score_includes_all_citation_required_questions(tmp_path: Path) -> None:
    """Verify citation_score denominator includes ALL questions where citation was required."""
    questions = [
        create_sample_question(qid="Q001", requires_citation=True),
        create_sample_question(qid="Q002", requires_citation=True),
        create_sample_question(qid="Q003", requires_citation=True),
    ]
    file_path = tmp_path / "citation_test.json"
    file_path.write_text(json.dumps({"questions": questions}), encoding="utf-8")

    evaluator = DocumentEvaluator(questions_path=file_path)

    # Q001 has citation, Q002 and Q003 do not
    def answer_fn(q: dict[str, Any]) -> str:
        if q["id"] == "Q001":
            return "Answer with citation on Page 1."
        return "Answer without citation."

    summary = evaluator.evaluate(answer_fn)

    # Out of 3 citation-required questions, only 1 valid citation was present
    # Score should be 1 / 3 = 0.333, not 1 / 1 = 1.0
    assert summary.citation_score == pytest.approx(0.333, abs=0.001)


# =====================================================================
# 11. Report Export Tests (save_report)
# =====================================================================

def test_save_report_creates_valid_json_report(sample_questions_file: Path, tmp_path: Path) -> None:
    """Verify save_report writes a complete, valid JSON evaluation artifact."""
    evaluator = DocumentEvaluator(questions_path=sample_questions_file)

    summary = evaluator.evaluate(lambda q: "Simple valid answer on Page 1.")
    report_file = tmp_path / "reports" / "evaluation_report.json"

    evaluator.save_report(summary=summary, path=report_file)

    assert report_file.exists()
    assert report_file.is_file()

    with open(report_file, "r", encoding="utf-8") as f:
        report_data = json.load(f)

    assert "timestamp" in report_data
    assert report_data["total_questions"] == 4
    assert report_data["evaluated_questions"] == 4
    assert report_data["passed_questions"] == 3  # Q003 (missing_info) didn't use uncertainty language
    assert "overall_score" in report_data
    assert "citation_score" in report_data
    assert "category_scores" in report_data
    assert "difficulty_scores" in report_data
    assert len(report_data["results"]) == 4
