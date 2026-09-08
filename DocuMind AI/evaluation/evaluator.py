"""Deterministic evaluation framework for DocuMind AI.

This module provides an offline, lightweight benchmarking engine to evaluate DocuMind AI's
RAG, structured extraction, citation attribution, multi-hop reasoning, missing-information
handling, and document comparison capabilities without external evaluation services.

Example Usage:
--------------
    from evaluation.evaluator import DocumentEvaluator
    from app.agents.document_agent import DocumentAgent

    agent = DocumentAgent()
    evaluator = DocumentEvaluator("evaluation/questions.json")

    def answer_fn(question_item: dict) -> str:
        return agent.run(question=question_item["question"])

    summary = evaluator.evaluate(answer_fn)
    evaluator.save_report(summary, "evaluation/results.json")
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
from typing import Any, Callable

# Ensure project root is in sys.path when executed directly as a script
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from app.core.logger import get_logger
from app.models.schemas import ChatResponse, SourceCitation

logger = get_logger(__name__)

# Permitted dataset enumerations
VALID_CATEGORIES = {
    "rag",
    "citation",
    "extraction",
    "reasoning",
    "missing_information",
    "comparison",
    "cross_document",
}

VALID_DIFFICULTIES = {"easy", "medium", "hard"}

REQUIRED_QUESTION_KEYS = {
    "id",
    "category",
    "question",
    "document_type",
    "difficulty",
    "requires_citation",
    "expected_behavior",
}

UNCERTAINTY_PHRASES = [
    "not found",
    "not provided",
    "not available",
    "cannot find",
    "could not find",
    "not mentioned",
    "unable to determine",
    "no information",
    "does not contain",
    "not specified",
    "insufficient information",
    "not stated",
]

COMPARISON_KEYWORDS = [
    "match",
    "matches",
    "mismatch",
    "mismatches",
    "differ",
    "differs",
    "different",
    "difference",
    "same",
    "identical",
    "missing",
    "discrepancy",
    "discrepancies",
    "consistent",
    "inconsistent",
    "aligned",
]


@dataclass
class EvaluationResult:
    """Evaluation output for an individual benchmark question."""

    question_id: str
    category: str
    question: str
    document_type: str
    difficulty: str
    answer: str
    passed: bool
    score: float
    citation_found: bool
    citation_valid: bool
    expected_behavior: str
    error: str | None = None


@dataclass
class EvaluationSummary:
    """Consolidated metrics across all evaluated benchmark questions."""

    total_questions: int
    evaluated_questions: int
    passed_questions: int
    failed_questions: int
    errors: int
    overall_score: float
    citation_score: float
    category_scores: dict[str, dict[str, Any]] = field(default_factory=dict)
    difficulty_scores: dict[str, dict[str, Any]] = field(default_factory=dict)
    results: list[EvaluationResult] = field(default_factory=list)


def load_questions(path: str | Path) -> list[dict[str, Any]]:
    """Load and perform structure validation on a questions JSON dataset file.

    Args:
        path: Filesystem path to the evaluation questions JSON.

    Returns:
        list[dict[str, Any]]: List of validated question dictionaries.

    Raises:
        FileNotFoundError: If the question file does not exist.
        ValueError: If JSON syntax is invalid or questions schema is malformed.
    """
    file_path = Path(path)
    if not file_path.exists() or not file_path.is_file():
        raise FileNotFoundError(f"Evaluation questions file not found at: {file_path}")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON format in questions file '{file_path}': {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError("Questions dataset root must be a JSON object.")

    if "questions" not in data or not isinstance(data["questions"], list):
        raise ValueError("Dataset JSON must contain a top-level 'questions' list.")

    questions = data["questions"]
    validate_questions(questions)
    return questions


def validate_questions(questions: list[dict[str, Any]]) -> None:
    """Validate question items for unique IDs, permitted enumerations, and key presence.

    Args:
        questions: List of question dictionaries.

    Raises:
        ValueError: If any validation rule is violated.
    """
    seen_ids: set[str] = set()

    for idx, item in enumerate(questions):
        if not isinstance(item, dict):
            raise ValueError(f"Question at index {idx} must be a JSON object, got {type(item).__name__}.")

        missing_keys = REQUIRED_QUESTION_KEYS - set(item.keys())
        if missing_keys:
            raise ValueError(
                f"Question at index {idx} (ID: {item.get('id', 'unknown')}) is missing keys: {sorted(missing_keys)}"
            )

        qid = item["id"]
        if not isinstance(qid, str) or not qid.strip():
            raise ValueError(f"Question at index {idx} has an invalid or empty 'id'.")

        if qid in seen_ids:
            raise ValueError(f"Duplicate question ID detected: '{qid}'.")
        seen_ids.add(qid)

        category = item["category"]
        if category not in VALID_CATEGORIES:
            raise ValueError(
                f"Question '{qid}' has invalid category '{category}'. Must be one of: {sorted(VALID_CATEGORIES)}"
            )

        difficulty = item["difficulty"]
        if difficulty not in VALID_DIFFICULTIES:
            raise ValueError(
                f"Question '{qid}' has invalid difficulty '{difficulty}'. Must be one of: {sorted(VALID_DIFFICULTIES)}"
            )

        if not isinstance(item["question"], str) or not item["question"].strip():
            raise ValueError(f"Question '{qid}' must have a non-empty string 'question'.")

        if not isinstance(item["expected_behavior"], str) or not item["expected_behavior"].strip():
            raise ValueError(f"Question '{qid}' must have a non-empty string 'expected_behavior'.")

        if not isinstance(item["requires_citation"], bool):
            raise ValueError(f"Question '{qid}' 'requires_citation' must be a boolean.")


class DocumentEvaluator:
    """Deterministic evaluation runner for DocuMind AI pipelines and agents."""

    def __init__(
        self,
        questions_path: str | Path = "evaluation/questions.json",
        citation_required: bool | None = None,
    ) -> None:
        """Initialize the evaluator with target dataset questions.

        Args:
            questions_path: Path to questions.json file.
            citation_required: Optional global override for citation requirements.
        """
        self.questions_path = Path(questions_path)
        self.citation_required_override = citation_required
        self.questions = load_questions(self.questions_path)
        logger.info(
            "DocumentEvaluator initialized with %d questions from '%s'.",
            len(self.questions),
            self.questions_path,
        )

    def _extract_answer(self, response: Any) -> tuple[str, list[Any]]:
        """Normalize various answer caller return structures into (answer_text, sources_list).

        Args:
            response: Raw response object (str, ChatResponse, dict, or object with attributes).

        Returns:
            tuple[str, list[Any]]: Normalized answer string and list of source objects/dicts.

        Raises:
            ValueError: If response format is unsupported or cannot be extracted.
        """
        if response is None:
            return "", []

        if isinstance(response, str):
            return response.strip(), []

        if isinstance(response, ChatResponse):
            return response.answer.strip(), list(response.sources)

        if isinstance(response, dict):
            answer_text = response.get("answer") or response.get("output") or response.get("response") or ""
            sources = response.get("sources") or response.get("citations") or []
            return str(answer_text).strip(), list(sources) if isinstance(sources, list) else []

        if hasattr(response, "answer"):
            sources = getattr(response, "sources", [])
            return str(response.answer).strip(), list(sources) if isinstance(sources, list) else []

        if hasattr(response, "output"):
            sources = getattr(response, "sources", [])
            return str(response.output).strip(), list(sources) if isinstance(sources, list) else []

        raise ValueError(f"Unsupported response format from answer callable: {type(response).__name__}")

    def check_citation(
        self,
        answer: str,
        sources: list[Any],
        requires_citation: bool,
    ) -> tuple[bool, bool]:
        """Perform structural validation of page-level citations.

        Args:
            answer: Generated textual answer.
            sources: Structured sources list if returned by RAG/agent.
            requires_citation: Whether the question specifies citation requirement.

        Returns:
            tuple[bool, bool]: (citation_found, citation_valid)
        """
        if not requires_citation:
            # If citation not required, citation absence is never a failure
            return False, True

        # Check structured sources first
        if sources:
            valid_source = False
            for src in sources:
                if isinstance(src, SourceCitation):
                    if src.filename and src.page_number is not None and src.page_number >= 1:
                        valid_source = True
                        break
                elif isinstance(src, dict):
                    page = src.get("page_number") or src.get("page")
                    if src.get("filename") and page is not None:
                        try:
                            if int(page) >= 1:
                                valid_source = True
                                break
                        except (ValueError, TypeError):
                            pass

            if valid_source:
                return True, True

        # Check for inline textual citation patterns: e.g. "Page 2", "Page: 1", "page 4"
        page_pattern = re.compile(r"\bpage\s*[:#]?\s*(\d+)\b", re.IGNORECASE)
        matches = page_pattern.findall(answer)
        if matches:
            has_valid_num = any(int(m) >= 1 for m in matches if m.isdigit())
            return True, has_valid_num

        return False, False

    def evaluate_answer(
        self,
        question_item: dict[str, Any],
        answer: str,
        sources: list[Any],
    ) -> tuple[bool, float, bool, bool, str | None]:
        """Evaluate answer quality and citation validity deterministically.

        Args:
            question_item: Question dictionary from the dataset.
            answer: Extracted plain text answer.
            sources: Extracted structured sources list.

        Returns:
            tuple[bool, float, bool, bool, str | None]:
                (passed, score, citation_found, citation_valid, error_message)
        """
        if not answer or not answer.strip():
            return False, 0.0, False, False, "Empty answer returned."

        category = question_item["category"]
        requires_citation = (
            self.citation_required_override
            if self.citation_required_override is not None
            else question_item.get("requires_citation", False)
        )

        citation_found, citation_valid = self.check_citation(
            answer=answer,
            sources=sources,
            requires_citation=requires_citation,
        )

        answer_lower = answer.lower()
        answer_score = 1.0
        error_msg: str | None = None

        if category == "missing_information":
            # For missing information, verify uncertainty / not-found language
            has_uncertainty = any(phrase in answer_lower for phrase in UNCERTAINTY_PHRASES)
            if has_uncertainty:
                answer_score = 1.0
            else:
                answer_score = 0.0
                error_msg = "Answer failed to indicate that information was missing or not found."

        elif category == "comparison":
            # For comparison, verify comparison/contrast terms
            has_comparison_lang = any(keyword in answer_lower for keyword in COMPARISON_KEYWORDS)
            if has_comparison_lang:
                answer_score = 1.0
            else:
                answer_score = 0.5  # Partial credit for generating an answer without explicit comparison cues

        # Compute weighted composite score
        if requires_citation and category != "missing_information":
            citation_weight = 0.4
            answer_weight = 0.6
            cit_score = 1.0 if (citation_found and citation_valid) else 0.0
            total_score = round((answer_score * answer_weight) + (cit_score * citation_weight), 2)
            passed = (answer_score >= 0.5) and citation_valid and citation_found
            if not (citation_found and citation_valid) and not error_msg:
                error_msg = "Required citation missing or structurally invalid."
        else:
            total_score = round(answer_score, 2)
            passed = total_score >= 0.5

        return passed, total_score, citation_found, citation_valid, error_msg

    def evaluate(self, answer_fn: Callable[[dict[str, Any]], Any]) -> EvaluationSummary:
        """Execute evaluation across all questions using the supplied answering callable.

        Args:
            answer_fn: Callable accepting a question dict and returning a response string/object.

        Returns:
            EvaluationSummary: Consolidated metrics and individual results.
        """
        results: list[EvaluationResult] = []
        total = len(self.questions)
        passed_count = 0
        failed_count = 0
        error_count = 0

        category_metrics: dict[str, dict[str, Any]] = {}
        difficulty_metrics: dict[str, dict[str, Any]] = {}

        logger.info("Beginning benchmark evaluation on %d questions...", total)

        for q in self.questions:
            qid = q["id"]
            cat = q["category"]
            diff = q["difficulty"]

            if cat not in category_metrics:
                category_metrics[cat] = {"count": 0, "passed": 0, "total_score": 0.0}
            if diff not in difficulty_metrics:
                difficulty_metrics[diff] = {"count": 0, "passed": 0, "total_score": 0.0}

            category_metrics[cat]["count"] += 1
            difficulty_metrics[diff]["count"] += 1

            try:
                raw_response = answer_fn(q)
                answer, sources = self._extract_answer(raw_response)

                passed, score, cit_found, cit_valid, err = self.evaluate_answer(
                    question_item=q,
                    answer=answer,
                    sources=sources,
                )

                if passed:
                    passed_count += 1
                    category_metrics[cat]["passed"] += 1
                    difficulty_metrics[diff]["passed"] += 1
                else:
                    failed_count += 1

                category_metrics[cat]["total_score"] += score
                difficulty_metrics[diff]["total_score"] += score

                results.append(
                    EvaluationResult(
                        question_id=qid,
                        category=cat,
                        question=q["question"],
                        document_type=q["document_type"],
                        difficulty=diff,
                        answer=answer,
                        passed=passed,
                        score=score,
                        citation_found=cit_found,
                        citation_valid=cit_valid,
                        expected_behavior=q["expected_behavior"],
                        error=err,
                    )
                )

            except Exception as exc:
                logger.exception("Error evaluating question '%s': %s", qid, exc)
                failed_count += 1
                error_count += 1

                results.append(
                    EvaluationResult(
                        question_id=qid,
                        category=cat,
                        question=q["question"],
                        document_type=q["document_type"],
                        difficulty=diff,
                        answer="",
                        passed=False,
                        score=0.0,
                        citation_found=False,
                        citation_valid=False,
                        expected_behavior=q["expected_behavior"],
                        error=str(exc),
                    )
                )

        # Aggregate category scores
        formatted_category_scores = {}
        for cat, val in category_metrics.items():
            cnt = val["count"]
            avg_score = round(val["total_score"] / cnt, 3) if cnt > 0 else 0.0
            formatted_category_scores[cat] = {
                "count": cnt,
                "passed": val["passed"],
                "average_score": avg_score,
            }

        # Aggregate difficulty scores
        formatted_difficulty_scores = {}
        for diff, val in difficulty_metrics.items():
            cnt = val["count"]
            avg_score = round(val["total_score"] / cnt, 3) if cnt > 0 else 0.0
            formatted_difficulty_scores[diff] = {
                "count": cnt,
                "passed": val["passed"],
                "average_score": avg_score,
            }

        # Calculate overall score and citation compliance score
        overall_score = round(sum(r.score for r in results) / total, 3) if total > 0 else 0.0
        citation_results = [
            r
            for r, q in zip(results, self.questions)
            if (
                self.citation_required_override
                if self.citation_required_override is not None
                else q.get("requires_citation", False)
            )
        ]

        cit_score = (
            round(
                sum(
                    1.0
                    for r in citation_results
                    if r.citation_found and r.citation_valid
                )
                / len(citation_results),
                3,
            )
            if citation_results
            else 1.0
        )

        logger.info(
            "Evaluation completed: Total=%d, Passed=%d, Failed=%d, Errors=%d, Score=%.2f",
            total,
            passed_count,
            failed_count,
            error_count,
            overall_score,
        )

        return EvaluationSummary(
            total_questions=total,
            evaluated_questions=len(results),
            passed_questions=passed_count,
            failed_questions=failed_count,
            errors=error_count,
            overall_score=overall_score,
            citation_score=cit_score,
            category_scores=formatted_category_scores,
            difficulty_scores=formatted_difficulty_scores,
            results=results,
        )

    def save_report(self, summary: EvaluationSummary, path: str | Path) -> None:
        """Serialize and export an evaluation report to a JSON file.

        Args:
            summary: Evaluated summary metrics.
            path: Destination file path.
        """
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)

        report_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_questions": summary.total_questions,
            "evaluated_questions": summary.evaluated_questions,
            "passed_questions": summary.passed_questions,
            "failed_questions": summary.failed_questions,
            "errors": summary.errors,
            "overall_score": summary.overall_score,
            "citation_score": summary.citation_score,
            "category_scores": summary.category_scores,
            "difficulty_scores": summary.difficulty_scores,
            "results": [asdict(r) for r in summary.results],
        }

        with open(destination, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)

        logger.info("Evaluation report successfully saved to '%s'.", destination)


if __name__ == "__main__":
    questions_file = Path("evaluation/questions.json")
    print("=" * 60)
    print(" DocuMind AI — Benchmark Evaluation Framework")
    print("=" * 60)

    try:
        loaded_qs = load_questions(questions_file)
        print(f" Loaded and validated {len(loaded_qs)} benchmark questions from '{questions_file}'.")
        print("\nDistribution by Category:")
        categories = {}
        for q in loaded_qs:
            categories[q["category"]] = categories.get(q["category"], 0) + 1
        for cat, cnt in sorted(categories.items()):
            print(f"  - {cat:<20}: {cnt}")

        print("\nDistribution by Difficulty:")
        difficulties = {}
        for q in loaded_qs:
            difficulties[q["difficulty"]] = difficulties.get(q["difficulty"], 0) + 1
        for diff, cnt in sorted(difficulties.items()):
            print(f"  - {diff:<20}: {cnt}")

        print("\n Note: To execute a live benchmark, connect an answering callable")
        print(" (e.g. DocumentAgent.run or RAGPipeline.answer) using DocumentEvaluator.evaluate().")
        print("=" * 60)
    except Exception as exc:
        print(f" Error loading evaluation dataset: {exc}")
