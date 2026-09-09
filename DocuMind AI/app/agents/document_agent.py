"""Document Analysis Agent for DocuMind AI.

This module provides the primary autonomous tool-calling agent that routes user
requests to specialized document intelligence tools (grounded RAG question answering
and structured entity extraction) to produce accurate, evidence-backed answers using Google Gemini.
"""

from typing import Any

try:
    from langchain.agents import AgentExecutor, create_tool_calling_agent
except ImportError:
    from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI

from app.agents.tools import (
    document_question_answering,
    extract_document_fields,
)
from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are DocuMind AI, an intelligent and precision-oriented Document Analysis Agent.

Your mission is to assist users in understanding, interrogating, and extracting structured insights from documents (e.g. contracts, invoices, application forms, receipts, and reports).

You have access to the following tools:
1. `document_question_answering`:
   - Purpose: Retrieve relevant semantic snippets and answer factual questions grounded in indexed documents with page-level citations.
   - When to use: When the user asks specific questions about document content, figures, clauses, dates, parties, or facts.
   - Inputs: Pass the user's question and any relevant `document_ids` provided in the context.

2. `extract_document_fields`:
   - Purpose: Extract structured key-value entities, classifications, and executive summaries.
   - When to use: When the user explicitly requests structured field extraction or comprehensive attribute extraction.
   - Inputs: Requires complete `document_text`, `document_id`, and `filename` from the context.

Crucial Operating Guidelines:
- Grounding: Never guess, assume, or invent document facts, dates, amounts, or terms. Rely exclusively on tool outputs.
- Evidence & Citations: Always preserve document names and page numbers in your final answer when provided by the tools.
- Extraction Prerequisite: If the user asks for structured field extraction, check if `Full Document Text` is available in the Document Context. If it is NOT provided (or marked "Not provided"), do NOT invent the fields. Instead, clearly explain that full document text is required for structured extraction.
- Insufficient Information: If the tool output indicates that information was not found or context is insufficient, clearly state that the provided documents do not contain the requested information.
- Off-topic Queries: If the user asks general or unrelated questions, politely inform them that you are specialized in document analysis and question answering.
- Output Clarity: Deliver concise, professional, and well-structured responses."""

HUMAN_PROMPT = """Document Context:
{document_context}

User Question:
{input}"""


class DocumentAgent:
    """Autonomous tool-calling agent orchestrating document intelligence tasks."""

    def __init__(
        self,
        model: str | None = None,
        temperature: float | None = None,
        tools: list[Any] | None = None,
    ) -> None:
        """Initialize the DocumentAgent with models, tools, and execution graph.

        Args:
            model: LLM model identifier (defaults to settings.LLM_MODEL).
            temperature: Sampling temperature (defaults to settings.LLM_TEMPERATURE).
            tools: Optional custom list of LangChain tools.

        Raises:
            ValueError: If the configured LLM provider is unsupported or credentials are missing.
        """
        if settings.LLM_PROVIDER.lower() not in {"gemini", "google"}:
            raise ValueError(
                f"Unsupported LLM provider '{settings.LLM_PROVIDER}'. DocumentAgent currently supports 'gemini' only."
            )

        api_key = settings.GEMINI_API_KEY or settings.GOOGLE_API_KEY
        if not api_key or not api_key.strip() or api_key.strip() == "YOUR_GEMINI_API_KEY_HERE":
            raise ValueError(
                "GEMINI_API_KEY is not configured. Please set GEMINI_API_KEY in your environment or .env file."
            )

        self.model_name = model or settings.LLM_MODEL
        self.temperature = (
            settings.LLM_TEMPERATURE if temperature is None else temperature
        )

        self.llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            temperature=self.temperature,
            google_api_key=api_key.strip(),
        )

        if tools is not None:
            if not tools:
                raise ValueError("Custom tools list must not be empty.")
            self.tools = tools
        else:
            self.tools = [
                document_question_answering,
                extract_document_fields,
            ]

        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                ("human", HUMAN_PROMPT),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        self.agent = create_tool_calling_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self.prompt,
        )

        self.executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=False,
            handle_parsing_errors=True,
            max_iterations=5,
        )

        logger.info(
            "DocumentAgent initialized successfully (model=%s, temperature=%.2f, tools_count=%d, key_configured=%s)",
            self.model_name,
            self.temperature,
            len(self.tools),
            bool(settings.GEMINI_API_KEY or settings.GOOGLE_API_KEY),
        )

    def _build_document_context(
        self,
        document_id: str | None = None,
        filename: str | None = None,
        document_type: str | None = None,
        document_text: str | None = None,
        document_ids: list[str] | None = None,
    ) -> str:
        """Construct structured metadata and text context for the agent prompt.

        Args:
            document_id: Optional primary document ID.
            filename: Optional original document filename.
            document_type: Optional classification type or category hint.
            document_text: Optional full document text.
            document_ids: Optional collection of document IDs available for retrieval.

        Returns:
            str: Standardized document context header.
        """
        context_lines: list[str] = []

        context_lines.append(f"Document ID: {document_id or 'Not provided'}")
        context_lines.append(f"Filename: {filename or 'Not provided'}")
        context_lines.append(f"Document Type Hint: {document_type or 'Not provided'}")

        if document_ids:
            doc_id_list = ", ".join(document_ids)
            context_lines.append(f"Available Document IDs for Retrieval: {doc_id_list}")
        else:
            context_lines.append("Available Document IDs for Retrieval: Not provided (search all indexed documents)")

        if document_text and document_text.strip():
            context_lines.append(f"Full Document Text:\n{document_text.strip()}")
        else:
            context_lines.append("Full Document Text: Not provided")

        return "\n".join(context_lines)

    def run(
        self,
        question: str,
        document_id: str | None = None,
        filename: str | None = None,
        document_type: str | None = None,
        document_text: str | None = None,
        document_ids: list[str] | None = None,
    ) -> str:
        """Execute the agent loop to analyze document context and answer user query.

        Args:
            question: Natural language question or instruction.
            document_id: Optional primary document ID.
            filename: Optional filename of primary document.
            document_type: Optional document type hint.
            document_text: Optional raw or formatted text of the document.
            document_ids: Optional list of document IDs to scope vector retrieval.

        Returns:
            str: Grounded textual answer or structured JSON result generated by the agent.

        Raises:
            ValueError: If input question is empty or document_ids structure is invalid.
            RuntimeError: If the agent execution loop fails.
        """
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question must be a non-empty string.")

        if document_ids is not None:
            if not isinstance(document_ids, list):
                raise ValueError("document_ids must be a list of strings if provided.")
            if not all(isinstance(d, str) for d in document_ids):
                raise ValueError("Every document_id must be a string.")

        clean_question = question.strip()
        doc_context = self._build_document_context(
            document_id=document_id,
            filename=filename,
            document_type=document_type,
            document_text=document_text,
            document_ids=document_ids,
        )

        logger.info(
            "DocumentAgent starting run (query_len=%d, doc_ids_count=%d, has_doc_text=%s, filename=%s)",
            len(clean_question),
            len(document_ids) if document_ids else 0,
            bool(document_text and document_text.strip()),
            filename,
        )

        try:
            result = self.executor.invoke(
                {
                    "input": clean_question,
                    "document_context": doc_context,
                }
            )
        except ValueError:
            raise
        except Exception as exc:
            logger.exception("Document agent execution encountered an unhandled failure.")
            raise RuntimeError("Document agent execution failed.") from exc

        output = result.get("output")
        if not output or not str(output).strip():
            raise RuntimeError("Document agent returned an empty response.")

        final_output = str(output).strip()
        logger.info("DocumentAgent run completed successfully.")
        return final_output


__all__ = ["DocumentAgent"]
