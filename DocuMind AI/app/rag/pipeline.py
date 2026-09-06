"""RAG orchestration pipeline module for DocuMind AI.

This module orchestrates semantic retrieval, context formulation, grounded prompt
generation, LLM inference, and page-level source citation synthesis to produce
structured ChatResponse outputs.
"""

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from app.core.config import settings
from app.core.logger import get_logger
from app.models.schemas import ChatResponse, SourceCitation
from app.rag.retriever import DocumentRetriever

logger = get_logger(__name__)

SUPPORTED_LLM_PROVIDERS = {"openai"}

SYSTEM_PROMPT = """You are DocuMind AI, an intelligent and precise document analysis assistant.

Your task is to answer the user's question strictly using ONLY the provided document context below.

Rules and Constraints:
1. Grounding: Answer ONLY using the facts directly stated in the supplied document context. Do NOT use outside knowledge, prior assumptions, or extrapolate beyond what is written.
2. Factuality: Accurately preserve all numbers, dates, currency amounts, percentages, names, IDs, addresses, and legal terms exactly as they appear in the text.
3. Insufficient Context: If the supplied context does not contain enough information to answer the question, clearly and concisely state: "The provided documents do not contain sufficient information to answer this question." Do not attempt to guess or fabricate an answer.
4. Multi-Document Clarity: When the context contains snippets from different documents or pages, clearly identify which document and page you are referring to.
5. Tone: Be professional, direct, concise, and helpful.
6. Transparency: Never mention internal implementation mechanisms, such as embeddings, FAISS, vectors, chunk indices, prompt templates, or retrieval algorithms.
7. Citation Integrity: Do not invent or hallucinate document citations or page numbers."""

HUMAN_PROMPT = """Document Context:
{context}

Question:
{question}

Answer:"""


class RAGPipeline:
    """Orchestrates retrieval-augmented generation for document question answering."""

    def __init__(
        self,
        retriever: DocumentRetriever | None = None,
        provider: str | None = None,
        model_name: str | None = None,
        temperature: float | None = None,
    ) -> None:
        """Initialize the RAG pipeline.

        Args:
            retriever: Optional DocumentRetriever instance. Defaults to a new instance.
            provider: LLM provider name (defaults to settings.LLM_PROVIDER).
            model_name: Name of the LLM model (defaults to settings.LLM_MODEL).
            temperature: Sampling temperature for generation (defaults to settings.LLM_TEMPERATURE).

        Raises:
            ValueError: If the provider is unsupported or required API keys are missing.
        """
        self.retriever = retriever or DocumentRetriever()
        self.provider = (provider or settings.LLM_PROVIDER).lower()
        self.model_name = model_name or settings.LLM_MODEL
        self.temperature = (
            temperature if temperature is not None else settings.LLM_TEMPERATURE
        )

        if self.provider not in SUPPORTED_LLM_PROVIDERS:
            raise ValueError(
                f"Unsupported LLM provider '{self.provider}'. "
                f"Supported providers: {sorted(SUPPORTED_LLM_PROVIDERS)}"
            )

        self.llm = self._initialize_llm()
        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                ("human", HUMAN_PROMPT),
            ]
        )

        logger.info(
            "RAGPipeline initialized (provider=%s, model=%s, temperature=%.2f)",
            self.provider,
            self.model_name,
            self.temperature,
        )

    def _initialize_llm(self) -> ChatOpenAI:
        """Instantiate and configure the LLM client.

        Returns:
            ChatOpenAI: Configured LangChain chat model.

        Raises:
            ValueError: If required API credentials are missing.
        """
        if self.provider == "openai":
            api_key = settings.OPENAI_API_KEY
            if not api_key or not api_key.strip():
                raise ValueError(
                    "OPENAI_API_KEY is not configured. Please set OPENAI_API_KEY in your "
                    "environment or .env file to initialize the RAG pipeline."
                )

            return ChatOpenAI(
                model=self.model_name,
                temperature=self.temperature,
                api_key=api_key,
            )

        raise ValueError(f"Unhandled LLM provider: {self.provider}")

    def _build_context(self, scored_documents: list[tuple[Document, float]]) -> str:
        """Format retrieved document chunks into a structured context string for the LLM.

        Args:
            scored_documents: List of retrieved (Document, score) tuples.

        Returns:
            str: Consolidated, labeled context string.
        """
        context_blocks: list[str] = []

        for doc, _ in scored_documents:
            filename = doc.metadata.get("filename", "Unknown Document")
            page_number = doc.metadata.get("page_number", 1)
            content = doc.page_content.strip()

            block = f"[Document: {filename} | Page: {page_number}]\n{content}"
            context_blocks.append(block)

        return "\n\n".join(context_blocks)

    def _build_sources(
        self, scored_documents: list[tuple[Document, float]]
    ) -> list[SourceCitation]:
        """Convert retrieved document chunks into deduplicated SourceCitation objects.

        Args:
            scored_documents: List of retrieved (Document, score) tuples.

        Returns:
            list[SourceCitation]: List of deduplicated source citation models.
        """
        citations: list[SourceCitation] = []
        seen_keys: set[tuple[str, int | None, str]] = set()

        for doc, _ in scored_documents:
            doc_id = str(doc.metadata.get("document_id", "unknown_id"))
            filename = str(doc.metadata.get("filename", "Unknown Document"))
            page_num = doc.metadata.get("page_number")
            content = doc.page_content.strip()

            # Deduplicate by document_id, page_number, and exact snippet content
            citation_key = (doc_id, page_num, content)
            if citation_key in seen_keys:
                continue

            seen_keys.add(citation_key)
            citations.append(
                SourceCitation(
                    document_id=doc_id,
                    filename=filename,
                    page_number=page_num,
                    content=content,
                )
            )

        return citations

    def answer(
        self,
        question: str,
        top_k: int | None = None,
        document_ids: list[str] | None = None,
    ) -> ChatResponse:
        """Answer a user query by retrieving relevant document chunks and generating a grounded response.

        Args:
            question: Natural language user query.
            top_k: Optional number of chunks to retrieve.
            document_ids: Optional list of document IDs to restrict retrieval scope.

        Returns:
            ChatResponse: Structured response with generated answer and source citations.

        Raises:
            ValueError: If question is invalid.
            RuntimeError: If retrieval or LLM generation fails.
        """
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question must be a non-empty string")

        logger.info("Processing RAG question (top_k=%s, doc_filter=%s)", top_k, bool(document_ids))

        try:
            scored_chunks = self.retriever.retrieve_with_scores(
                query=question.strip(),
                top_k=top_k,
                document_ids=document_ids,
            )

            if not scored_chunks:
                logger.info("No relevant document chunks retrieved. Returning fallback response.")
                return ChatResponse(
                    answer="I could not find any relevant information in the provided documents to answer your question.",
                    sources=[],
                )

            context_str = self._build_context(scored_chunks)
            sources = self._build_sources(scored_chunks)

            logger.info(
                "Generating LLM response using %d retrieved chunk(s) and %d citation(s)...",
                len(scored_chunks),
                len(sources),
            )

            chain = self.prompt_template | self.llm
            result = chain.invoke(
                {
                    "context": context_str,
                    "question": question.strip(),
                }
            )

            # Safely extract message content
            answer_text = (
                result.content if hasattr(result, "content") else str(result)
            )

            logger.info("RAG answer generation completed successfully.")

            return ChatResponse(
                answer=str(answer_text).strip(),
                sources=sources,
            )

        except (ValueError, FileNotFoundError):
            raise
        except Exception as exc:
            logger.error("RAG pipeline execution failed: %s", exc)
            raise RuntimeError(f"RAG pipeline execution failed: {exc}") from exc
