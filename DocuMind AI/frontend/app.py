"""Streamlit Frontend application for DocuMind AI.

This module provides a web interface for document ingestion,
vector indexing, RAG conversational Q&A with citations, structured extraction,
two-document comparison, and autonomous agent-based analysis by communicating
exclusively with the FastAPI backend endpoints.
"""

from typing import Any
import requests
import streamlit as st

# =====================================================================
# Page Configuration & Constants
# =====================================================================

st.set_page_config(
    page_title="DocuMind AI",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEFAULT_API_URL = "http://localhost:8000"
SUPPORTED_FILE_TYPES = ["pdf", "png", "jpg", "jpeg"]
REQUEST_TIMEOUT_SECONDS = 120

# =====================================================================
# Session State Initialization
# =====================================================================

if "api_url" not in st.session_state:
    st.session_state.api_url = DEFAULT_API_URL

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "last_document_id" not in st.session_state:
    st.session_state.last_document_id = None

if "last_filename" not in st.session_state:
    st.session_state.last_filename = None


# =====================================================================
# API Helper Functions
# =====================================================================

def _build_url(endpoint: str) -> str:
    """Construct full API URL from base API setting and endpoint path."""
    base = st.session_state.api_url.rstrip("/")
    path = endpoint.lstrip("/")
    return f"{base}/{path}"


def api_get(endpoint: str) -> tuple[bool, dict[str, Any] | str]:
    """Perform a GET request against the DocuMind API.

    Returns:
        tuple[bool, dict | str]: (is_success, response_json_or_error_message)
    """
    url = _build_url(endpoint)
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return True, response.json()
        return False, f"API returned status code {response.status_code}: {response.text}"
    except requests.exceptions.ConnectionError:
        return False, "Unable to connect to the DocuMind API. Please ensure the backend server is running."
    except requests.exceptions.Timeout:
        return False, "API request timed out. Please try again."
    except Exception as exc:
        return False, f"Unexpected request error: {exc}"


def api_post_json(endpoint: str, payload: dict[str, Any]) -> tuple[bool, dict[str, Any] | str]:
    """Perform a JSON POST request against the DocuMind API.

    Returns:
        tuple[bool, dict | str]: (is_success, response_json_or_error_message)
    """
    url = _build_url(endpoint)
    try:
        response = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT_SECONDS)
        if response.status_code in (200, 201):
            return True, response.json()

        detail = _extract_error_detail(response)
        return False, detail
    except requests.exceptions.ConnectionError:
        return False, "Unable to connect to the DocuMind API. Please verify the API base URL."
    except requests.exceptions.Timeout:
        return False, "The request took too long and timed out. Try reducing retrieval top_k or query size."
    except Exception as exc:
        return False, f"Request failed: {exc}"


def api_post_file(
    endpoint: str,
    file_bytes: bytes,
    filename: str,
    form_data: dict[str, Any] | None = None,
) -> tuple[bool, dict[str, Any] | str]:
    """Send a single file and optional form data to a multipart/form-data endpoint.

    Returns:
        tuple[bool, dict | str]: (is_success, response_json_or_error_message)
    """
    url = _build_url(endpoint)
    files = {"file": (filename, file_bytes)}
    data = form_data or {}

    try:
        response = requests.post(
            url,
            files=files,
            data=data,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        if response.status_code in (200, 201):
            return True, response.json()

        detail = _extract_error_detail(response)
        return False, detail
    except requests.exceptions.ConnectionError:
        return False, "Unable to reach the DocuMind backend server. Is FastAPI running?"
    except requests.exceptions.Timeout:
        return False, "Document processing timed out. Large files or OCR scans may require additional time."
    except Exception as exc:
        return False, f"File upload failed: {exc}"


def api_post_multiple_files(
    endpoint: str,
    files_dict: dict[str, tuple[str, bytes]],
    form_data: dict[str, Any] | None = None,
) -> tuple[bool, dict[str, Any] | str]:
    """Send multiple uploaded files to a multipart/form-data endpoint.

    Returns:
        tuple[bool, dict | str]: (is_success, response_json_or_error_message)
    """
    url = _build_url(endpoint)
    data = form_data or {}

    try:
        response = requests.post(
            url,
            files=files_dict,
            data=data,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        if response.status_code in (200, 201):
            return True, response.json()

        detail = _extract_error_detail(response)
        return False, detail
    except requests.exceptions.ConnectionError:
        return False, "Unable to reach the DocuMind backend server. Please verify connection."
    except requests.exceptions.Timeout:
        return False, "Document comparison timed out. Please try again with smaller documents."
    except Exception as exc:
        return False, f"Comparison request failed: {exc}"


def _extract_error_detail(response: requests.Response) -> str:
    """Extract a user-friendly error string from a FastAPI error response."""
    try:
        err_json = response.json()
        if isinstance(err_json, dict) and "detail" in err_json:
            detail = err_json["detail"]
            if isinstance(detail, list):
                # FastAPI validation error list
                messages = [f"{item.get('loc', ['field'])[-1]}: {item.get('msg', 'Invalid')}" for item in detail]
                return "; ".join(messages)
            return str(detail)
        return str(err_json)
    except Exception:
        return f"HTTP {response.status_code}: {response.text.strip() or 'Server Error'}"


def _format_size(size_bytes: int) -> str:
    """Format file size in bytes to human-readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.2f} MB"


# =====================================================================
# Sidebar Layout
# =====================================================================

with st.sidebar:
    st.title("⚙️ Configuration")

    st.subheader("API Connection")
    api_url_input = st.text_input(
        "FastAPI Base URL",
        value=st.session_state.api_url,
        help="Base address where the DocuMind FastAPI server is hosted.",
    )
    st.session_state.api_url = api_url_input.strip() or DEFAULT_API_URL

    if st.button("Check API Health", use_container_width=True):
        with st.spinner("Checking health..."):
            success, result = api_get("/health")
            if success and isinstance(result, dict) and result.get("status") == "healthy":
                st.success("🟢 API Healthy")
            else:
                st.error("🔴 API Unavailable")
                if isinstance(result, str):
                    st.caption(result)

    st.divider()

    st.subheader("About DocuMind AI")
    st.markdown(
        """
        **DocuMind AI** is an intelligent document analysis and agent platform providing:
        - 📄 **Document Ingestion & OCR**: Multimodal PDF and image processing.
        - ⚡ **FAISS Vector Indexing**: Persistent semantic vector embeddings.
        - 💬 **Grounded RAG Q&A**: Strict citation-backed factual Q&A.
        - 🔍 **Structured Extraction**: Automated key-value entity parsing.
        - ⚖️ **Document Comparison**: Automated discrepancy & field matching.
        - 🤖 **Autonomous Agent**: Tool-calling reasoning across collections.
        """
    )

    st.divider()
    if st.session_state.last_document_id:
        st.caption(f"📌 Last Indexed Document: `{st.session_state.last_filename}`")
        st.caption(f"ID: `{st.session_state.last_document_id}`")


# =====================================================================
# Main Header
# =====================================================================

st.title("📄 DocuMind AI")
st.markdown("#### *Intelligent Document Analysis & Agent Platform*")
st.write(
    "Upload documents, ask grounded questions, extract structured information, "
    "compare documents, and analyze them using an autonomous AI agent."
)
st.write("")

# =====================================================================
# Navigation Tabs
# =====================================================================

tab_upload, tab_chat, tab_extract, tab_compare, tab_agent = st.tabs(
    [
        "📥 Upload & Index",
        "💬 Ask Documents",
        "🔍 Extract",
        "⚖️ Compare",
        "🤖 Agent",
    ]
)


# =====================================================================
# Tab 1: Upload & Index
# =====================================================================

with tab_upload:
    st.subheader("Upload & Index Document")
    st.write("Upload a PDF or image document (PNG, JPG, JPEG) to run OCR, normalize text, and index chunks in FAISS.")

    uploaded_file = st.file_uploader(
        "Choose a document to ingest",
        type=SUPPORTED_FILE_TYPES,
        key="uploader_single",
        help="Supported formats: PDF, PNG, JPG, JPEG",
    )

    if uploaded_file is not None:
        file_details_col1, file_details_col2 = st.columns(2)
        with file_details_col1:
            st.info(f"**Filename:** {uploaded_file.name}")
        with file_details_col2:
            st.info(f"**Size:** {_format_size(uploaded_file.size)}")

        if st.button("Process & Index Document", type="primary", use_container_width=True):
            with st.spinner("Processing document, extracting text, generating embeddings, and updating index..."):
                file_bytes = uploaded_file.getvalue()
                success, response = api_post_file(
                    endpoint="/documents/upload",
                    file_bytes=file_bytes,
                    filename=uploaded_file.name,
                )

                if success and isinstance(response, dict):
                    st.success("✅ Document indexed successfully!")

                    doc_id = response.get("document_id", "N/A")
                    doc_fn = response.get("filename", uploaded_file.name)
                    st.session_state.last_document_id = doc_id
                    st.session_state.last_filename = doc_fn

                    # Metrics display
                    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                    with m_col1:
                        st.metric("Document ID", f"{doc_id[:8]}..." if len(doc_id) > 8 else doc_id, help=doc_id)
                    with m_col2:
                        st.metric("Document Type", response.get("document_type", "N/A").upper())
                    with m_col3:
                        st.metric("Total Pages", response.get("total_pages", 1))
                    with m_col4:
                        st.metric("Chunks Indexed", response.get("chunks_indexed", 0))

                    with st.expander("View Full Indexing Metadata"):
                        st.json(response)
                else:
                    st.error("❌ Document Indexing Failed")
                    st.write(str(response))


# =====================================================================
# Tab 2: Ask Documents (RAG Chat)
# =====================================================================

with tab_chat:
    st.subheader("Ask Questions About Your Documents")
    st.write("Perform semantic retrieval and generate factually grounded answers backed by verified source citations.")

    chat_ctrl_col1, chat_ctrl_col2, chat_ctrl_col3 = st.columns([2, 1, 1])
    with chat_ctrl_col1:
        doc_filter_input = st.text_input(
            "Document ID Filter (Optional)",
            value=st.session_state.last_document_id or "",
            placeholder="e.g. 4a12bc90-..., doc-id-2 (leave blank to search all documents)",
            help="Comma-separated list of document IDs to restrict retrieval scope.",
        )
    with chat_ctrl_col2:
        top_k_val = st.selectbox("Retrieval Top K", options=[3, 4, 5, 8, 10], index=1)
    with chat_ctrl_col3:
        st.write("")
        st.write("")
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    # Display historical chat conversation
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander(f"📚 Source Citations ({len(message['sources'])})"):
                    for idx, src in enumerate(message["sources"], 1):
                        fn = src.get("filename", "Unknown Document")
                        pg = src.get("page_number")
                        page_str = f"Page {pg}" if pg is not None else "Page N/A"
                        st.markdown(f"**Citation #{idx}:** `{fn}` ({page_str})")
                        st.markdown(f"> *{src.get('content', '').strip()}*")
                        if idx < len(message["sources"]):
                            st.divider()

    # Chat question input
    user_query = st.chat_input("Ask a question about your indexed documents...")
    if user_query:
        # Display user message immediately
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Parse document IDs if specified
        doc_ids = [did.strip() for did in doc_filter_input.split(",") if did.strip()] if doc_filter_input else []

        payload = {
            "question": user_query,
            "document_ids": doc_ids,
            "top_k": top_k_val,
        }

        with st.chat_message("assistant"):
            with st.spinner("Searching document context and generating answer..."):
                success, response = api_post_json("/chat", payload)

                if success and isinstance(response, dict):
                    answer_text = response.get("answer", "No answer generated.")
                    sources_list = response.get("sources", [])

                    st.markdown(answer_text)

                    if sources_list:
                        with st.expander(f"📚 Source Citations ({len(sources_list)})"):
                            for idx, src in enumerate(sources_list, 1):
                                fn = src.get("filename", "Unknown Document")
                                pg = src.get("page_number")
                                page_str = f"Page {pg}" if pg is not None else "Page N/A"
                                st.markdown(f"**Citation #{idx}:** `{fn}` ({page_str})")
                                st.markdown(f"> *{src.get('content', '').strip()}*")
                                if idx < len(sources_list):
                                    st.divider()
                    else:
                        st.caption("No sources returned.")

                    st.session_state.chat_history.append(
                        {
                            "role": "assistant",
                            "content": answer_text,
                            "sources": sources_list,
                        }
                    )
                else:
                    err_msg = f"⚠️ Query failed: {response}"
                    st.error(err_msg)
                    st.session_state.chat_history.append({"role": "assistant", "content": err_msg})


# =====================================================================
# Tab 3: Structured Extraction
# =====================================================================

with tab_extract:
    st.subheader("Structured Document Extraction")
    st.write("Extract schema-validated key-value entities, numbers, dates, parties, and executive summaries.")

    ext_col1, ext_col2 = st.columns([3, 2])
    with ext_col1:
        extract_file = st.file_uploader(
            "Choose a document for extraction",
            type=SUPPORTED_FILE_TYPES,
            key="uploader_extract",
        )
    with ext_col2:
        type_options = ["Auto Detect", "Invoice", "Contract", "Agreement", "Receipt", "Form", "Other"]
        selected_type = st.selectbox(
            "Document Type Hint",
            options=type_options,
            help="Optional classification hint to guide extraction.",
        )

    if extract_file is not None:
        st.info(f"Selected: **{extract_file.name}** ({_format_size(extract_file.size)})")

        if st.button("Extract Structured Information", type="primary", use_container_width=True):
            with st.spinner("Analyzing document layout and extracting entities..."):
                form_data = {}
                if selected_type and selected_type != "Auto Detect":
                    form_data["document_type"] = selected_type.lower()

                success, response = api_post_file(
                    endpoint="/documents/extract",
                    file_bytes=extract_file.getvalue(),
                    filename=extract_file.name,
                    form_data=form_data,
                )

                if success and isinstance(response, dict):
                    st.success("✅ Extraction Completed")

                    # Document summary card
                    st.markdown("### 📋 Executive Summary")
                    st.info(response.get("summary", "No summary provided."))

                    # Category and Metadata
                    doc_category = response.get("document_type") or "Unclassified"
                    fields = response.get("fields", [])

                    col_type, col_count = st.columns(2)
                    with col_type:
                        st.metric("Detected Document Type", doc_category.replace("_", " ").title())
                    with col_count:
                        st.metric("Extracted Entities", len(fields))

                    st.markdown("### 🏷️ Extracted Fields")
                    if fields:
                        formatted_rows = []
                        for f in fields:
                            conf = f.get("confidence", 1.0)
                            formatted_rows.append(
                                {
                                    "Field Name": f.get("field_name", "").replace("_", " ").title(),
                                    "Extracted Value": str(f.get("value", "")),
                                    "Confidence": f"{conf * 100:.1f}%" if isinstance(conf, (int, float)) else str(conf),
                                    "Source Page": f.get("source_page") or "N/A",
                                }
                            )
                        st.dataframe(formatted_rows, use_container_width=True)
                    else:
                        st.warning("No structured fields were identified in this document.")

                    with st.expander("View Raw JSON Output"):
                        st.json(response)
                else:
                    st.error(f"Extraction failed: {response}")


# =====================================================================
# Tab 4: Compare Documents
# =====================================================================

with tab_compare:
    st.subheader("Compare Two Documents")
    st.write("Compare two documents side-by-side to detect discrepancies across financial, legal, and operational attributes.")

    cmp_col1, cmp_col2 = st.columns(2)
    with cmp_col1:
        st.markdown("##### 📄 Document A (Base Document)")
        file_a = st.file_uploader("Upload Document A", type=SUPPORTED_FILE_TYPES, key="uploader_cmp_a")
        if file_a:
            st.caption(f"Selected: **{file_a.name}** ({_format_size(file_a.size)})")

    with cmp_col2:
        st.markdown("##### 📄 Document B (Comparison Target)")
        file_b = st.file_uploader("Upload Document B", type=SUPPORTED_FILE_TYPES, key="uploader_cmp_b")
        if file_b:
            st.caption(f"Selected: **{file_b.name}** ({_format_size(file_b.size)})")

    if file_a is not None and file_b is not None:
        if st.button("Run Document Comparison", type="primary", use_container_width=True):
            with st.spinner("Extracting entities from both documents and checking for mismatches..."):
                files_payload = {
                    "file1": (file_a.name, file_a.getvalue()),
                    "file2": (file_b.name, file_b.getvalue()),
                }

                success, response = api_post_multiple_files(
                    endpoint="/documents/compare",
                    files_dict=files_payload,
                )

                if success and isinstance(response, dict):
                    st.success("✅ Comparison Completed")

                    st.markdown("### 📊 Comparison Summary")
                    has_mismatches = response.get("has_mismatches", False)
                    summary_text = response.get("summary", "")

                    if has_mismatches:
                        st.warning(f"⚠️ **Discrepancies Detected:** {summary_text}")
                    else:
                        st.success(f"✨ **Full Alignment:** {summary_text}")

                    fields_list = response.get("fields", [])
                    total_fields = len(fields_list)
                    matches = [f for f in fields_list if f.get("match") is True]
                    mismatches = [f for f in fields_list if f.get("match") is False]

                    c_m1, c_m2, c_m3 = st.columns(3)
                    with c_m1:
                        st.metric("Total Fields", total_fields)
                    with c_m2:
                        st.metric("Matching Fields", len(matches))
                    with c_m3:
                        st.metric("Mismatches / Missing", len(mismatches))

                    st.markdown("### 🔍 Detailed Field Comparison")
                    if fields_list:
                        table_data = []
                        for item in fields_list:
                            is_match = item.get("match", False)
                            val_a = item.get("document_a_value")
                            val_b = item.get("document_b_value")

                            if is_match:
                                status_tag = "✅ MATCH"
                            elif val_a is None or val_b is None:
                                status_tag = "⚠️ MISSING"
                            else:
                                status_tag = "❌ MISMATCH"

                            table_data.append(
                                {
                                    "Field": item.get("field_name", "").replace("_", " ").title(),
                                    f"Document A ({response.get('document_a', 'Doc A')})": str(val_a if val_a is not None else "(Not present)"),
                                    f"Document B ({response.get('document_b', 'Doc B')})": str(val_b if val_b is not None else "(Not present)"),
                                    "Status": status_tag,
                                    "Details": item.get("details", ""),
                                }
                            )

                        st.dataframe(table_data, use_container_width=True)
                    else:
                        st.info("No comparative fields were extracted from the documents.")

                    with st.expander("View Comparison JSON"):
                        st.json(response)
                else:
                    st.error(f"Comparison failed: {response}")


# =====================================================================
# Tab 5: Autonomous Agent
# =====================================================================

with tab_agent:
    st.subheader("🤖 Document Analysis Agent")
    st.write(
        "Interact with an autonomous tool-calling AI agent. The agent can synthesize multi-step reasoning, "
        "execute grounded document retrieval, and perform comprehensive entity extraction."
    )

    agent_question = st.text_area(
        "Agent Instruction / Question",
        placeholder="e.g. Find the agreement date and check if the total amount matches between our contracts.",
        height=100,
    )

    with st.expander("Advanced Context Configuration (Optional)"):
        ag_doc_ids = st.text_input(
            "Document IDs (comma-separated)",
            value=st.session_state.last_document_id or "",
            placeholder="e.g. doc-id-1, doc-id-2",
        )
        ag_fn = st.text_input("Document Filename", value=st.session_state.last_filename or "", placeholder="e.g. contract.pdf")
        ag_type = st.text_input("Document Type Hint", placeholder="e.g. contract, invoice")
        ag_text = st.text_area(
            "Full Document Text (Required if asking agent to extract all structured fields)",
            placeholder="Paste raw text here if asking the agent to perform structured field extraction on unindexed text...",
            height=120,
        )

    if st.button("Run Agent Analysis", type="primary", use_container_width=True):
        if not agent_question.strip():
            st.warning("Please enter a question or instruction for the agent.")
        else:
            with st.spinner("Agent is reasoning and executing tools..."):
                doc_ids_list = [d.strip() for d in ag_doc_ids.split(",") if d.strip()] if ag_doc_ids else None
                agent_payload = {
                    "question": agent_question.strip(),
                    "document_ids": doc_ids_list,
                    "document_text": ag_text.strip() if ag_text.strip() else None,
                    "document_id": doc_ids_list[0] if doc_ids_list else None,
                    "filename": ag_fn.strip() if ag_fn.strip() else None,
                    "document_type": ag_type.strip() if ag_type.strip() else None,
                }

                success, response = api_post_json("/agent/query", agent_payload)

                if success and isinstance(response, dict):
                    st.markdown("### 💡 Agent Response")
                    st.success(response.get("answer", "No response returned."))
                else:
                    st.error(f"Agent analysis failed: {response}")
