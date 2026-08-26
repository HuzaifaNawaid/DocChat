import streamlit as st
import hashlib
import os

from document_processor.file_handler import DocumentProcessor
from retriever.builder import RetrieverBuilder
from agents.workflow import AgentWorkflow
from config import constants
from utils.logging import logger

# 1) Example data (same as original)
EXAMPLES = {
    "Google 2024 Environmental Report": {
        "question": "Retrieve the data center PUE efficiency values in Singapore 2nd facility in 2019 and 2022. Also retrieve regional average CFE in Asia pacific in 2023",
        "file_paths": ["examples/google-2024-environmental-report.pdf"]
    },
    "DeepSeek-R1 Technical Report": {
        "question": "Summarize DeepSeek-R1 model's performance evaluation on all coding tasks against OpenAI o1-mini model",
        "file_paths": ["examples/DeepSeek Technical Report.pdf"]
    }
}


# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
def inject_css():
    st.markdown(
        """
        <style>
            /* ---- Global (dark / black & red theme) ---- */
            html, body, [class*="css"] {
                font-family: 'Inter', 'Segoe UI', sans-serif;
            }
            .stApp {
                background: radial-gradient(circle at top left, #1a0000 0%, #0d0d0d 45%, #050505 100%);
                color: #f2e9e9;
            }
            #MainMenu, footer {visibility: hidden;}
            header[data-testid="stHeader"] {
                background: transparent;
            }

            h1, h2, h3, h4, h5, h6, p, span, label, .stMarkdown {
                color: #f2e9e9;
            }

            /* ---- Hero header ---- */
            .docchat-hero {
                background: linear-gradient(120deg, #1a0000 0%, #3d0000 45%, #7a0e0e 100%);
                border-radius: 20px;
                padding: 2.2rem 2.4rem;
                margin-bottom: 1.6rem;
                box-shadow: 0 10px 30px rgba(200, 0, 0, 0.25);
                border: 1px solid rgba(220, 40, 40, 0.35);
                color: #fdf2f2;
            }
            .docchat-hero h1 {
                font-size: 2.1rem;
                font-weight: 800;
                margin: 0 0 0.35rem 0;
                letter-spacing: -0.5px;
                color: #ff3b3b;
                text-shadow: 0 0 18px rgba(255, 59, 59, 0.35);
            }
            .docchat-hero p {
                font-size: 1.02rem;
                opacity: 0.92;
                margin: 0;
                color: #f2e0e0;
            }
            .docchat-steps {
                display: flex;
                gap: 0.9rem;
                margin-top: 1.1rem;
                flex-wrap: wrap;
            }
            .docchat-step {
                background: rgba(255, 59, 59, 0.1);
                border: 1px solid rgba(255, 59, 59, 0.35);
                border-radius: 12px;
                padding: 0.55rem 0.9rem;
                font-size: 0.87rem;
                font-weight: 500;
                backdrop-filter: blur(6px);
                color: #ffd6d6;
            }

            /* ---- Cards ---- */
            .docchat-card {
                background: #121212;
                border-radius: 16px;
                padding: 1.5rem 1.5rem 1.2rem 1.5rem;
                box-shadow: 0 4px 18px rgba(0, 0, 0, 0.5);
                border: 1px solid rgba(220, 40, 40, 0.25);
                margin-bottom: 1.2rem;
            }
            .docchat-card h3 {
                margin-top: 0;
                font-size: 1.08rem;
                font-weight: 700;
                color: #ff5c5c;
            }

            /* ---- Format badge row ---- */
            .format-badges {
                display: flex;
                gap: 0.5rem;
                flex-wrap: wrap;
                margin: 0.4rem 0 1rem 0;
            }
            .format-badge {
                background: linear-gradient(120deg, #2a0808, #3a0a0a);
                color: #ff8080;
                border: 1px solid #7a1414;
                border-radius: 999px;
                padding: 0.25rem 0.75rem;
                font-size: 0.78rem;
                font-weight: 600;
            }

            /* ---- Buttons ---- */
            div.stButton > button {
                border-radius: 10px;
                font-weight: 600;
                border: 1px solid rgba(220, 40, 40, 0.4);
                background: #1a1a1a;
                color: #f2e9e9;
                transition: transform 0.08s ease, box-shadow 0.15s ease;
            }
            div.stButton > button:hover {
                transform: translateY(-1px);
                box-shadow: 0 6px 16px rgba(220, 0, 0, 0.35);
                border-color: #ff3b3b;
                color: #ffffff;
            }
            div.stButton > button[kind="primary"] {
                background: linear-gradient(120deg, #7a0e0e, #b81414);
                color: white;
                border: none;
            }
            div.stButton > button[kind="primary"]:hover {
                background: linear-gradient(120deg, #921212, #d61a1a);
            }

            /* ---- Text areas / inputs ---- */
            .stTextArea textarea, .stSelectbox div[data-baseweb="select"] {
                border-radius: 12px !important;
                background-color: #1a1a1a !important;
                color: #f2e9e9 !important;
                border: 1px solid rgba(220, 40, 40, 0.25) !important;
            }
            .stSelectbox div[data-baseweb="select"] * {
                color: #f2e9e9 !important;
            }
            .stTextArea textarea::placeholder {
                color: #8a7070;
            }

            /* ---- File uploader ---- */
            [data-testid="stFileUploaderDropzone"] {
                border-radius: 14px;
                border: 1.5px dashed #b81414;
                background: #150808;
            }
            [data-testid="stFileUploaderDropzone"] * {
                color: #f2e9e9 !important;
            }
            [data-testid="stFileUploader"] section button {
                background-color: #2a0808 !important;
                color: #ff8080 !important;
                border: 1px solid #7a1414 !important;
            }

            /* ---- Result section ---- */
            .result-label {
                font-weight: 700;
                font-size: 0.95rem;
                color: #ff5c5c;
                display: flex;
                align-items: center;
                gap: 0.4rem;
                margin-bottom: 0.35rem;
            }

            /* ---- Divider ---- */
            .soft-divider {
                border: none;
                border-top: 1px solid rgba(220, 40, 40, 0.25);
                margin: 1.2rem 0;
            }

            /* ---- Alerts (info/warning/error) ---- */
            div[data-testid="stAlert"] {
                background-color: #1a1010 !important;
                border: 1px solid rgba(220, 40, 40, 0.3) !important;
                color: #f2e9e9 !important;
                border-radius: 12px !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def get_backend():
    """Initialize backend components once and cache them across reruns."""
    processor = DocumentProcessor()
    retriever_builder = RetrieverBuilder()
    workflow = AgentWorkflow()
    return processor, retriever_builder, workflow


def get_file_hashes(uploaded_files) -> frozenset:
    """Generate SHA-256 hashes for uploaded files (Streamlit UploadedFile objects)."""
    hashes = set()
    for file in uploaded_files:
        file.seek(0)
        hashes.add(hashlib.sha256(file.read()).hexdigest())
        file.seek(0)
    return frozenset(hashes)


def load_example(example_key: str):
    ex_data = EXAMPLES[example_key]
    question = ex_data["question"]
    file_paths = [p for p in ex_data["file_paths"] if os.path.exists(p)]
    if not file_paths:
        logger.warning(f"Example files not found for '{example_key}'")
    return file_paths, question


def main():
    st.set_page_config(page_title="DocChat", page_icon="💻", layout="wide")
    inject_css()

    processor, retriever_builder, workflow = get_backend()

    # --- Session state (replaces Gradio's gr.State) ---
    if "file_hashes" not in st.session_state:
        st.session_state.file_hashes = frozenset()
    if "retriever" not in st.session_state:
        st.session_state.retriever = None
    if "example_question" not in st.session_state:
        st.session_state.example_question = ""

    # --- Hero header ---
    st.markdown(
        """
        <div class="docchat-hero">
            <h1>💻 DocChat — Agentic RAG Document Q&A</h1>
            <p>DocChat is a multi-agent Retrieval-Augmented Generation (RAG) application that lets you upload documents (PDF, DOCX, TXT, MD) and ask questions about them. Instead of a single LLM call, DocChat routes every question through a 3-agent LangGraph pipeline — a relevance checker, a research agent, and a verification agent — combined with hybrid retrieval (keyword + semantic search) to reduce hallucinations and give grounded, verified answers.</p>
            <div class="docchat-steps">
                <div class="docchat-step">📤 1. Upload a document</div>
                <div class="docchat-step">📝 2. Ask your question</div>
                <div class="docchat-step">🚀 3. Submit &amp; get a verified answer</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("### 📂 Try an example")
        example_key = st.selectbox(
            "Select an example:",
            options=["-- none --"] + list(EXAMPLES.keys()),
            label_visibility="collapsed",
        )
        if st.button("Load Example 🛠️", use_container_width=True):
            if example_key != "-- none --":
                file_paths, question = load_example(example_key)
                st.session_state.example_question = question
                st.session_state.example_file_paths = file_paths
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("### 📄 Your documents")
        uploaded_files = st.file_uploader(
            "Upload documents",
            type=[ext.lstrip(".") for ext in constants.ALLOWED_TYPES],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

        st.markdown("### ❓ Your question")
        question_text = st.text_area(
            "Question",
            value=st.session_state.get("example_question", ""),
            height=100,
            label_visibility="collapsed",
        )

        submit = st.button("Submit 🚀", type="primary", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="result-label">🚀 Answer</div>', unsafe_allow_html=True)
        answer_placeholder = st.empty()
        if not submit:
            answer_placeholder.info("Your answer will appear here once you submit a question.")
        st.markdown('<hr class="soft-divider">', unsafe_allow_html=True)
        st.markdown('<div class="result-label">✅ Verification Report</div>', unsafe_allow_html=True)
        verification_placeholder = st.empty()
        if not submit:
            verification_placeholder.info("A verification report will appear here after processing.")
        st.markdown('</div>', unsafe_allow_html=True)

    if submit:
        try:
            if not question_text.strip():
                raise ValueError("❌ Question cannot be empty")

            # Handle example files (loaded from disk) vs uploaded files
            example_paths = st.session_state.get("example_file_paths", [])
            if example_paths and not uploaded_files:
                # Wrap file paths so processor can handle them the same way
                class _LocalFile:
                    def __init__(self, path):
                        self.name = path
                files_to_process = [_LocalFile(p) for p in example_paths]
                current_hashes = frozenset(
                    hashlib.sha256(open(p.name, "rb").read()).hexdigest() for p in files_to_process
                )
            else:
                if not uploaded_files:
                    raise ValueError("❌ No documents uploaded")
                files_to_process = uploaded_files
                current_hashes = get_file_hashes(uploaded_files)

            with st.spinner("✨ Processing document(s) and running the multi-agent pipeline..."):
                if st.session_state.retriever is None or current_hashes != st.session_state.file_hashes:
                    logger.info("Processing new/changed documents...")
                    chunks = processor.process(files_to_process)
                    retriever = retriever_builder.build_hybrid_retriever(chunks)
                    st.session_state.file_hashes = current_hashes
                    st.session_state.retriever = retriever

                result = workflow.full_pipeline(
                    question=question_text,
                    retriever=st.session_state.retriever
                )

            answer_placeholder.text_area(
                "Answer", value=result["draft_answer"], height=150, label_visibility="collapsed"
            )
            verification_placeholder.text_area(
                "Verification", value=result["verification_report"], height=200, label_visibility="collapsed"
            )
            st.toast("Done! Answer generated successfully.", icon="✅")

        except Exception as e:
            logger.error(f"Processing error: {str(e)}")
            st.error(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    main()