import re

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

# ==========================================================
# Answer Formatting
# ==========================================================
# The LLM sometimes writes server lists straight from the runbook PDF
# using a "•" bullet character and inline "Set N" / "Group N" labels,
# e.g.: "Set 1 • SHNUANPRNDQ102 • SHNUANPRNDQ101 Set 3 • SHNUANPLPAP101"
# all on one line. Standard Markdown treats a single line break as a
# soft break (just a space), not a new line, and "•" isn't a Markdown
# list marker — so Streamlit renders the whole thing as one run-on
# paragraph. This converts it into real Markdown before display:
# each "Set N"/"Group N" becomes its own bold header line, and each
# "• item" becomes its own "- item" Markdown list entry.


def format_answer_for_display(answer: str) -> str:
    # Walk through the text once, splitting on runbook section labels
    # ("Set N", "Group N", "Reboot Group N", "Test - Old", "Test - New",
    # "Prod - Old", "Prod - New", "Old Test servers", "New Test servers")
    # and "•" bullet markers wherever they occur — including when a
    # label is glued directly onto the previous item's text with no
    # bullet or line break separating them (e.g. "...AP102 Test - New").
    # A single-pass tokenizer avoids the ordering issues that sequential
    # regex substitutions ran into. The LLM is also now prompted to
    # place labels correctly on its own, so this mainly acts as a
    # safety net for anything it still gets wrong.
    token_pattern = re.compile(
        r"(Reboot\s+Group\s+\d+"
        r"|Set\s+\d+"
        r"|Group\s+\d+"
        r"|(?:Test|Prod)\s*-\s*(?:Old|New)"
        r"|Old\s+Test\s+servers"
        r"|New\s+Test\s+servers)"
        r"|•"
    )

    pieces = []
    last_end = 0
    for m in token_pattern.finditer(answer):
        pieces.append(answer[last_end:m.start()])
        if m.group(1):
            pieces.append(f"\n\n**{m.group(1)}**\n")
        else:
            pieces.append("\n- ")
        last_end = m.end()
    pieces.append(answer[last_end:])

    text = "".join(pieces)
    text = re.sub(r"- +", "- ", text)          # collapse extra spaces after bullets
    text = re.sub(r"[ \t]+\n", "\n", text)      # trim trailing spaces on each line
    text = re.sub(r"\n{3,}", "\n\n", text)      # collapse runs of blank lines

    return text.strip()


# ==========================================================
# Page Configuration  (must be first Streamlit call)
# ==========================================================

st.set_page_config(
    page_title="SCCM AI Assistant",
    page_icon="🖥️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# Custom CSS
# ==========================================================

st.markdown("""
<style>
    /* ── App background ── */
    .stApp { background-color: #0f1117; color: #ffffff; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background-color: #0d1b3e !important;
        border-right: 1px solid #1e3a6e;
    }

    [data-testid="stSidebar"] h2 {
        color: #ffffff !important;
        font-weight: 700 !important;
    }

    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] small,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div {
        color: #cbd5e1 !important;
    }

    /* ── Sample question buttons ── */
    [data-testid="stSidebar"] .stButton > button {
        background-color: #1e3a6e !important;
        color: #ffffff !important;
        border: 1px solid #3b82f6 !important;
        border-radius: 8px !important;
        font-size: 0.82rem !important;
        font-weight: 500 !important;
        padding: 8px 12px !important;
        width: 100% !important;
        margin-bottom: 4px !important;
    }

    [data-testid="stSidebar"] .stButton > button:hover {
        background-color: #2563eb !important;
        color: #ffffff !important;
    }

    /* ── Clear chat button ── */
    [data-testid="stSidebar"] .stButton > button[kind="secondary"] {
        background-color: #7f1d1d !important;
        color: #fecaca !important;
        border: 1px solid #ef4444 !important;
    }

    /* ── Chat bubbles ── */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        background-color: #1e3a6e;
        border-radius: 12px;
        padding: 4px;
        margin-bottom: 8px;
    }

    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
        background-color: #131929;
        border-left: 3px solid #3b82f6;
        border-radius: 12px;
        padding: 4px;
        margin-bottom: 8px;
    }

    /* ── Answer text — bright white ── */
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li,
    [data-testid="stChatMessage"] ol li,
    [data-testid="stChatMessage"] ul li,
    [data-testid="stChatMessage"] span,
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stMarkdownContainer"] ol li,
    [data-testid="stMarkdownContainer"] ul li,
    [data-testid="stMarkdownContainer"] span,
    [data-testid="stMarkdownContainer"] ol,
    [data-testid="stMarkdownContainer"] ul {
        color: #ffffff !important;
        font-size: 0.95rem !important;
        line-height: 1.8 !important;
    }

    /* ── Bold text in answers — yellow so it pops ── */
    [data-testid="stChatMessage"] strong,
    [data-testid="stMarkdownContainer"] strong {
        color: #facc15 !important;
        font-weight: 700 !important;
    }

    /* ── Inline code ── */
    [data-testid="stChatMessage"] code,
    [data-testid="stMarkdownContainer"] code {
        background-color: #1e3a5f !important;
        color: #7dd3fc !important;
        padding: 2px 6px !important;
        border-radius: 4px !important;
    }

    /* ── Header ── */
    .main-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #1d4ed8 100%);
        padding: 20px 24px;
        border-radius: 12px;
        margin-bottom: 20px;
        border: 1px solid #2563eb;
    }
    .main-header h1 { color: #ffffff; font-size: 1.6rem; font-weight: 700; margin: 0; }
    .main-header p  { color: #bfdbfe; font-size: 0.875rem; margin: 6px 0 0 0; }

    .status-badge {
        display: inline-block;
        background-color: #065f46;
        color: #6ee7b7;
        font-size: 0.75rem;
        padding: 3px 10px;
        border-radius: 20px;
        margin-top: 8px;
        font-weight: 600;
    }

    .sidebar-section {
        color: #60a5fa !important;
        font-size: 0.75rem !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        margin: 16px 0 8px 0 !important;
    }

    /* ── Metric cards ── */
    [data-testid="stMetric"] {
        background-color: #1e2130;
        border-radius: 8px;
        padding: 8px 12px;
        border: 1px solid #2d3748;
    }
    [data-testid="stMetricLabel"] { color: #94a3b8 !important; font-size: 0.75rem !important; }
    [data-testid="stMetricValue"] { color: #60a5fa !important; font-size: 1.2rem !important; }
</style>
""", unsafe_allow_html=True)

# ==========================================================
# Load RAG Backend  (cached so it only runs once per session)
# ==========================================================

@st.cache_resource(show_spinner=False)
def load_backend():
    """Import and initialise the RAG pipeline from app.py."""
    import app                        # triggers all loading code in app.py
    return app.get_rag_chain()        # returns (rag_chain, doc_count)

# ==========================================================
# Sidebar
# ==========================================================

SAMPLE_QUESTIONS = [
    "What are security updates in SCCM?",
    "How to configure a software update point?",
    "How to create an Automatic Deployment Rule?",
    "What is CMPivot used for?",
    "How to set up PXE deployment?",
    "How to create a device collection?",
    "What is the content library?",
    "How to configure maintenance windows?",
]

with st.sidebar:
    st.markdown("## 🖥️ SCCM AI Assistant")
    st.markdown("---")

    st.markdown('<div class="sidebar-section">About</div>', unsafe_allow_html=True)
    st.markdown("""
    <small>
    Powered by <b>LLaMA 3.3 70B</b> via Groq with
    hybrid RAG retrieval (BM25 + Semantic Search)
    over official Microsoft SCCM documentation.
    </small>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="sidebar-section">💡 Sample Questions</div>', unsafe_allow_html=True)

    for question in SAMPLE_QUESTIONS:
        if st.button(question, key=question, use_container_width=True):
            st.session_state["suggested_question"] = question

    st.markdown("---")

    if st.button("🗑️ Clear Chat History", use_container_width=True, type="secondary"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()

    st.markdown("---")
    st.markdown(
        '<small style="color:#64748b;">Data sourced from Microsoft Learn SCCM Docs</small>',
        unsafe_allow_html=True
    )

# ==========================================================
# Header
# ==========================================================

st.markdown("""
<div class="main-header">
    <h1>🖥️ SCCM AI Assistant</h1>
    <p>Ask anything about System Center Configuration Manager</p>
    <span class="status-badge">● Online</span>
</div>
""", unsafe_allow_html=True)

# ==========================================================
# Initialise Pipeline
# ==========================================================

with st.spinner("🔄 Loading SCCM knowledge base... (first load may take a few minutes)"):
    try:
        rag_chain, doc_count = load_backend()
    except Exception as e:
        st.error(f"❌ Failed to load pipeline: {e}")
        st.stop()

col1, col2, col3 = st.columns(3)
col1.metric("📄 Knowledge Chunks", doc_count)
col2.metric("🤖 Model", "LLaMA 3.3 70B")
col3.metric("🔍 Retrieval", "Hybrid RAG")

st.markdown("---")

# ==========================================================
# Session State
# ==========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []       # [{role, content}]  — for display

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []   # [HumanMessage / AIMessage] — for RAG

# ==========================================================
# Render Existing Messages
# ==========================================================

for msg in st.session_state.messages:
    avatar = "👤" if msg["role"] == "user" else "🖥️"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# ==========================================================
# Determine Prompt  (typed input or sidebar button click)
# ==========================================================

prompt = st.session_state.pop("suggested_question", None) or st.chat_input(
    "Ask an SCCM question..."
)

# ==========================================================
# Handle Prompt
# ==========================================================

if prompt:
    # Show user bubble
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate and show assistant answer
    with st.chat_message("assistant", avatar="🖥️"):
        with st.spinner("Searching SCCM documentation..."):
            try:
                import app
                answer = app.ask(prompt, st.session_state.chat_history)
                display_answer = format_answer_for_display(answer)
            except Exception as e:
                answer = f"❌ Error: {e}"
                display_answer = answer
        st.markdown(display_answer)

    # Persist to session state
    st.session_state.messages.append({"role": "assistant", "content": display_answer})
    st.session_state.chat_history.extend([
        HumanMessage(content=prompt),
        AIMessage(content=answer),
    ])