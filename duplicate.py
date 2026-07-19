import os
import pickle

# Must be set BEFORE any langchain_community imports — WebBaseLoader reads
# this env var at module import time, so setting it later has no effect.
os.environ["USER_AGENT"] = "sccm-assistant/1.0"

from dotenv import load_dotenv

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from langchain_groq import ChatGroq

from langchain.chains import (
    create_history_aware_retriever,
    create_retrieval_chain
)

from langchain.chains.combine_documents import (
    create_stuff_documents_chain
)

from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder
)

from langchain_core.messages import HumanMessage, AIMessage

import re

# duckduckgo-search was renamed to "ddgs" — try the new package first,
# fall back to the old one if that's what's installed.
try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS

# ==========================================================
# Load Environment Variables
# ==========================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HUGGINGFACEHUB_API_KEY = os.getenv("HUGGINGFACEHUB_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env file.")
if not HUGGINGFACEHUB_API_KEY:
    raise ValueError("HUGGINGFACEHUB_API_KEY not found in .env file.")

# ==========================================================
# SCCM Knowledge Base URLs
# ==========================================================

urls = [

    "https://learn.microsoft.com/en-us/intune/configmgr/core/servers/deploy/configure/install-and-configure-management-points",

    "https://learn.microsoft.com/en-us/intune/configmgr/core/plan-design/hierarchy/plan-for-site-system-servers-and-site-system-roles",

    "https://learn.microsoft.com/en-us/intune/configmgr/core/servers/deploy/configure/define-site-boundaries-and-boundary-groups",

    "https://learn.microsoft.com/en-us/intune/configmgr/core/servers/deploy/configure/about-discovery-methods",

    "https://learn.microsoft.com/en-us/intune/configmgr/core/clients/deploy/about-client-settings",

    "https://learn.microsoft.com/en-us/intune/configmgr/core/clients/manage/inventory/introduction-to-hardware-inventory",

    "https://learn.microsoft.com/en-us/intune/configmgr/core/clients/manage/inventory/introduction-to-software-inventory",

    "https://learn.microsoft.com/en-us/intune/configmgr/compliance/deploy-use/create-configuration-items-for-windows-desktop-and-server-computers",

    "https://learn.microsoft.com/en-us/intune/configmgr/compliance/deploy-use/create-configuration-baselines",

    "https://learn.microsoft.com/en-us/intune/configmgr/core/clients/manage/collections/introduction-to-collections",

    "https://learn.microsoft.com/en-us/intune/configmgr/core/servers/manage/cmpivot",

    "https://learn.microsoft.com/en-us/intune/configmgr/osd/deploy-use/use-pxe-to-deploy-windows-over-the-network",

    "https://learn.microsoft.com/en-us/intune/configmgr/osd/get-started/manage-boot-images",

    "https://learn.microsoft.com/en-us/intune/configmgr/osd/get-started/manage-operating-system-images",

    "https://learn.microsoft.com/en-us/intune/configmgr/osd/get-started/manage-drivers",

    "https://learn.microsoft.com/en-us/intune/configmgr/apps/deploy-use/monitor-app-usage-with-software-metering",

    "https://learn.microsoft.com/en-us/intune/configmgr/core/servers/manage/introduction-to-reporting",

    "https://learn.microsoft.com/en-us/intune/configmgr/core/understand/fundamentals-of-role-based-administration",

    "https://learn.microsoft.com/en-us/intune/configmgr/core/clients/manage/collections/use-maintenance-windows",

    "https://learn.microsoft.com/en-us/intune/configmgr/core/plan-design/hierarchy/the-content-library",

    "https://learn.microsoft.com/en-us/intune/configmgr/core/servers/manage/backup-and-recovery",

    "https://learn.microsoft.com/en-us/intune/configmgr/core/servers/manage/database-replication",

    "https://learn.microsoft.com/en-us/intune/configmgr/core/plan-design/hierarchy/design-a-hierarchy-of-sites",

    "https://learn.microsoft.com/en-us/intune/configmgr/comanage/overview",

    "https://learn.microsoft.com/en-us/intune/configmgr/sum/understand/software-updates-introduction",

    "https://learn.microsoft.com/en-us/intune/configmgr/sum/plan-design/plan-for-software-updates",

    "https://learn.microsoft.com/en-us/intune/configmgr/sum/get-started/prepare-for-software-updates-management",

    "https://learn.microsoft.com/en-us/intune/configmgr/sum/get-started/install-a-software-update-point",

    "https://learn.microsoft.com/en-us/intune/configmgr/sum/get-started/configure-classifications-and-products",

    "https://learn.microsoft.com/en-us/intune/configmgr/sum/deploy-use/synchronize-software-updates",

    "https://learn.microsoft.com/en-us/intune/configmgr/sum/deploy-use/manually-deploy-software-updates",

    "https://learn.microsoft.com/en-us/intune/configmgr/sum/deploy-use/automatically-deploy-software-updates",

    "https://learn.microsoft.com/en-us/intune/configmgr/sum/deploy-use/automatic-deployment-rules",

    "https://learn.microsoft.com/en-us/intune/configmgr/sum/deploy-use/monitor-software-updates"
]

# ==========================================================
# Embedding Model
# ==========================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

CHROMA_PATH = "chroma_db"
DOCS_CACHE_PATH = "docs_cache.pkl"

# Max number of (human, ai) turn pairs kept in chat_history before older
# turns are dropped, to avoid silently exceeding the LLM's context window
# in long terminal sessions.
MAX_HISTORY_TURNS = 6

# ==========================================================
# Create or Load Vector DB
# ==========================================================

if os.path.exists(CHROMA_PATH):

    print("Loading existing Chroma DB...")

    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )

    if os.path.exists(DOCS_CACHE_PATH):

        print("Loading cached documents for BM25 Retriever...")

        with open(DOCS_CACHE_PATH, "rb") as f:
            docs = pickle.load(f)

    else:

        print("No docs cache found — scraping documents for BM25 Retriever...")

        loader = WebBaseLoader(urls)
        data = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        docs = splitter.split_documents(data)

        with open(DOCS_CACHE_PATH, "wb") as f:
            pickle.dump(docs, f)

        print("Docs cache saved")

else:

    print("Creating new vector database...")

    loader = WebBaseLoader(urls)
    data = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    docs = splitter.split_documents(data)

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )

    print("Vector DB created successfully")

    with open(DOCS_CACHE_PATH, "wb") as f:
        pickle.dump(docs, f)

    print("Docs cache saved")

# ==========================================================
# Diagnostics — confirm what actually made it into the index
# ==========================================================

_loaded_sources = sorted({d.metadata.get("source", "unknown") for d in docs})
_missing_sources = sorted(set(urls) - set(_loaded_sources))

print(f"\nTotal document chunks loaded: {len(docs)}")
print(f"Unique source URLs loaded: {len(_loaded_sources)} / {len(urls)}")

if _missing_sources:
    print("WARNING: the following URLs did not load into docs:")
    for src in _missing_sources:
        print(f"  - {src}")
else:
    print("All URLs loaded successfully.")
print()

# ==========================================================
# Retrievers
# ==========================================================

bm25_retriever = BM25Retriever.from_documents(docs)
bm25_retriever.k = 6

vector_retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 6,
        "fetch_k": 20
    }
)

hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.5, 0.5]
)

# ==========================================================
# Reranker
# ==========================================================
# Re-scores the fused BM25 + vector candidates against the query
# using a cross-encoder, then keeps only the top_n most relevant
# before they're passed to the LLM.

cross_encoder = HuggingFaceCrossEncoder(
    model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
)

reranker = CrossEncoderReranker(
    model=cross_encoder,
    top_n=4
)

reranked_retriever = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=hybrid_retriever
)

# ==========================================================
# LLM
# ==========================================================

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=GROQ_API_KEY,
    temperature=0.3
)

# ==========================================================
# Prompts
# ==========================================================

contextualize_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Given chat history and latest question,
rewrite the question as standalone question.
Do not answer."""
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ]
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are an SCCM (Microsoft Configuration Manager) Expert Assistant.

Use ONLY the provided context. Do not use outside knowledge.A

If the retrieved context does not contain enough information to answer the question completely, explicitly state that the documentation does not contain sufficient information.

Do not speculate.
Do not infer facts beyond the retrieved context.
Do not use prior knowledge.

Structure every answer as follows:

1. Start with a short 2-4 sentence conceptual overview that directly
   answers "what is X" / "what does X do" in plain language, synthesized
   from the context.
2. Follow with detailed, step-by-step or organized information (settings,
   procedures, requirements, prerequisites, etc.) pulled from the context.
3. If the context only contains procedural/reference details and no
   explicit definition, infer a concise overview from how the feature
   is used across the context, then present the details.

Formatting rules:
- Use numbered or bulleted lists for steps and settings.
- Do not repeat the same information twice.
- Do not pad the answer with information not present in the context.

If the context contains no relevant information at all, respond with
exactly:
'I could not find this information in the SCCM documentation.'

Context:
{context}
"""
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ]
)

# ==========================================================
# Chains
# ==========================================================

history_aware_retriever = create_history_aware_retriever(
    llm,
    reranked_retriever,
    contextualize_prompt
)

qa_chain = create_stuff_documents_chain(
    llm,
    qa_prompt
)

rag_chain = create_retrieval_chain(
    history_aware_retriever,
    qa_chain
)

# ==========================================================
# Live Windows Update Catalog Search (DuckDuckGo)
# ==========================================================

KB_PATTERN = re.compile(r"\bKB\s?-?\d{6,7}\b", re.IGNORECASE)


def search_windows_update_catalog(query: str, max_results: int = 5) -> list[dict]:
    """
    Run a live DuckDuckGo search restricted to catalog.update.microsoft.com
    to find KB articles matching the query.

    Args:
        query: Free-text query, e.g. "KB5034441" or "January 2026 cumulative update".
        max_results: Max number of results to return.

    Returns:
        List of dicts with 'title', 'url', 'snippet'.
    """
    search_query = f"site:catalog.update.microsoft.com {query}"
    results = []

    try:
        with DDGS() as ddgs:
            for r in ddgs.text(search_query, max_results=max_results):
                results.append(
                    {
                        "title": r.get("title", ""),
                        "url": r.get("href") or r.get("link", ""),
                        "snippet": r.get("body", ""),
                    }
                )
    except Exception as e:
        print(f"DuckDuckGo search failed: {e}")

    return results


def format_kb_results(results: list[dict]) -> str:
    """Format live search results into a readable markdown block."""
    if not results:
        return (
            "No matching KB articles were found on the Microsoft Update Catalog. "
            "Try searching manually at https://catalog.update.microsoft.com/"
        )

    lines = []
    for r in results:
        lines.append(f"- **{r['title']}**\n  {r['url']}\n  {r['snippet']}")

    return "\n\n".join(lines)


def needs_live_kb_search(query: str, rag_answer: str) -> bool:
    """
    Only trigger a live catalog search when the user is explicitly asking
    about a KB article (e.g. "KB5034441"). We deliberately do NOT trigger
    this just because the SCCM docs RAG had no answer — most SCCM
    questions (e.g. "what is software distribution?") have nothing to do
    with the Windows Update Catalog, and appending catalog results to
    those answers is just noise.
    """
    return bool(KB_PATTERN.search(query))


def search_kb_catalog(query: str, max_results: int = 5) -> str:
    """
    Standalone, UI-facing entry point for a manual 'Search KB Catalog' button
    in streamlit_app.py. Independent of the RAG chain / chat history —
    just runs a live DuckDuckGo search against the Microsoft Update Catalog
    and returns a ready-to-display markdown string.

    Example (in streamlit_app.py):
        if st.button("Search KB Catalog"):
            result_md = search_kb_catalog(kb_query)
            st.markdown(result_md)

    Args:
        query: KB number or free-text search, e.g. "KB5034441" or
               "January 2026 cumulative update Windows 11".
        max_results: Max number of results to return.

    Returns:
        Markdown-formatted string of results.
    """
    results = search_windows_update_catalog(query, max_results=max_results)
    return format_kb_results(results)


# ==========================================================
# Public API — used by streamlit_app.py
# ==========================================================

def get_rag_chain():
    """Return the RAG chain and document count for use in the frontend."""
    return rag_chain, len(docs)


def ask(query: str, chat_history: list) -> str:
    """
    Send a question to the RAG chain and return the answer string.

    Args:
        query: The user's question.
        chat_history: List of HumanMessage / AIMessage objects.

    Returns:
        The assistant's answer as a string.
    """
    response = rag_chain.invoke(
        {
            "input": query,
            "chat_history": chat_history
        }
    )
    answer = response["answer"]

    # If the query looks like a KB lookup, or the local docs had no answer,
    # fall back to a live search of the Microsoft Update Catalog.
    if needs_live_kb_search(query, answer):
        kb_results = search_windows_update_catalog(query)
        live_section = format_kb_results(kb_results)
        answer = (
            f"{answer}\n\n---\n"
            f"**Live Windows Update Catalog search:**\n\n{live_section}"
        )

    return answer


# ==========================================================
# Terminal Chat Loop (only runs when executed directly)
# ==========================================================

if __name__ == "__main__":

    chat_history = []

    print("\nSCCM AI Assistant Started")
    print("Type 'exit' to quit\n")

    while True:

        query = input("Ask SCCM Question: ")

        if query.lower() == "exit":
            print("Goodbye!")
            break

        answer = ask(query, chat_history)

        print("\nAnswer:\n")
        print(answer)

        chat_history.extend(
            [
                HumanMessage(content=query),
                AIMessage(content=answer)
            ]
        )

        print("\n" + "=" * 100)