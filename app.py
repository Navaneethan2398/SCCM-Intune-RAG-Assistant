import os
import pickle

# Must be set BEFORE any langchain_community imports — WebBaseLoader reads
# this env var at module import time, so setting it later has no effect.
os.environ["USER_AGENT"] = "sccm-assistant/1.0"

from dotenv import load_dotenv

from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
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
from langchain_core.documents import Document

import re

from guardrails import (
    detect_prompt_injection,
    append_citations,
    BLOCK_MESSAGE
)

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
# Local PDF Documents
# ==========================================================
# These get merged into the same knowledge base as the URLs above,
# so questions about reboot sequences / server names are answerable
# by the same RAG chain. Paths are relative to the project root —
# drop the PDF file next to app.py, or use an absolute path.

PDF_PATHS = [
    "Reboot_Sequences_Formatted.pdf",
]

# ==========================================================
# Embedding Model
# ==========================================================

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    model_kwargs={"device": "cpu"},      # or "cuda" if using GPU
    encode_kwargs={"normalize_embeddings": True}
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

splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=100
)

# The PDF runbook packs many short bullet lists (Set 1, Set 2, Set 3...)
# tightly together. Splitting it with the same 800-char splitter used for
# prose web pages can cut a single application's sets across two chunks,
# so a query like "list all PowerScribe sets" only retrieves half of them.
# This splitter uses a much bigger chunk size so a full application
# section fits in one chunk whenever possible, with overlap as a
# safety net for anything that still spans a chunk boundary.
pdf_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2500,
    chunk_overlap=400
)


def load_pdf_docs(paths: list[str]) -> list:
    """
    Load one or more local PDFs into raw (unsplit) LangChain Documents.

    Each PDF's pages are merged into a single Document (rather than kept
    as one Document per page) so that a runbook section split across a
    page boundary in the original PDF isn't lost when we later chunk it
    with pdf_splitter.
    """
    raw_docs = []
    for path in paths:
        if not os.path.exists(path):
            print(f"WARNING: PDF not found, skipping: {path}")
            continue
        print(f"Loading PDF: {path}")
        loader = PyPDFLoader(path)
        pages = loader.load()
        merged_text = "\n\n".join(p.page_content for p in pages)
        raw_docs.append(Document(page_content=merged_text, metadata={"source": path}))
    return raw_docs


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
        web_data = loader.load()
        pdf_data = load_pdf_docs(PDF_PATHS)

        docs = splitter.split_documents(web_data) + pdf_splitter.split_documents(pdf_data)

        vectorstore.add_documents(docs)

        with open(DOCS_CACHE_PATH, "wb") as f:
            pickle.dump(docs, f)

        print("Docs cache saved")

    # Pick up any PDFs that were added to PDF_PATHS after the Chroma DB
    # and docs cache were already built, without re-scraping the URLs
    # or re-embedding PDFs that are already indexed.
    cached_sources = {d.metadata.get("source") for d in docs}
    new_pdf_paths = [p for p in PDF_PATHS if p not in cached_sources and os.path.exists(p)]

    if new_pdf_paths:

        print(f"Found {len(new_pdf_paths)} new PDF(s) not yet indexed — adding them...")

        new_pdf_data = load_pdf_docs(new_pdf_paths)
        new_pdf_chunks = pdf_splitter.split_documents(new_pdf_data)

        vectorstore.add_documents(new_pdf_chunks)

        docs.extend(new_pdf_chunks)

        with open(DOCS_CACHE_PATH, "wb") as f:
            pickle.dump(docs, f)

        print(f"Added {len(new_pdf_chunks)} PDF chunks to the vector DB and docs cache.")

else:

    print("Creating new vector database...")

    loader = WebBaseLoader(urls)
    web_data = loader.load()
    pdf_data = load_pdf_docs(PDF_PATHS)

    docs = splitter.split_documents(web_data) + pdf_splitter.split_documents(pdf_data)

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

_pdf_chunks_loaded = sum(1 for d in docs if d.metadata.get("source") in PDF_PATHS)

print(f"\nTotal document chunks loaded: {len(docs)}")
print(f"Unique source URLs loaded: {len(_loaded_sources)} / {len(urls)}")
print(f"PDF chunks loaded: {_pdf_chunks_loaded} (from {len(PDF_PATHS)} configured PDF file(s))")

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

# search_type="mmr" deliberately trades relevance for diversity (it wants
# results that differ from each other). That's the opposite of what
# context_precision rewards, which is "every retrieved chunk is actually
# relevant". Plain similarity search ranks purely by closeness to the
# query, which is what we want feeding into the reranker.
vector_retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 6
    }
)

hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.3, 0.7]
)

# ==========================================================
# Reranker
# ==========================================================
# Re-scores the fused BM25 + vector candidates against the query
# using a cross-encoder, then keeps only the top_n most relevant
# before they're passed to the LLM.

cross_encoder = HuggingFaceCrossEncoder(
    model_name="BAAI/bge-reranker-base"
)


class ScoringCrossEncoderReranker(CrossEncoderReranker):
   
    min_score: float = 0.0

    def compress_documents(self, documents, query, callbacks=None):
        scores = self.model.score([(query, doc.page_content) for doc in documents])
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda pair: pair[1], reverse=True)

        top_docs = []
        for doc, score in scored_docs[: self.top_n]:
            if score < self.min_score and top_docs:
                # Already have at least one relevant doc — stop padding
                # with weak ones. (We never return zero docs: the first,
                # highest-scoring doc is always kept even if it's below
                # threshold, so context_recall doesn't collapse on a
                # genuinely hard question.)
                break
            doc.metadata = {**doc.metadata, "rerank_score": float(score)}
            top_docs.append(doc)

        return top_docs


# min_score of 0.0 on BAAI/bge-reranker-base's raw logit output is a rough
# "roughly as likely relevant as not" cutoff. Log the scores from a few
# real queries (doc.metadata["rerank_score"]) and adjust this threshold
# up if you're still seeing weak chunks pass through, or down if good
# chunks are getting cut on harder questions.
reranker = ScoringCrossEncoderReranker(
    model=cross_encoder,
    top_n=3,
    min_score=0.0
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
    temperature=0.2
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

Use ONLY the provided context. Do not use outside knowledge.

If the retrieved context does not contain enough information to answer the question completely, explicitly state that the documentation does not contain sufficient information.

Do not speculate.
Do not infer facts beyond the retrieved context.
Do not use prior knowledge.

Reboot runbook rule (critical):
- Some context comes from a server reboot runbook that organizes servers
  into labeled sections such as "Set 1", "Set 2", "Group 1", "Group 2",
  "Test - Old", "Test - New", "Prod - Old", "Prod - New", "Old Test
  servers", "New Test servers", etc., without always spelling out the
  words "reboot order" or "sequence."
- If the question asks for the reboot sequence, order, or "complete
  list" for an application, and the context lists that application's
  servers grouped into these labeled sections, treat those groups as
  the answer: list every section and every server in each, in the same
  order they appear in the context. Do not say the documentation lacks
  sufficient information just because the word "order" or "sequence"
  isn't used verbatim — labeled sections presented in sequence ARE the
  reboot grouping.
- Only say the documentation does not contain sufficient information
  if the application is not mentioned in the context at all, or if no
  servers/groups are listed for it whatsoever.
- Never omit or summarize away any server names that appear under the
  application asked about — a "complete list" request means every
  server in every section, not a representative sample.
- CRITICAL LABEL PLACEMENT: every section label (e.g. "Set 3", "Test -
  New", "Prod - Old") MUST appear on its own line, in bold, immediately
  BEFORE the numbered/bulleted list of servers that belong to it, and
  AFTER a blank line. A label must never be appended to the end of the
  previous section's last server name (e.g. never write "SHIECGTRNAP102
  Test - New" — the label starts a brand new line and a brand new list).
  Each section's list restarts its own numbering or bullets from 1.

Example of correct formatting for a runbook answer:

Test - Old
1. SERVERNAME101
2. SERVERNAME102

Test - New
1. SERVERNAME103
2. SERVERNAME104

Match the answer's structure to the question actually asked:

- For "What is X?" / "What does X do?" / definitional questions: answer
  with a clear 4-7 sentence explanation covering the definition, its
  purpose, and how it is used in an SCCM environment. Do not add
  steps, settings, prerequisites, or procedures unless the question
  explicitly asks how to configure, set up, or use the feature.
- For "How do you configure/set up/deploy X?" / procedural questions:
  answer with the steps, settings, or procedure directly — skip a
  separate conceptual overview paragraph and go straight into the
  numbered steps pulled from the context.
- If the context only contains procedural/reference details and the
  question is definitional, infer a concise 2-4 sentence overview from
  how the feature is used across the context — do not also append the
  procedural details unless asked.

Formatting rules:
- Use numbered or bulleted lists for steps and settings, only when
  the question calls for steps or settings.
- Do not repeat the same information twice.
- Do not pad the answer with information not present in the context.
- Do not include information that answers a different question than
  the one asked, even if it appears in the same retrieved context.
- Do not restate or rephrase the question at the start of the answer.
- Do not include meta-commentary like "Based on the context provided"
  or "According to the documentation."

Grounding rule (critical):
- Do not state specific examples, categories, or details (e.g. types
  of content, specific settings, specific numbers) unless those exact
  specifics appear verbatim in the provided context.
- If the context is general or vague on specifics, keep your answer
  equally general. Do not fill in plausible-sounding specifics from
  outside knowledge, even if they are commonly known facts about SCCM.

Scope rule for procedural answers (critical):
- A procedural answer should normally take no more than 3-5 steps.
- Include ONLY the steps that directly accomplish the exact task asked.
  Do not include steps about configuring a different feature, setting
  up a related site system role in general, or optional/fallback
  behavior, even if that information appears in the same retrieved
  context.
- Stop the list as soon as the core task described in the question is
  complete.

  Example — question: "How do you configure Boundary Groups?"
  BAD (too broad, drifts into unrelated configuration):
    1. Define boundaries.
    2. Create a boundary group.
    3. Add boundaries to the group.
    4. Configure site systems like distribution points, management
       points, and software update points in general.
    5. Assign individual software update points to different boundary
       groups to control fallback behavior.
  GOOD (stays scoped to the exact task):
    1. Define network locations as boundaries.
    2. Create a boundary group.
    3. Add boundaries to the boundary group.
    4. Assign site systems to the boundary group.

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
    Send a question to the RAG chain and return the answer string,
    with the prompt-injection guardrail applied and source citations
    appended.

    Args:
        query: The user's question.
        chat_history: List of HumanMessage / AIMessage objects.

    Returns:
        The assistant's answer as a string, including citations.
    """
    if detect_prompt_injection(query):
        return BLOCK_MESSAGE

    response = rag_chain.invoke(
        {
            "input": query,
            "chat_history": chat_history
        }
    )
    answer = response["answer"]
    documents = response["context"]

    # If the query looks like a KB lookup, fall back to a live search
    # of the Microsoft Update Catalog.
    if needs_live_kb_search(query, answer):
        kb_results = search_windows_update_catalog(query)
        live_section = format_kb_results(kb_results)
        answer = (
            f"{answer}\n\n---\n"
            f"**Live Windows Update Catalog search:**\n\n{live_section}"
        )

    answer = append_citations(answer, documents)

    return answer

def ask_with_context(query: str, chat_history: list) -> dict:
    """
    Used only for evaluation (RAGAS). Returns both the generated
    answer and the retrieved context strings, bypassing citations and
    the KB catalog fallback since neither is relevant to evaluation
    metrics like faithfulness or context precision.

    Args:
        query: The user's question.
        chat_history: List of HumanMessage / AIMessage objects.

    Returns:
        Dict with "answer" (str) and "contexts" (list of str).
    """
    if detect_prompt_injection(query):
        return {
            "answer": BLOCK_MESSAGE,
            "contexts": []
        }

    response = rag_chain.invoke(
        {
            "input": query,
            "chat_history": chat_history
        }
    )
    answer = response["answer"]
    documents = response["context"]
    contexts = [doc.page_content for doc in documents]

    return {
        "answer": answer,
        "contexts": contexts
    }


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