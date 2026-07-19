import os
import re

# ==========================================================
# Prompt Injection Protection
# ==========================================================

PROMPT_INJECTION_PATTERNS = [

    r"ignore\s+(all\s+)?previous\s+instructions",
    r"forget\s+(all\s+)?previous\s+instructions",
    r"system\s+prompt",
    r"reveal\s+.*prompt",
    r"show\s+.*prompt",
    r"developer\s+message",
    r"you\s+are\s+now",
    r"act\s+as",
    r"pretend\s+to\s+be",
    r"bypass",
    r"disable\s+guardrails",
    r"jailbreak",
    r"override",
    r"ignore\s+your\s+rules",
]

BLOCK_MESSAGE = (
    "This request attempts to override the assistant's instructions. "
    "I can only answer Microsoft SCCM and Intune related questions."
)


def detect_prompt_injection(query: str):

    query = query.lower()

    for pattern in PROMPT_INJECTION_PATTERNS:
        if re.search(pattern, query):
            return True

    return False


# ==========================================================
# Source Citation Builder
# ==========================================================

# BAAI/bge-reranker-base here returns a sigmoid probability in [0, 1]
# for each (query, document) pair, not a raw logit. Real observed
# scores for a genuinely relevant chunk land well above 0.3, while
# irrelevant chunks (including PDF chunks retrieved but not actually
# about the topic asked) cluster near 0.0 (typically under 0.01).
# 0.05 sits cleanly in that gap. If you notice genuinely relevant
# sources getting dropped from citations, or clearly irrelevant ones
# still slipping through, turn on CITATION_DEBUG below to see real
# scores for your queries and adjust this threshold accordingly.
CITATION_SCORE_THRESHOLD = 0.05

# Set the CITATION_DEBUG environment variable to "1" to print each
# retrieved document's rerank_score and source to the console. Use
# this to see the actual score distribution bge-reranker-base
# produces for your queries, then adjust CITATION_SCORE_THRESHOLD
# above based on real numbers rather than a guess.
CITATION_DEBUG = os.getenv("CITATION_DEBUG", "0") == "1"


def build_citations(documents, score_threshold: float = CITATION_SCORE_THRESHOLD):

    citations = []
    seen = set()

    for doc in documents:

        # rerank_score is only present when documents went through
        # ScoringCrossEncoderReranker (see app.py). If it's missing for
        # any reason, fall back to the old behavior of citing it
        # rather than silently dropping a source we can't score.
        score = doc.metadata.get("rerank_score")
        source = doc.metadata.get("source", "")

        if CITATION_DEBUG:
            print(f"[citation debug] rerank_score={score!r}  source={source}")

        if score is not None and score < score_threshold:
            continue

        if source and source not in seen:
            seen.add(source)
            citations.append(source)

    return citations


def append_citations(answer: str, documents):

    citations = build_citations(documents)

    if not citations:
        return answer

    answer += "\n\n---\n"
    answer += "**Sources**\n"

    for i, src in enumerate(citations, 1):
        answer += f"{i}. {src}\n"

    return answer