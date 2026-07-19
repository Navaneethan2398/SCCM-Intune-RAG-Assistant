import time
import pandas as pd
from datasets import Dataset

from ragas.run_config import RunConfig
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from langchain_groq import ChatGroq

from app import (
    ask_with_context,
    llm,
    embeddings,
    GROQ_API_KEY,
)

# ==========================================================
# Wrap Groq LLM & Embeddings for RAGAS
# ==========================================================
# Judge model choice:
#   - llama-3.3-70b-versatile: more reliable/consistent scoring, but
#     shares the app's 100K TPD budget. Use this for real, trustworthy
#     before/after comparisons.
#   - llama-3.1-8b-instant: much larger daily quota, but has shown
#     significant run-to-run scoring noise on the same answer text.
#     Only use for quick, low-stakes sanity checks during dev iteration,
#     not for deciding whether a prompt change actually helped.
#
# Set to False once your 70B daily quota allows — this gives you
# trustworthy scores for comparing prompt versions.
USE_CHEAP_JUDGE = False

if USE_CHEAP_JUDGE:
    judge_llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=GROQ_API_KEY,
        temperature=0,
    )
else:
    judge_llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=GROQ_API_KEY,
        temperature=0,
    )

ragas_llm = LangchainLLMWrapper(judge_llm)
ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)

# ==========================================================
# Load Evaluation Dataset
# ==========================================================

print("Loading evaluation dataset...")

df = pd.read_csv("evaluation_dataset.csv")
BATCH_START =0
BATCH_END = 5
df = df.iloc[BATCH_START:BATCH_END].reset_index(drop=True)

questions = []
answers = []
contexts = []
ground_truths = []

print(f"Evaluating {len(df)} questions...\n")

start_time = time.time()

# ==========================================================
# Run RAG Pipeline
# ==========================================================

for idx, row in df.iterrows():

    question = row["question"]

    print(f"[{idx+1}/{len(df)}] {question}")

    result = ask_with_context(
        question,
        []
    )

    questions.append(question)
    answers.append(result["answer"])
    contexts.append(result["contexts"])
    ground_truths.append(row["ground_truth"])

elapsed = time.time() - start_time

# ==========================================================
# Create HuggingFace Dataset
# ==========================================================

dataset = Dataset.from_dict(
    {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    }
)

print("\nRunning RAGAS Evaluation...\n")

# ==========================================================
# Evaluate
# ==========================================================

results = evaluate(
    dataset=dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ],
    llm=ragas_llm,
    embeddings=ragas_embeddings,
    run_config=RunConfig(
        max_workers=1,
        timeout=120,
    ),
)

# ==========================================================
# Save Results
# ==========================================================

results_df = results.to_pandas()

results_df.to_csv(
    f"evaluation_results_{BATCH_START}_{BATCH_END}.csv",
    index=False
)

# ==========================================================
# Display Summary
# ==========================================================

print("\n" + "=" * 60)
print("RAG EVALUATION SUMMARY")
print("=" * 60)

print(results_df.mean(numeric_only=True))

print(f"\nQuestions Evaluated : {len(df)}")
print(f"Total Time          : {elapsed:.2f} sec")
print(f"Average Time/Query  : {elapsed/len(df):.2f} sec")

print(f"\nResults saved to evaluation_results_{BATCH_START}_{BATCH_END}.csv")