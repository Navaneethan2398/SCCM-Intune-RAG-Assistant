# 🤖 SCCM & Intune AI Assistant using RAG

An intelligent AI-powered assistant that answers Microsoft SCCM and Intune administration questions using Retrieval-Augmented Generation (RAG). The application combines Hybrid Search, ChromaDB, Cross-Encoder Reranking, and Groq LLM to deliver accurate, context-aware responses while reducing hallucinations.

---

## 📌 Overview

Managing Microsoft SCCM and Intune documentation can be time-consuming. Administrators often need to search through multiple Microsoft Learn articles to find configuration steps, troubleshooting guidance, or best practices.

This project solves that problem by building an AI assistant that retrieves relevant documentation and generates reliable answers grounded in Microsoft Learn content.

---

## 🚀 Features

- Hybrid Search (BM25 + Vector Search)
- ChromaDB Vector Database
- MMR (Maximum Marginal Relevance) Retrieval
- Cross-Encoder Reranking
- History-Aware Conversational Retrieval
- Prompt Injection Protection
- Source Citations
- Streamlit Web Interface
- RAG Evaluation using RAGAS
- Persistent Vector Database
- Microsoft Learn Documentation Knowledge Base

---

## 🏗️ System Architecture

```
                    User Query
                         │
                         ▼
                  Streamlit Interface
                         │
                         ▼
             History Aware Retriever
                         │
         ┌───────────────┴───────────────┐
         ▼                               ▼
    BM25 Retriever                 Chroma Retriever
         │                               │
         └───────────────┬───────────────┘
                         ▼
                 Ensemble Retriever
                         ▼
                  MMR Retrieval
                         ▼
              Cross Encoder Reranker
                         ▼
            Top Relevant Documents
                         ▼
              Groq Llama 3.3 70B
                         ▼
               AI Generated Answer
                         ▼
                Source Citations
```

---

## 🛠️ Tech Stack

### Languages

- Python

### Frameworks

- LangChain
- Streamlit

### Embedding Model

- BAAI/bge-base-en-v1.5

### Large Language Model

- Llama-3.3-70B-Versatile (Groq)

### Vector Database

- ChromaDB

### Retrieval Techniques

- Hybrid Search
- BM25
- Dense Retrieval
- Maximum Marginal Relevance (MMR)

### Reranking

- Cross-Encoder (MS MARCO MiniLM)

### Evaluation

- RAGAS

---

## 📂 Project Structure

```
SCCM-Intune-RAG-Assistant/

│── app.py
│── streamlit.py
│── evaluate.py
│── guardrails.py
│── requirements.txt
│── evaluation_dataset.csv
│── LICENSE
│── README.md
│── .gitignore
```

---

## ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/Navaneethan2398/SCCM-Intune-RAG-Assistant.git

cd SCCM-Intune-RAG-Assistant
```

Create virtual environment

```bash
python -m venv venv
```

Activate

Windows

```bash
venv\Scripts\activate
```

Linux / Mac

```bash
source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

Create a `.env` file

```env
GROQ_API_KEY=YOUR_API_KEY
```

Run the application

```bash
streamlit run streamlit.py
```

---

## 📊 Evaluation

The system was evaluated using **RAGAS**.

Metrics used:

- Faithfulness
- Answer Relevancy
- Context Precision
- Context Recall

Example Results

| Metric | Score |
|----------|---------|
| Faithfulness | 0.93 |
| Answer Relevancy | 0.81 |
| Context Precision | 1.00 |
| Context Recall | 1.00 |

---

## 🛡️ Guardrails

The application includes:

- Prompt Injection Detection
- Source Citation Enforcement
- Context-only Response Generation
- Hallucination Reduction using RAG

---

## 💡 Future Improvements

- LangGraph Agent Workflow
- Multi-Agent Support
- Feedback Learning
- User Authentication
- Docker Deployment
- Kubernetes Deployment
- CI/CD Pipeline
- Redis Caching
- PostgreSQL Metadata Storage


## 📚 Knowledge Base

The assistant retrieves information from Microsoft Learn documentation related to:

- Microsoft Intune
- Microsoft Configuration Manager (SCCM)
- Windows Update Management
- Device Configuration
- Endpoint Protection
- Software Deployment

## 👨‍💻 Author

**Navaneethan R**

AI Engineer | Generative AI | LLM | RAG | LangChain | Python

GitHub:
https://github.com/Navaneethan2398

LinkedIn:
(Add your LinkedIn URL)

---

## ⭐ If you found this project useful, please give it a star!
