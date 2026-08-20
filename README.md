# NABL RAG Agent 

> **Note:** This project is currently under active development.

An intelligent, agentic RAG (Retrieval-Augmented Generation) system built to analyze and evaluate laboratory test reports and certificates against NABL (National Accreditation Board for Testing and Calibration Laboratories) standards.

## Features
- **Agentic Routing:** Automatically classifies user intent to switch between conversational chat and strict database searches.
- **Smart Document Analysis:** Upload test reports (PDFs) directly into the UI for the agent to evaluate.
- **Agentic Query Expansion:** Intelligently extracts key rules and ULRs from uploaded documents to formulate targeted database queries.
- **PostgreSQL Vector Database:** Uses `pgvector` and Hybrid Search (BM25 + Dense Embeddings) to retrieve precise NABL standards.
- **Flexible AI Architecture:** Supports fully local generation (e.g., Ollama `qwen3:8b`) for complete data privacy, with seamless scaling to cloud models via environment configuration.

## Tech Stack
- **Frontend / Chat UI:** Chainlit
- **Backend API:** FastAPI
- **Database:** PostgreSQL (with `pgvector` and SQLAlchemy)
- **AI / LLM:** Provider-agnostic (Local via Ollama, e.g., `qwen3:8b`, or Cloud APIs)
- **Embeddings & Reranking:** `nomic-embed-text` & `ms-marco-MiniLM-L-6-v2` CrossEncoder

## Setup (Local Development)
1. Clone the repository.
2. Ensure PostgreSQL is installed with the `pgvector` extension.
3. Copy `.env.example` to `.env` and fill in your database credentials and secret keys.
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Run database migrations:
   ```bash
   alembic upgrade head
   ```
6. Start the Chainlit UI:
   ```bash
   chainlit run src/chainlit_app.py
   ```
