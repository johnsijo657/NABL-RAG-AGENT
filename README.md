# NABL RAG Agent

> **Note:** This project is currently under active development.

An intelligent, agentic RAG (Retrieval-Augmented Generation) system built to analyze and evaluate laboratory test reports and calibration certificates against **NABL** (National Accreditation Board for Testing and Calibration Laboratories) and **ISO/IEC 17025** compliance standards.

---

## Features

- **Agentic Tool Calling & Routing:** Dynamically triggers NABL database lookups via tool calling (`search_nabl_database`) on cloud models or intent-based routing on local models.
- **Smart Multi-Modal Document Analysis:** Upload laboratory test reports (PDFs) directly into the UI. The pipeline extracts text page-by-page and renders high-fidelity page images for vision-capable models.
- **Hybrid Search with Reciprocal Rank Fusion (RRF):** Combines dense vector similarity (`pgvector` + `nomic-embed-text`) with sparse keyword matching (`BM25Okapi`) for maximum retrieval accuracy.
- **CrossEncoder Reranking:** Re-scores retrieved candidates using a local `ms-marco-MiniLM-L-6-v2` cross-encoder model with automatic GPU acceleration (CUDA) and CPU fallback.
- **Dual Interface:**
  - **Chainlit Web UI:** Interactive chat application with real-time streaming, document drag-and-drop, and session management.
  - **FastAPI REST API:** Headless endpoints (`/api/v1/chat`, `/api/v1/auth`, `/api/v1/health`) for third-party LIMS integrations.
- **Flexible LLM Architecture:** Supports cloud models via APIs or fully local, privacy-first deployment (via Ollama).

---

## Tech Stack

- **Frontend / Chat UI:** Chainlit
- **Backend API:** FastAPI
- **Database & Vector Store:** PostgreSQL with `pgvector` (SQLAlchemy + Alembic)
- **Embeddings & Reranking:** `nomic-embed-text` & `ms-marco-MiniLM-L-6-v2` CrossEncoder
- **LLM Orchestration:** LangChain (`langchain-groq`, `langchain-ollama`)

---

## Setup & Local Development

### 1. Prerequisites
- Python 3.10+
- PostgreSQL with the `pgvector` extension enabled
- (Optional) [Ollama](https://ollama.com/) if running local embeddings/LLMs (`ollama pull nomic-embed-text` and `ollama pull qwen3:8b`)

### 2. Installation
```bash
# Clone the repository
git clone <repo-url>
cd "NABL RAG AGENT"

# Create and activate virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Configuration
Copy `.env.example` to `.env` and configure your database credentials and API keys:
```bash
cp .env.example .env
```

### 4. Database Setup & Migrations
```bash
# Run database migrations
alembic upgrade head

# Create initial default users (admin, tester)
python scripts/create_users.py
```

### 5. Download Local Reranker Model
```bash
python scripts/download_model.py
```

### 6. Ingest NABL Documents
Place NABL standard PDFs inside `NABL_DOCUMENTS/` and run:
```bash
python scripts/ingest_documents.py
```

---

## Running the Application

### Option A: Interactive Web UI (Chainlit)
```bash
chainlit run src/chainlit_app.py
```

### Option B: REST API Backend (FastAPI)
```bash
uvicorn src.api.main:app --port 8001
```
