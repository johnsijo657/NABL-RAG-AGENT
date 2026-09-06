# NABL RAG Agent

An intelligent, agentic RAG (Retrieval-Augmented Generation) system built to analyze and evaluate laboratory test reports and calibration certificates against **NABL** (National Accreditation Board for Testing and Calibration Laboratories) and **ISO/IEC 17025:2017** compliance standards.

The system supports two independent interaction channels:
1. **Interactive Web UI (Chainlit)**
2. **REST API (FastAPI)** with API key authentication and rate limiting for external developers and automated LIMS (Laboratory Information Management Systems) integrations.

---

## Features

- **Hybrid Search with Reciprocal Rank Fusion (RRF):** Combines dense vector similarity (`pgvector` + `nomic-embed-text`) with sparse keyword matching (`BM25Okapi`) for pinpoint clause retrieval.
- **Local Cross-Encoder Reranking:** Re-scores retrieved candidates using a local `ms-marco-MiniLM-L-6-v2` model with automatic GPU (CUDA) acceleration and CPU fallback.
- **On-Demand Visual Document Inspection:** The vision tool (`inspect_document_visuals`) executes only when needed to inspect physical artifacts on certificates (NABL accredited logo, authorized signatory signature, official lab seal/stamp, and QR codes), eliminating rate limits and token waste.
- **Dynamic Multi-Provider Registry:** Seamlessly routes between cloud and local providers, switched dynamically via `.env`.
- **X-API-Key Authentication:** Secure developer authentication using cryptographic SHA-256 key hashing in PostgreSQL.
- **Request-Arrival Sliding-Window Rate Limiter:** Per-key rate limiting that protects the pipeline and returns standard `HTTP 429 Too Many Requests` with a `Retry-After` header (currently in-memory, with pluggable support for Redis in distributed deployments).
- **Dual Endpoint Architecture:**
  - `POST /api/v1/chat/`: Unified endpoint supporting pure text questions, conversation history, and optional PDF file upload for custom chat interfaces.
  - `POST /api/v1/audit/`: Headless quality gate endpoint for automated LIMS pipelines to audit PDF test reports prior to release.
- **CLI API Key Management:** Command-line tool to create, inspect, and revoke API keys with custom rate limits.

---

## Tech Stack

- **Frontend UI:** [Chainlit](https://chainlit.io/) (Port 8002)
- **Backend REST API:** [FastAPI](https://fastapi.tiangolo.com/) + Uvicorn (Port 8001)
- **Database & Vector Store:** PostgreSQL with [`pgvector`](https://github.com/pgvector/pgvector)
- **ORM & Migrations:** SQLAlchemy 2.0 & Alembic
- **Document & Image Processing:** PyMuPDF (`fitz`) + Pillow
- **Information Retrieval:** `rank-bm25` (BM25Okapi) + Sentence-Transformers (`CrossEncoder`)
- **LLM Orchestration:** LangChain

---

## Project Structure

```
├── NABL_DOCUMENTS/          # Official NABL & ISO/IEC 17025 standard PDFs for reference
├── alembic/                 # Database schema migrations
├── scripts/                 # CLI tools (ingestion, API key management, model downloads)
├── src/
│   ├── api/                 # FastAPI REST API (routes, auth, rate limiting)
│   ├── generation/          # LLM orchestration, provider registry, prompts & vision
│   ├── ingestion/           # Document parsing, chunking, and embedding pipeline
│   ├── retrieval/           # Hybrid search (dense pgvector + sparse BM25) and CrossEncoder reranking
│   ├── chainlit_app.py      # Interactive Chainlit web UI
│   ├── config.py            # Global configuration settings
│   └── models.py            # SQLAlchemy database models
└── tests/                   # Automated test suites
```

---

## Setup & Local Development

### 1. Prerequisites
- Python 3.10+
- PostgreSQL with the `pgvector` extension enabled
- (Optional) Local LLM runner (e.g., Ollama) if running local models

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
Copy `.env.example` to `.env` and configure your settings:
```bash
cp .env.example .env
```
Key configuration parameters include:
- `DATABASE_URL`: PostgreSQL connection string with `pgvector` support.
- `LLM_PROVIDER`: Provider selector (`cloud` or `local`).
- Provider API keys and target model identifiers.
- Embedding and reranker model paths.

### 4. Database Setup & Migrations
```bash
# Run database migrations
alembic upgrade head

# (Optional) Create initial web UI accounts
python scripts/create_users.py
```

### 5. Download Local Reranker Model
```bash
python scripts/download_model.py
```

### 6. Ingest Reference NABL Standards
Place official NABL standard PDFs in `NABL_DOCUMENTS/` and run:
```bash
python scripts/ingest_documents.py
```

---

## Running the Application

### 1. Interactive Web UI (Chainlit)
Runs on port **8002**:
```bash
chainlit run src/chainlit_app.py --port 8002
```

### 2. REST API Backend (FastAPI)
Runs on port **8001**:
```bash
uvicorn src.api.main:app --port 8001 --reload
```
Interactive Swagger documentation is available at: `http://localhost:8001/docs`

---

## API Key Management

Developer access to the REST API requires a valid API key passed via the `X-API-Key` header. Keys are managed using the CLI:

```bash
# Create a new API key with a rate limit (requests per minute)
python scripts/manage_api_keys.py create --name "LIMS-Integration" --rate-limit 60

# List all active API keys
python scripts/manage_api_keys.py list

# Revoke an API key
python scripts/manage_api_keys.py revoke --key-id <KEY_UUID>
```

---

## Developer API Usage

### 1. Unified Chat (`POST /api/v1/chat/`)
Supports general compliance questions or contextual document inquiries.

**JSON Text Request:**
```bash
curl -X POST "http://localhost:8001/api/v1/chat/" \
  -H "X-API-Key: nabl_live_your_key_here" \
  -H "Content-Type: application/json" \
  -d '{"message": "What are the requirements for reporting calibration results under ISO 17025?"}'
```

**Multipart Request with PDF:**
```bash
curl -X POST "http://localhost:8001/api/v1/chat/" \
  -H "X-API-Key: nabl_live_your_key_here" \
  -F "message=Does this certificate meet NABL reporting requirements?" \
  -F "file=@sample_report.pdf;type=application/pdf"
```

### 2. Document Compliance Audit (`POST /api/v1/audit/`)
Headless audit endpoint for pre-release quality gates.

```bash
curl -X POST "http://localhost:8001/api/v1/audit/" \
  -H "X-API-Key: nabl_live_your_key_here" \
  -F "file=@sample_report.pdf;type=application/pdf"
```
