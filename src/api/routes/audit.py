import time
import base64
import logging
import pymupdf
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
from pydantic import BaseModel

from src.database import get_db
from src.api.auth import get_api_key
from src.models import ApiKey, QueryLog
from src.config import settings
from src.retrieval.hybrid_search import hybrid_search
from src.retrieval.reranker import rerank_results
from src.retrieval.tracing import RAGPipelineTracer
from src.generation.provider_factory import get_llm_provider

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/audit", tags=["audit"])

class SourceRef(BaseModel):
    content: str
    source: str
    page_number: int

class AuditResponse(BaseModel):
    filename: str
    verdict: str
    sources: List[SourceRef]
    latency_ms: float

DEFAULT_AUDIT_PROMPT = (
    "Evaluate this test report/certificate for compliance against NABL and ISO/IEC 17025 requirements. "
    "Check for mandatory reporting elements (clause 7.8), physical artifacts (NABL symbol/logo, authorized signature, laboratory seal), "
    "and identify any non-conformances or missing details with clause citations."
)

@router.post("/", response_model=AuditResponse)
async def audit_document_endpoint(
    file: UploadFile = File(..., description="PDF test report or certificate to audit against NABL standards"),
    message: Optional[str] = Form(None, description="Optional custom audit question or instruction"),
    db: Session = Depends(get_db),
    api_key: ApiKey = Depends(get_api_key)
):
    """
    Ephemeral Document Compliance Audit:
    Accepts a developer-uploaded document (PDF or image) in-memory.
    Audits it against official NABL/ISO 17025 standards in the vector database without persisting the document to disk or DB.
    """
    start_time = time.time()
    filename = file.filename or "uploaded_document"
    audit_instruction = (message or "").strip() or DEFAULT_AUDIT_PROMPT

    # 1. Read document into memory (Ephemeral - never stored in Vector DB)
    extracted_text = ""
    uploaded_images: List[str] = []

    try:
        content_bytes = await file.read()
        if not content_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        # Strictly enforce PDF-only uploads for compliance audit
        if not filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format: '{filename}'. Only PDF documents (.pdf) are supported for compliance audit at present."
            )

        # Handle PDF documents in-memory
        doc = pymupdf.open(stream=content_bytes, filetype="pdf")
        total_pages = len(doc)
        
        for i, page in enumerate(doc):
            page_text = page.get_text("text").strip()
            if page_text:
                extracted_text += f"\n--- Page {i+1} ---\n{page_text}\n"
            
            # Render first 3 pages as base64 images for visual inspection tool
            if i < 3:
                matrix = pymupdf.Matrix(1.0, 1.0)
                pix = page.get_pixmap(matrix=matrix)
                img_b64 = base64.b64encode(pix.tobytes("png")).decode("utf-8")
                uploaded_images.append(img_b64)
                
        doc.close()


    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to process document {filename} in-memory: {e}")
        raise HTTPException(status_code=422, detail=f"Failed to parse document: {str(e)}")

    # 2. Build effective prompt with in-memory document text
    effective_prompt = (
        f"{audit_instruction}\n\n"
        f"--- Attached Document Under Audit ({filename}) ---\n"
        f"{extracted_text.strip()}"
    )

    # 3. Service builder resolves LLM provider under the hood
    active_provider = get_llm_provider()

    # Track sources and telemetry across tool calls
    last_sources: List[Dict[str, Any]] = []
    last_trace_data: Dict[str, Any] = {}

    # Define search callback: queries official NABL knowledge base in PostgreSQL
    async def run_search(search_query: str) -> List[Dict[str, Any]]:
        tracer = RAGPipelineTracer(search_query)
        hybrid_results = hybrid_search(db, search_query, top_k=10, tracer=tracer)
        reranked_results = rerank_results(search_query, hybrid_results, top_k=settings.RERANK_TOP_N, tracer=tracer)
        last_sources.extend(reranked_results)
        last_trace_data.update(tracer.stages)
        return reranked_results

    # 4. Generate Compliance Audit Report using the LLM agent
    try:
        chunks = []
        async for chunk in active_provider.generate_response_stream(
            query=effective_prompt,
            chat_history=[],
            search_callback=run_search,
            uploaded_images=uploaded_images if uploaded_images else None
        ):
            chunks.append(chunk)

        verdict = "".join(chunks).strip()
    except Exception as e:
        logger.error(f"LLM audit generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Audit generation failed: {str(e)}")

    # 5. Calculate latency and log query (without storing document data)
    latency_ms = (time.time() - start_time) * 1000
    sources_used_str = "; ".join([res["source"] for res in last_sources]) if last_sources else "None"

    q_log = QueryLog(
        user_id=api_key.owner_id,
        query=f"[Audit Document: {filename}] {audit_instruction}",
        response=verdict,
        latency_ms=latency_ms,
        sources_used=sources_used_str,
        model_used=active_provider.model_name,
        trace_data=last_trace_data if last_trace_data else None
    )
    db.add(q_log)
    db.commit()

    # 6. Format sources for developer response
    sources = [
        SourceRef(
            content=res["content"],
            source=res["source"],
            page_number=res["page_number"]
        ) for res in last_sources
    ]

    return AuditResponse(
        filename=filename,
        verdict=verdict,
        sources=sources,
        latency_ms=latency_ms
    )
