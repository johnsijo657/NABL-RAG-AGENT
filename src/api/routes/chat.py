import time
import base64
import json
import pymupdf
from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, File, Form
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from src.database import get_db
from src.api.auth import get_api_key
from src.models import User, QueryLog, ApiKey
from src.config import settings
from src.retrieval.hybrid_search import hybrid_search
from src.retrieval.reranker import rerank_results
from src.retrieval.tracing import RAGPipelineTracer
from src.generation.provider_factory import get_llm_provider

router = APIRouter(prefix="/api/v1/chat", tags=["chat"])

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: Optional[str] = None
    query: Optional[str] = None
    document_text: Optional[str] = None
    history: Optional[List[ChatMessage]] = []

class SourceRef(BaseModel):
    content: str
    source: str
    page_number: int

class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceRef]

@router.post(
    "/",
    response_model=ChatResponse,
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "message": {"type": "string", "description": "Question or compliance instruction"},
                            "history": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "role": {"type": "string"},
                                        "content": {"type": "string"}
                                    }
                                }
                            },
                            "document_text": {"type": "string", "description": "Optional raw text of a document to evaluate"}
                        },
                        "required": ["message"]
                    }
                },
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "message": {"type": "string", "description": "Question or compliance instruction"},
                            "file": {"type": "string", "format": "binary", "description": "Optional PDF test report or calibration certificate"},
                            "history": {"type": "string", "description": "Optional JSON array string of previous conversation turns"}
                        },
                        "required": ["message"]
                    }
                }
            }
        }
    }
)
async def chat_endpoint(
    request: Request,
    db: Session = Depends(get_db), 
    api_key: ApiKey = Depends(get_api_key)
):
    """
    Unified Conversational & Document Compliance Chat Endpoint:
    - Pure text question (JSON or Form) -> Answers using Vector DB & NABL standards.
    - Question + Optional PDF upload (Multipart) -> Evaluates PDF in-memory against NABL standards with visual checks.
    - Supports ongoing conversation history for multi-turn dialogues.
    """
    start_time = time.time()
    content_type = request.headers.get("content-type", "").lower()
    
    prompt_text = ""
    history_dicts: List[Dict[str, str]] = []
    extracted_text = ""
    uploaded_images: List[str] = []
    
    # 1. Parse Input: Handle both Multipart Form-Data and Application/JSON
    if "multipart/form-data" in content_type:
        form = await request.form()
        prompt_text = (form.get("message") or form.get("query") or "").strip()
        
        # Parse history if provided as JSON string in form data
        history_raw = form.get("history")
        if history_raw:
            try:
                history_dicts = json.loads(history_raw) if isinstance(history_raw, str) else history_raw
            except Exception:
                history_dicts = []
                
        # Handle optional PDF file attachment in-memory
        uploaded_file = form.get("file")
        if uploaded_file and hasattr(uploaded_file, "read"):
            filename = getattr(uploaded_file, "filename", "") or "uploaded_document.pdf"
            if not filename.lower().endswith(".pdf"):
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file format: '{filename}'. Only PDF documents (.pdf) are supported at present."
                )
            file_bytes = await uploaded_file.read()
            if file_bytes:
                try:
                    doc = pymupdf.open(stream=file_bytes, filetype="pdf")
                    for i, page in enumerate(doc):
                        page_text = page.get_text("text").strip()
                        if page_text:
                            extracted_text += f"\n--- Page {i+1} ---\n{page_text}\n"
                        if i < 3:
                            matrix = pymupdf.Matrix(1.0, 1.0)
                            pix = page.get_pixmap(matrix=matrix)
                            img_b64 = base64.b64encode(pix.tobytes("png")).decode("utf-8")
                            uploaded_images.append(img_b64)
                    doc.close()
                except Exception as e:
                    raise HTTPException(status_code=422, detail=f"Failed to parse PDF document: {str(e)}")
    else:
        # Default to JSON payload
        try:
            body = await request.json()
        except Exception:
            body = {}
        prompt_text = (body.get("message") or body.get("query") or "").strip()
        history_dicts = body.get("history", []) or []
        doc_text = body.get("document_text")
        if doc_text and doc_text.strip():
            extracted_text = doc_text.strip()
            
    # Validate prompt or assign default if file was uploaded with empty message
    if not prompt_text:
        if extracted_text:
            prompt_text = "Evaluate this test report/certificate for compliance against NABL and ISO/IEC 17025 requirements. Detail any missing mandatory information or non-conformances."
        else:
            raise HTTPException(status_code=422, detail="Missing required field: 'message' (or 'query')")
            
    # 2. Build effective prompt combining question + in-memory document text
    effective_prompt = prompt_text
    if extracted_text:
        effective_prompt = (
            f"{prompt_text}\n\n"
            f"--- Attached Document Under Evaluation ---\n"
            f"{extracted_text.strip()}"
        )
        
    # 3. Service builder determines the LLM powering the engine under the hood
    active_provider = get_llm_provider()
    
    # 4. Search callback for the LLM tool execution
    last_sources: List[Dict[str, Any]] = []
    last_trace_data: Dict[str, Any] = {}
    
    async def run_search(search_query: str) -> List[Dict[str, Any]]:
        tracer = RAGPipelineTracer(search_query)
        hybrid_results = hybrid_search(db, search_query, top_k=10, tracer=tracer)
        reranked_results = rerank_results(search_query, hybrid_results, top_k=settings.RERANK_TOP_N, tracer=tracer)
        last_sources.extend(reranked_results)
        last_trace_data.update(tracer.stages)
        return reranked_results
    
    # 5. Generate Answer / Audit Verdict using the LLM agent
    try:
        chunks = []
        async for chunk in active_provider.generate_response_stream(
            query=effective_prompt,
            chat_history=history_dicts,
            search_callback=run_search,
            uploaded_images=uploaded_images if uploaded_images else None
        ):
            chunks.append(chunk)
            
        answer = "".join(chunks).strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM Generation failed: {str(e)}")
        
    # 6. Calculate latency
    latency_ms = (time.time() - start_time) * 1000
    sources_used_str = "; ".join([res["source"] for res in last_sources]) if last_sources else "None"
    
    # Log query with trace telemetry (internal audit records model used)
    q_log = QueryLog(
        user_id=api_key.owner_id,
        query=prompt_text,
        response=answer,
        latency_ms=latency_ms,
        sources_used=sources_used_str,
        model_used=active_provider.model_name,
        trace_data=last_trace_data if last_trace_data else None
    )
    db.add(q_log)
    db.commit()
    
    # 7. Format sources for response
    sources = [
        SourceRef(
            content=res["content"], 
            source=res["source"], 
            page_number=res["page_number"]
        ) for res in last_sources
    ]
    
    return ChatResponse(answer=answer, sources=sources)
