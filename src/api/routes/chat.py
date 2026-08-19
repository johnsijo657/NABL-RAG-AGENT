import time
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from src.database import get_db
from src.api.auth import get_current_user
from src.models import User, QueryLog
from src.retrieval.hybrid_search import hybrid_search
from src.retrieval.reranker import rerank_results
from src.generation.ollama_provider import OllamaProvider

router = APIRouter(prefix="/api/v1/chat", tags=["chat"])

# Initialize the LLM strategy
llm_provider = OllamaProvider()

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    query: str
    history: Optional[List[ChatMessage]] = []

class SourceRef(BaseModel):
    content: str
    source: str
    page_number: int

class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceRef]

@router.post("/", response_model=ChatResponse)
def chat_endpoint(
    request: ChatRequest, 
    db: Session = Depends(get_db), 
    current_user: User = Depends(get_current_user)
):
    start_time = time.time()
    
    # 1. Retrieve Context
    hybrid_results = hybrid_search(db, request.query, top_k=10)
    
    # 2. Re-rank
    reranked_results = rerank_results(request.query, hybrid_results, top_k=5)
    
    # Convert Pydantic history to dict
    history_dicts = [{"role": msg.role, "content": msg.content} for msg in request.history]
    
    # 3. Generate Answer
    try:
        answer = llm_provider.generate_response(request.query, reranked_results, history_dicts)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM Generation failed: {str(e)}")
        
    # Calculate latency
    latency_ms = (time.time() - start_time) * 1000
    
    # 4. Log the query
    q_log = QueryLog(
        user_id=current_user.id,
        query=request.query,
        response=answer,
        latency_ms=latency_ms
    )
    db.add(q_log)
    db.commit()
    
    # Format sources for response
    sources = [
        SourceRef(
            content=res["content"], 
            source=res["source"], 
            page_number=res["page_number"]
        ) for res in reranked_results
    ]
    
    return ChatResponse(answer=answer, sources=sources)
