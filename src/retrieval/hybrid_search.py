from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
from src.retrieval.pgvector_store import vector_search
from src.retrieval.bm25_store import bm25_retriever
from src.retrieval.tracing import RAGPipelineTracer
import time
import logging

logger = logging.getLogger(__name__)

def rrf_score(rank: int, k: int = 60) -> float:
    """Calculates Reciprocal Rank Fusion score."""
    return 1.0 / (k + rank)

def hybrid_search(db: Session, query: str, top_k: int = 5, tracer: Optional[RAGPipelineTracer] = None) -> List[Dict[str, Any]]:
    """
    Performs hybrid search combining vector (semantic) and BM25 (keyword) search
    with query expansion and Reciprocal Rank Fusion (RRF).
    """
    logger.info(f"Starting hybrid search for: '{query}'")
    
    # 1. Ensure BM25 is initialized (lazy loading)
    if not bm25_retriever.bm25:
        bm25_retriever.initialize(db)
        
    fetch_k = top_k * 2
    
    # 2. Vector Search with timing
    t0 = time.time()
    vector_results = vector_search(db, query, top_k=fetch_k)
    vec_duration = (time.time() - t0) * 1000
    if tracer:
        tracer.log_vector_stage(len(vector_results), vec_duration)
        
    # 3. BM25 Search with timing
    t1 = time.time()
    keyword_results = bm25_retriever.search(query, top_k=fetch_k)
    kw_duration = (time.time() - t1) * 1000
    if tracer:
        tracer.log_bm25_stage(len(keyword_results), kw_duration)
    
    # 4. Fuse scores using RRF
    t2 = time.time()
    fused_scores = {}
    
    def process_results(results, result_type):
        for rank, res in enumerate(results):
            chunk_key = res["content"]
            if chunk_key not in fused_scores:
                fused_scores[chunk_key] = {
                    "content": res["content"],
                    "source": res["source"],
                    "page_number": res["page_number"],
                    "rrf_score": 0.0,
                    "matched_by": []
                }
            fused_scores[chunk_key]["rrf_score"] += rrf_score(rank + 1)
            if result_type not in fused_scores[chunk_key]["matched_by"]:
                fused_scores[chunk_key]["matched_by"].append(result_type)

    process_results(vector_results, "vector")
    process_results(keyword_results, "keyword")
    
    # Sort by fused score and take top_k
    ranked_results = sorted(fused_scores.values(), key=lambda x: x["rrf_score"], reverse=True)
    final_results = ranked_results[:top_k]
    
    overlap_count = sum(1 for item in fused_scores.values() if len(item["matched_by"]) > 1)
    rrf_duration = (time.time() - t2) * 1000
    if tracer:
        tracer.log_rrf_stage(len(fused_scores), overlap_count, rrf_duration)
        
    logger.info(f"Hybrid search returned {len(final_results)} fused results (overlap: {overlap_count}).")
    return final_results
