from sqlalchemy.orm import Session
from typing import List, Dict, Any
from src.retrieval.pgvector_store import vector_search
from src.retrieval.bm25_store import bm25_retriever
import logging

logger = logging.getLogger(__name__)

def rrf_score(rank: int, k: int = 60) -> float:
    """Calculates Reciprocal Rank Fusion score."""
    return 1.0 / (k + rank)

def hybrid_search(db: Session, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Performs hybrid search combining vector (semantic) and BM25 (keyword) search
    using Reciprocal Rank Fusion (RRF).
    """
    logger.info(f"Starting hybrid search for: '{query}'")
    
    # 1. Ensure BM25 is initialized (lazy loading)
    if not bm25_retriever.bm25:
        bm25_retriever.initialize(db)
        
    # 2. Get results from both retrievers
    # Fetch slightly more to ensure good overlap for RRF
    fetch_k = top_k * 2
    
    vector_results = vector_search(db, query, top_k=fetch_k)
    keyword_results = bm25_retriever.search(query, top_k=fetch_k)
    
    # 3. Fuse scores using RRF
    fused_scores = {}
    
    # We use 'content' as the unique identifier for the chunk to merge
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
            fused_scores[chunk_key]["matched_by"].append(result_type)

    process_results(vector_results, "vector")
    process_results(keyword_results, "keyword")
    
    # 4. Sort by fused score and take top_k
    ranked_results = sorted(fused_scores.values(), key=lambda x: x["rrf_score"], reverse=True)
    
    final_results = ranked_results[:top_k]
    logger.info(f"Hybrid search returned {len(final_results)} fused results.")
    return final_results
