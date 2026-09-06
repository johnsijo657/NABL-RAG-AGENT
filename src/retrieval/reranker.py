from typing import List, Dict, Any, Optional
from sentence_transformers import CrossEncoder
import logging
from pathlib import Path
from src.config import settings

logger = logging.getLogger(__name__)

# Resolve the absolute path to our local self-contained model
project_root = Path(__file__).parent.parent.parent.absolute()
local_model_path = project_root / "local_models" / "cross-encoder" / "ms-marco-MiniLM-L-6-v2"

import torch

# Check if an NVIDIA GPU is available, otherwise fallback to CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize the cross-encoder model strictly from the local folder
try:
    if local_model_path.exists():
        logger.info(f"Loading local CrossEncoder model from {local_model_path} on device: {device}")
        reranker_model = CrossEncoder(str(local_model_path), max_length=512, local_files_only=True, device=device)
    else:
        logger.warning(f"Local model not found at {local_model_path}. Please run scripts/download_model.py")
        reranker_model = None
except Exception as e:
    logger.error(f"Failed to load CrossEncoder: {e}")
    reranker_model = None

import time

def rerank_results(
    query: str, 
    results: List[Dict[str, Any]], 
    top_k: int = 5, 
    score_threshold: Optional[float] = None,
    tracer: Optional[Any] = None
) -> List[Dict[str, Any]]:
    """
    Re-ranks search results using a CrossEncoder and filters out candidates below score_threshold.
    """
    if not results:
        if tracer:
            tracer.log_rerank_stage(0, 0, [], 0.0)
            tracer.finish_and_log()
        return []
        
    threshold = score_threshold if score_threshold is not None else settings.RERANK_SCORE_THRESHOLD
        
    if reranker_model is None:
        logger.warning("Reranker model not loaded. Returning original results.")
        if tracer:
            tracer.log_rerank_stage(len(results), min(len(results), top_k), [], 0.0)
            tracer.finish_and_log()
        return results[:top_k]
        
    logger.info(f"Re-ranking {len(results)} results using CrossEncoder...")
    
    # Prepare pairs of (query, chunk_content)
    pairs = [[query, res["content"]] for res in results]
    
    try:
        t0 = time.time()
        # Predict scores
        scores = reranker_model.predict(pairs)
        duration_ms = (time.time() - t0) * 1000
        
        # Attach scores to results
        for idx, score in enumerate(scores):
            results[idx]["rerank_score"] = float(score)
            
        # Sort by rerank_score
        ranked_results = sorted(results, key=lambda x: x["rerank_score"], reverse=True)
        
        # Filter by score threshold
        filtered = ranked_results
        if threshold is not None:
            filtered = [r for r in ranked_results if r.get("rerank_score", -999) >= threshold]
            if not filtered:
                logger.info(f"All candidates scored below relevance threshold ({threshold}). Returning empty to prevent hallucination.")
                if tracer:
                    tracer.log_rerank_stage(len(results), 0, [], duration_ms)
                    tracer.finish_and_log()
                return []
                
        final_results = filtered[:top_k]
        if tracer:
            tracer.log_rerank_stage(
                in_count=len(results),
                out_count=len(final_results),
                top_scores=[r.get("rerank_score", 0.0) for r in final_results],
                duration_ms=duration_ms
            )
            tracer.finish_and_log()
            
        return final_results
    except Exception as e:
        logger.error(f"Error during reranking: {e}")
        if tracer:
            tracer.finish_and_log()
        return results[:top_k]
