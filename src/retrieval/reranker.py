from typing import List, Dict, Any
from sentence_transformers import CrossEncoder
import logging

import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Resolve the absolute path to our local self-contained model
project_root = Path(__file__).parent.parent.parent.absolute()
local_model_path = project_root / "local_models" / "cross-encoder" / "ms-marco-MiniLM-L-6-v2"

# Initialize the cross-encoder model strictly from the local folder
try:
    if local_model_path.exists():
        logger.info(f"Loading local CrossEncoder model from {local_model_path}")
        reranker_model = CrossEncoder(str(local_model_path), max_length=512, local_files_only=True)
    else:
        logger.warning(f"Local model not found at {local_model_path}. Please run scripts/download_model.py")
        reranker_model = None
except Exception as e:
    logger.error(f"Failed to load CrossEncoder: {e}")
    reranker_model = None

def rerank_results(query: str, results: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Re-ranks search results using a CrossEncoder.
    """
    if not results:
        return []
        
    if reranker_model is None:
        logger.warning("Reranker model not loaded. Returning original results.")
        return results[:top_k]
        
    logger.info(f"Re-ranking {len(results)} results using CrossEncoder...")
    
    # Prepare pairs of (query, chunk_content)
    pairs = [[query, res["content"]] for res in results]
    
    try:
        # Predict scores
        scores = reranker_model.predict(pairs)
        
        # Attach scores to results
        for idx, score in enumerate(scores):
            results[idx]["rerank_score"] = float(score)
            
        # Sort by rerank_score
        ranked_results = sorted(results, key=lambda x: x["rerank_score"], reverse=True)
        return ranked_results[:top_k]
    except Exception as e:
        logger.error(f"Error during reranking: {e}")
        return results[:top_k]
