from sqlalchemy.orm import Session
from typing import List, Dict, Any
from src.models import DocumentChunk
from src.ingestion.embedder import generate_embeddings
import logging

logger = logging.getLogger(__name__)

def vector_search(db: Session, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Performs a semantic vector search using pgvector.
    """
    logger.info(f"Performing vector search for query: '{query}'")
    # 1. Embed the query
    try:
        query_embedding = generate_embeddings([query])[0]
    except Exception as e:
        logger.error(f"Failed to embed query: {e}")
        return []
    
    # 2. Search database using L2 distance (<->)
    results = (
        db.query(DocumentChunk)
        .order_by(DocumentChunk.embedding.l2_distance(query_embedding))
        .limit(top_k)
        .all()
    )
    
    # Format results
    search_results = []
    for rank, chunk in enumerate(results):
        search_results.append({
            "content": chunk.content,
            "page_number": chunk.page_number,
            "source": f"{chunk.document.filename} (Page {chunk.page_number})",
            "score": 1.0 / (rank + 1), # simple inverse rank score for vector
            "type": "vector"
        })
        
    return search_results
