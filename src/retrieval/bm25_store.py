from sqlalchemy.orm import Session
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
from src.models import Document, DocumentChunk
import logging
import re

logger = logging.getLogger(__name__)

class BM25Retriever:
    def __init__(self):
        self.bm25 = None
        self.corpus_chunks = []

    def invalidate(self):
        """Invalidates the cached BM25 index to trigger a rebuild on next search."""
        logger.info("Invalidating BM25 index cache.")
        self.bm25 = None
        self.corpus_chunks = []
        
    def _tokenize(self, text: str) -> List[str]:
        # Simple tokenization: lowercase and split by non-alphanumeric
        return re.findall(r'\w+', text.lower())
        
    def initialize(self, db: Session):
        """Loads all chunks from the DB and builds the BM25 index using a joined query."""
        logger.info("Initializing BM25 keyword index...")
        results = (
            db.query(DocumentChunk.content, DocumentChunk.page_number, Document.filename)
            .join(Document, DocumentChunk.document_id == Document.id)
            .all()
        )
        if not results:
            logger.warning("No documents found in DB to index for BM25.")
            self.bm25 = None
            self.corpus_chunks = []
            return
            
        self.corpus_chunks = [
            {
                "content": row.content,
                "page_number": row.page_number,
                "filename": row.filename or "Unknown"
            }
            for row in results
        ]
            
        tokenized_corpus = [self._tokenize(chunk["content"]) for chunk in self.corpus_chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
        logger.info(f"BM25 index built with {len(self.corpus_chunks)} chunks.")
        
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not self.bm25:
            logger.error("BM25 index not initialized.")
            return []
            
        tokenized_query = self._tokenize(query)
        # Get scores
        doc_scores = self.bm25.get_scores(tokenized_query)
        
        # Get top k indices
        top_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:top_k]
        
        search_results = []
        for idx in top_indices:
            score = doc_scores[idx]
            if score <= 0:
                continue # Skip zero matches
            chunk = self.corpus_chunks[idx]
            search_results.append({
                "content": chunk["content"],
                "page_number": chunk["page_number"],
                "source": f"{chunk['filename']} (Page {chunk['page_number']})",
                "score": score,
                "type": "keyword"
            })
            
        return search_results

# Global instance to cache the index across requests
bm25_retriever = BM25Retriever()
