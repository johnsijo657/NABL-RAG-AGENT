import time
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class RAGPipelineTracer:
    """
    Lightweight telemetry and performance tracer for the RAG retrieval pipeline.
    Tracks execution times, candidate counts, and score distributions.
    """
    def __init__(self, query: str):
        self.query = query
        self.start_time = time.time()
        self.stages: Dict[str, Any] = {}
        
    def log_vector_stage(self, candidates_count: int, duration_ms: float):
        self.stages["vector"] = {
            "candidates": candidates_count,
            "latency_ms": round(duration_ms, 2)
        }
        
    def log_bm25_stage(self, candidates_count: int, duration_ms: float):
        self.stages["bm25"] = {
            "candidates": candidates_count,
            "latency_ms": round(duration_ms, 2)
        }
        
    def log_rrf_stage(self, unique_count: int, overlap_count: int, duration_ms: float):
        self.stages["rrf_fusion"] = {
            "unique_chunks": unique_count,
            "overlap_chunks": overlap_count,
            "latency_ms": round(duration_ms, 2)
        }
        
    def log_rerank_stage(self, in_count: int, out_count: int, top_scores: List[float], duration_ms: float):
        self.stages["reranking"] = {
            "in_count": in_count,
            "out_count": out_count,
            "top_scores": [round(s, 3) for s in top_scores[:3]],
            "latency_ms": round(duration_ms, 2)
        }
        
    def finish_and_log(self):
        total_latency = (time.time() - self.start_time) * 1000
        self.stages["total_retrieval_ms"] = round(total_latency, 2)
        
        v_info = self.stages.get("vector", {})
        b_info = self.stages.get("bm25", {})
        r_info = self.stages.get("rrf_fusion", {})
        rr_info = self.stages.get("reranking", {})
        
        logger.info(
            f"[RAG Trace] Query: '{self.query[:50]}' | "
            f"Vector: {v_info.get('candidates', 0)} chunks ({v_info.get('latency_ms', 0)}ms) | "
            f"BM25: {b_info.get('candidates', 0)} chunks ({b_info.get('latency_ms', 0)}ms) | "
            f"RRF: {r_info.get('unique_chunks', 0)} merged (overlap: {r_info.get('overlap_chunks', 0)}) | "
            f"Reranked: {rr_info.get('out_count', 0)} chunks (scores: {rr_info.get('top_scores', [])}) | "
            f"Total Time: {self.stages['total_retrieval_ms']}ms"
        )
        return self.stages
