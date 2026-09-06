import time
import threading
from collections import defaultdict
from typing import Dict, List
from fastapi import HTTPException, status
import logging

logger = logging.getLogger(__name__)

class SlidingWindowRateLimiter:
    """
    High-performance in-memory sliding-window rate limiter.
    Tracks timestamped requests per API key within a 60-second rolling window.
    Enforces per-key custom limits and responds with HTTP 429 and Retry-After header.
    """
    def __init__(self, window_seconds: float = 60.0):
        self.window_seconds = window_seconds
        self._requests: Dict[int, List[float]] = defaultdict(list)
        self._lock = threading.Lock()

    def check_rate_limit(self, api_key_id: int, limit_per_minute: int) -> None:
        """
        Check if the specified API key has exceeded its allowed rate limit.
        Executed upon request arrival before running any RAG or LLM operations.
        """
        now = time.time()
        
        with self._lock:
            timestamps = self._requests[api_key_id]
            # Prune timestamps outside the rolling window
            cutoff = now - self.window_seconds
            recent = [t for t in timestamps if t > cutoff]
            
            if len(recent) >= limit_per_minute:
                oldest = recent[0]
                retry_after = max(1, int(self.window_seconds - (now - oldest)) + 1)
                logger.warning(
                    f"API Key {api_key_id} rate limit exceeded: "
                    f"{len(recent)}/{limit_per_minute} req/min. Retry-After: {retry_after}s"
                )
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=f"Rate limit exceeded. You are allowed {limit_per_minute} requests per minute.",
                    headers={"Retry-After": str(retry_after)}
                )
            
            # Record current request arrival
            recent.append(now)
            self._requests[api_key_id] = recent

    def reset(self, api_key_id: int = None) -> None:
        """Reset rate limit history for testing or manual overrides."""
        with self._lock:
            if api_key_id is not None:
                self._requests.pop(api_key_id, None)
            else:
                self._requests.clear()

# Global rate limiter singleton
rate_limiter = SlidingWindowRateLimiter()
