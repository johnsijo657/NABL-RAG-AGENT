from abc import ABC, abstractmethod
from typing import List, Dict, Any, AsyncGenerator

class LLMProvider(ABC):
    """
    Abstract base class for all LLM providers (Strategy Pattern).
    """
    @abstractmethod
    def generate_response(self, query: str, context: List[Dict[str, Any]], chat_history: List[Dict[str, str]] = None) -> str:
        """Generates a synchronous response."""
        pass
        
    @abstractmethod
    async def generate_response_stream(self, query: str, context: List[Dict[str, Any]], chat_history: List[Dict[str, str]] = None) -> AsyncGenerator[str, None]:
        """Generates a streaming response (for real-time UI)."""
        pass
