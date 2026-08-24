from abc import ABC, abstractmethod
from typing import List, Dict, Any, AsyncGenerator, Callable, Optional, Awaitable

class LLMProvider(ABC):
    """
    Abstract base class for all LLM providers (Strategy Pattern).
    """
    @abstractmethod
    def generate_response(self, query: str, chat_history: List[Dict[str, str]] = None, search_callback: Optional[Callable[[str], Awaitable[List[Dict[str, Any]]]]] = None, uploaded_images: Optional[List[str]] = None) -> str:
        """Generates a synchronous response."""
        pass
        
    @abstractmethod
    async def generate_response_stream(self, query: str, chat_history: List[Dict[str, str]] = None, search_callback: Optional[Callable[[str], Awaitable[List[Dict[str, Any]]]]] = None, uploaded_images: Optional[List[str]] = None) -> AsyncGenerator[str, None]:
        """Generates a streaming response (for real-time UI)."""
        pass
