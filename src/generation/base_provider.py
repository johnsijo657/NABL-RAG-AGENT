from abc import ABC, abstractmethod
from typing import List, Dict, Any, AsyncGenerator, Callable, Optional, Awaitable

class LLMProvider(ABC):
    """
    Abstract base class for all LLM providers (Strategy Pattern).
    """
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Returns the identifier name of this provider (e.g. 'ollama', 'groq', 'openrouter')."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Returns the active model name used by this provider."""
        pass

    @abstractmethod
    def generate_response(self, query: str, chat_history: List[Dict[str, str]] = None, search_callback: Optional[Callable[[str], Awaitable[List[Dict[str, Any]]]]] = None, uploaded_images: Optional[List[str]] = None) -> str:
        """Generates a synchronous response."""
        pass
        
    @abstractmethod
    async def generate_response_stream(self, query: str, chat_history: List[Dict[str, str]] = None, search_callback: Optional[Callable[[str], Awaitable[List[Dict[str, Any]]]]] = None, uploaded_images: Optional[List[str]] = None) -> AsyncGenerator[str, None]:
        """Generates a streaming response (for real-time UI)."""
        pass
