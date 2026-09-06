from typing import List, Dict, Any, AsyncGenerator
from src.generation.base_provider import LLMProvider
import logging

logger = logging.getLogger(__name__)

class CloudProvider(LLMProvider):
    """
    Stub for a future Cloud Provider (e.g., OpenAI, Anthropic, Google Vertex).
    Demonstrates the Strategy Pattern.
    """
    def __init__(self, model_name: str = "cloud-stub"):
        logger.info("Initializing CloudProvider (Stub)")
        self._model_name = model_name

    @property
    def provider_name(self) -> str:
        return "cloud"

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate_response(self, query: str, context: List[Dict[str, Any]], chat_history: List[Dict[str, str]] = None) -> str:
        return "This is a response from the Cloud Provider stub."
        
    async def generate_response_stream(self, query: str, context: List[Dict[str, Any]], chat_history: List[Dict[str, str]] = None) -> AsyncGenerator[str, None]:
        yield "This "
        yield "is "
        yield "a "
        yield "stream "
        yield "from "
        yield "the "
        yield "Cloud "
        yield "Provider "
        yield "stub."
