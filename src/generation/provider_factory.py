from typing import Dict, Type, Optional
from src.generation.base_provider import LLMProvider
from src.generation.ollama_provider import OllamaProvider
from src.generation.groq_provider import GroqProvider
from src.generation.openrouter_provider import OpenRouterProvider
from src.config import settings
import logging

logger = logging.getLogger(__name__)

# Dynamic registry mapping provider names to their classes
PROVIDER_REGISTRY: Dict[str, Type[LLMProvider]] = {
    "ollama": OllamaProvider,
    "groq": GroqProvider,
    "openrouter": OpenRouterProvider,
    "open_router": OpenRouterProvider,
}

def register_provider(name: str, provider_cls: Type[LLMProvider]):
    """Dynamically register any custom or new LLM provider at runtime."""
    key = name.lower().strip()
    PROVIDER_REGISTRY[key] = provider_cls
    logger.info(f"Registered new LLM provider: '{key}' -> {provider_cls.__name__}")

def get_llm_provider(provider_name: Optional[str] = None, model_name: Optional[str] = None) -> LLMProvider:
    """
    Dynamically resolve and instantiate an LLM provider.
    If provider_name is not specified, defaults to settings.LLM_PROVIDER.
    """
    selected = (provider_name or settings.LLM_PROVIDER).lower().strip()
    
    if selected not in PROVIDER_REGISTRY:
        available = ", ".join(sorted(set(PROVIDER_REGISTRY.keys())))
        raise ValueError(f"Unknown LLM provider: '{selected}'. Currently available providers: [{available}]")
        
    provider_cls = PROVIDER_REGISTRY[selected]
    return provider_cls(model_name=model_name) if model_name else provider_cls()
