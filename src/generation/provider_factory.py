from src.generation.base_provider import LLMProvider
from src.generation.ollama_provider import OllamaProvider
from src.generation.groq_provider import GroqProvider

def get_llm_provider(provider_name: str) -> LLMProvider:
    provider_name = provider_name.lower().strip()
    if provider_name == "ollama":
        return OllamaProvider()
    elif provider_name == "groq":
        return GroqProvider()
    else:
        raise ValueError(f"Unknown LLM provider: {provider_name}")
