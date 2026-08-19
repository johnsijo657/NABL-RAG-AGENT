from langchain_ollama import OllamaEmbeddings
from typing import List
from src.config import settings
import logging

logger = logging.getLogger(__name__)

# Initialize the Ollama embeddings model globally so it's ready to use
# This assumes Ollama is running locally at OLLAMA_BASE_URL
embeddings_model = OllamaEmbeddings(
    model=settings.OLLAMA_EMBED_MODEL,
    base_url=settings.OLLAMA_BASE_URL
)

def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Generates vector embeddings for a list of text strings using Ollama.
    """
    try:
        logger.info(f"Generating embeddings for {len(texts)} chunks using {settings.OLLAMA_EMBED_MODEL}...")
        # generate embeddings
        return embeddings_model.embed_documents(texts)
    except Exception as e:
        logger.error(f"Failed to generate embeddings: {str(e)}")
        raise
