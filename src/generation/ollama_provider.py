from typing import List, Dict, Any, AsyncGenerator, Callable, Optional
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from src.generation.base_provider import LLMProvider
from src.config import settings
import logging

logger = logging.getLogger(__name__)

class OllamaProvider(LLMProvider):
    def __init__(self, model_name: Optional[str] = None):
        self._model_name = model_name or settings.OLLAMA_CHAT_MODEL
        logger.info(f"Initializing OllamaProvider with model: {self._model_name}")
        self.llm = ChatOllama(
            model=self._model_name,
            base_url=settings.OLLAMA_BASE_URL,
            # temperature=0.0 # Low temp for factual RAG
        )

    @property
    def provider_name(self) -> str:
        return "ollama"

    @property
    def model_name(self) -> str:
        return self._model_name
        
    def _build_messages(self, query: str, context: List[Dict[str, Any]], chat_history: List[Dict[str, str]]) -> List[Any]:
        # Construct system prompt with context
        if not context:
            context_text = "No matching NABL compliance rules found in knowledge base."
        else:
            formatted = []
            for idx, c in enumerate(context):
                formatted.append(f"[Source {idx + 1}: {c.get('source')}]\nContent: {c.get('content')}")
            context_text = "\n\n---\n\n".join(formatted)
        
        system_prompt = f"""You are a helpful and polite professional assistant specialized in NABL (National Accreditation Board for Testing and Calibration Laboratories) compliance and guidelines.
Your guidelines:
1. GREETINGS: Respond normally and politely to general greetings.
2. NABL & DOCUMENT ANALYSIS: If the user asks about NABL or uploads a document, analyze the document and the provided context carefully to give a well-presented answer in simple words based ONLY on the provided context. Cite sources by standard name and clause (e.g. ISO/IEC 17025:2017, Cl 7.8).
   - Evaluate FAIRLY: First identify what the document DOES have (e.g., ULR number, test methods, authorized signatory, lab details). Then check if anything critical is genuinely missing based on the context provided.
   - Only cite a specific violation if the context explicitly states it is a mandatory requirement AND the uploaded document clearly lacks it. Do NOT assume something is missing just because the document does not explicitly mention it in detail.
   - Be balanced: a document can be compliant even if it does not contain every single element mentioned in NABL policies. Focus on what is actually required for the specific type of document being evaluated.
3. IRRELEVANT: If the query is completely unrelated to NABL, document review, or if no relevant context is found in the database, you must reply politely: "I don't have information about this in my NABL knowledge base."
4. DO NOT output raw database chunks directly to the user.
CONTEXT:
{context_text}"""

        messages = [SystemMessage(content=system_prompt)]
        
        # Add history
        if chat_history:
            for msg in chat_history:
                if msg.get("role") == "user":
                    messages.append(HumanMessage(content=msg.get("content")))
                elif msg.get("role") == "assistant":
                    messages.append(AIMessage(content=msg.get("content")))
                    
        # Add current query
        messages.append(HumanMessage(content=query))
        return messages

    def generate_response(self, query: str, chat_history: List[Dict[str, str]] = None, search_callback: Optional[Callable] = None, uploaded_images: Optional[List[str]] = None) -> str:
        # Sync version assumes no context since callback is async
        messages = self._build_messages(query, [], chat_history or [])
        response = self.llm.invoke(messages)
        return response.content
        
    async def generate_response_stream(self, query: str, chat_history: List[Dict[str, str]] = None, search_callback: Optional[Callable] = None, uploaded_images: Optional[List[str]] = None) -> AsyncGenerator[str, None]:
        intent = await self.route_query(query)
        context = []
        if intent == "SEARCH" and search_callback:
            context = await search_callback(query)
            
        messages = self._build_messages(query, context, chat_history or [])
        async for chunk in self.llm.astream(messages):
            yield chunk.content
            
    async def route_query(self, query: str) -> str:
        """
        A lightweight router to classify intent:
        Returns "CHAT" for greetings/conversational requests.
        Returns "SEARCH" for questions requiring NABL knowledge.
        """
        router_prompt = f"""You are a router classifying user queries.
If the query is a simple greeting, conversational pleasantry, or asking what you can do (e.g., "Hi", "Hello", "Who are you"), output exactly: CHAT
If the query asks a specific question, mentions documents, NABL, standards, or requires factual lookup, output exactly: SEARCH

If the query contains BOTH a greeting AND a specific question, output exactly: SEARCH

User Query: "{query}"
Output only CHAT or SEARCH."""
        
        response = await self.llm.ainvoke([SystemMessage(content=router_prompt)])
        result = response.content.strip().upper()
        
        # Prioritize SEARCH over CHAT for mixed responses
        if "SEARCH" in result:
            return "SEARCH"
        return "CHAT"
