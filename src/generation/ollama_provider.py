from typing import List, Dict, Any, AsyncGenerator
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from src.generation.base_provider import LLMProvider
from src.config import settings
import logging

logger = logging.getLogger(__name__)

class OllamaProvider(LLMProvider):
    def __init__(self):
        logger.info(f"Initializing OllamaProvider with model: {settings.OLLAMA_CHAT_MODEL}")
        self.llm = ChatOllama(
            model=settings.OLLAMA_CHAT_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
            # temperature=0.0 # Low temp for factual RAG
        )
        
    def _build_messages(self, query: str, context: List[Dict[str, Any]], chat_history: List[Dict[str, str]]) -> List[Any]:
        # Construct system prompt with context
        context_text = "\n\n---\n\n".join([f"Source: {c.get('source')}\nContent: {c.get('content')}" for c in context])
        
        system_prompt = f"""You are a helpful and polite professional assistant specialized in NABL (National Accreditation Board for Testing and Calibration Laboratories) compliance and guidelines.
Your guidelines:
1. GREETINGS: Respond normally and politely to general greetings.
2. NABL & DOCUMENT ANALYSIS: If the user asks about NABL or uploads a document, analyze the document and the provided context carefully to give a well-reasoned answer based ONLY on the provided context. 
3. IRRELEVANT: If the query is completely unrelated to NABL, document review, or the provided context, you must reply politely: "well I don't know about it . sorry". 
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

    def generate_response(self, query: str, context: List[Dict[str, Any]], chat_history: List[Dict[str, str]] = None) -> str:
        messages = self._build_messages(query, context, chat_history or [])
        response = self.llm.invoke(messages)
        return response.content
        
    async def generate_response_stream(self, query: str, context: List[Dict[str, Any]], chat_history: List[Dict[str, str]] = None) -> AsyncGenerator[str, None]:
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
        
    async def expand_search_query(self, user_query: str, document_text: str) -> str:
        """
        Agentic Query Expansion: If a user uploads a document, generating a search query based on the raw prompt 
        (e.g., 'Is this compliant?') is useless for vector search. This method extracts key entities from the 
        document (up to 1500 chars to save time) and combines them with the user's intent to build a rich search query.
        """
        # Truncate document text to avoid massive token overhead for a quick routing decision
        truncated_doc = document_text[:1500]
        
        expansion_prompt = f"""You are an expert search query generator.
The user asked a question about an uploaded document.
User's Question: "{user_query}"
Document Extract: "{truncated_doc}"

Based on the question and the document extract, generate a concise, highly targeted search query to find relevant rules in the NABL database.
Include important keywords from the document (like ULR, test methods, subject matter) and the user's core intent.
Output ONLY the search query. Do not add quotes, prefixes, or explanations."""
        
        response = await self.llm.ainvoke([SystemMessage(content=expansion_prompt)])
        return response.content.strip()
