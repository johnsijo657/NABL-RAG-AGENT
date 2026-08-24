from typing import List, Dict, Any, AsyncGenerator, Callable, Optional, Awaitable
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from src.generation.base_provider import LLMProvider
from src.config import settings
import logging

logger = logging.getLogger(__name__)

class GroqProvider(LLMProvider):
    def __init__(self):
        logger.info(f"Initializing GroqProvider with model: {settings.GROQ_CHAT_MODEL}")
        self.llm = ChatGroq(
            api_key=settings.GROQ_API_KEY,
            model=settings.GROQ_CHAT_MODEL,
        )
        
    def _build_system_prompt(self) -> SystemMessage:
        system_prompt = """You are a helpful and polite professional assistant specialized in NABL (National Accreditation Board for Testing and Calibration Laboratories) compliance and guidelines.
Your guidelines:
1. GREETINGS: Respond normally and politely to general greetings.
2. NABL & DOCUMENT ANALYSIS: If the user asks about NABL or uploads a document, analyze the document carefully. If you need NABL rules, use the search_nabl_database tool. Give a well-presented answer in simple words.
   - Evaluate FAIRLY: First identify what the document DOES have (e.g., ULR number, test methods, authorized signatory, lab details). Then check if anything critical is genuinely missing based on the rules.
   - Only cite a specific violation if the rules explicitly state it is a mandatory requirement AND the uploaded document clearly lacks it.
   - Be balanced: focus on what is actually required for the specific type of document being evaluated.
3. IRRELEVANT: If the query is completely unrelated to NABL, document review, or the rules, you must reply politely: "well I don't know about it . sorry". 
4. DO NOT output raw database chunks directly to the user."""
        return SystemMessage(content=system_prompt)
        
    def _format_context(self, context: List[Dict[str, Any]]) -> str:
        if not context:
            return "No context found."
        return "\n\n---\n\n".join([f"Source: {c.get('source')}\nContent: {c.get('content')}" for c in context])

    def generate_response(self, query: str, chat_history: List[Dict[str, str]] = None, search_callback: Optional[Callable] = None, uploaded_images: Optional[List[str]] = None) -> str:
        raise NotImplementedError("Sync generation not fully implemented for Groq with Tool Calling.")

    async def generate_response_stream(self, query: str, chat_history: List[Dict[str, str]] = None, search_callback: Optional[Callable[[str], Awaitable[List[Dict[str, Any]]]]] = None, uploaded_images: Optional[List[str]] = None) -> AsyncGenerator[str, None]:
        
        messages = [self._build_system_prompt()]
        
        # Add history
        if chat_history:
            for msg in chat_history:
                if msg.get("role") == "user":
                    messages.append(HumanMessage(content=msg.get("content")))
                elif msg.get("role") == "assistant":
                    messages.append(AIMessage(content=msg.get("content")))
                    
        # Add current query and images
        content_parts = [{"type": "text", "text": query}]
        if uploaded_images:
            for img in uploaded_images:
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img}"}
                })
        
        messages.append(HumanMessage(content=content_parts))
        
        # Bind the tool if callback is provided
        if search_callback:
            @tool
            def search_nabl_database(search_query: str) -> str:
                """Search the NABL database for compliance rules, policies, and guidelines."""
                pass # LangChain needs this for the schema, but we intercept it
            
            llm_with_tools = self.llm.bind_tools([search_nabl_database])
            
            # Step 1: Check if the model wants to call a tool
            logger.info("Groq checking if tool call is needed...")
            response = await llm_with_tools.ainvoke(messages)
            
            if response.tool_calls:
                logger.info(f"Groq decided to call tools: {response.tool_calls}")
                messages.append(response) # Append AIMessage with tool_calls
                
                for tool_call in response.tool_calls:
                    if tool_call["name"] == "search_nabl_database":
                        search_query = tool_call["args"].get("search_query", query)
                        context_chunks = await search_callback(search_query)
                        formatted_context = self._format_context(context_chunks)
                        
                        messages.append(ToolMessage(
                            tool_call_id=tool_call["id"],
                            content=formatted_context
                        ))
            else:
                # If no tool calls, just yield the text response
                if response.content:
                    import re
                    # Strip <think>...</think> from the direct response
                    cleaned_content = re.sub(r'<think>.*?(</think>|$)', '', response.content, flags=re.DOTALL).strip()
                    if cleaned_content:
                        yield cleaned_content
                    return
                    
        # Step 2: Stream final response
        in_think_tag = False
        buffer = ""
        
        async for chunk in self.llm.astream(messages):
            if not chunk.content:
                continue
                
            buffer += chunk.content
            
            while True:
                if not in_think_tag:
                    think_start = buffer.find("<think>")
                    if think_start != -1:
                        if think_start > 0:
                            yield buffer[:think_start]
                        buffer = buffer[think_start + len("<think>"):]
                        in_think_tag = True
                    else:
                        # Yield safe parts
                        last_lt = buffer.rfind("<")
                        if last_lt != -1 and "<think>".startswith(buffer[last_lt:]):
                            if last_lt > 0:
                                yield buffer[:last_lt]
                                buffer = buffer[last_lt:]
                        else:
                            yield buffer
                            buffer = ""
                        break # Wait for more chunks
                else:
                    think_end = buffer.find("</think>")
                    if think_end != -1:
                        buffer = buffer[think_end + len("</think>"):]
                        in_think_tag = False
                    else:
                        # Drop the thought chunks
                        last_lt = buffer.rfind("<")
                        if last_lt != -1 and "</think>".startswith(buffer[last_lt:]):
                            buffer = buffer[last_lt:]
                        else:
                            buffer = ""
                        break # Wait for more chunks
                        
        if buffer and not in_think_tag:
            yield buffer
