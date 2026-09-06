from typing import List, Dict, Any, AsyncGenerator, Callable, Optional, Awaitable
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from src.generation.base_provider import LLMProvider
from src.config import settings
import logging
import re
import json
import uuid

logger = logging.getLogger(__name__)

class OpenRouterProvider(LLMProvider):
    """
    OpenRouter LLM Provider for the NABL RAG Agent.
    Supports high-speed streaming, on-demand visual audits, and multi-model routing
    (e.g., google/gemini-2.0-flash-001, meta-llama/llama-3.3-70b-instruct, anthropic/claude-3.5-sonnet).
    """
    def __init__(self, model_name: Optional[str] = None):
        if not settings.OPENROUTER_API_KEY:
            logger.warning("OPENROUTER_API_KEY is not set in environment or .env!")
            
        self._model_name = model_name or settings.OPENROUTER_CHAT_MODEL
        logger.info(f"Initializing OpenRouterProvider with model: {self._model_name}")
        
        default_headers = {
            "HTTP-Referer": "https://nabl-rag-agent.local",
            "X-Title": "NABL RAG Agent"
        }
        
        self.llm = ChatOpenAI(
            base_url=settings.OPENROUTER_BASE_URL,
            api_key=settings.OPENROUTER_API_KEY or "missing_key",
            model=self._model_name,
            default_headers=default_headers,
            temperature=0.1,
            max_tokens=2000,
        )

    @property
    def provider_name(self) -> str:
        return "openrouter"

    @property
    def model_name(self) -> str:
        return self._model_name
        
    def _build_system_prompt(self) -> SystemMessage:
        system_prompt = """You are a professional NABL (National Accreditation Board for Testing and
Calibration Laboratories) compliance assistant used by working lab QA staff.
They need a fast, decisive answer they can act on — not a detailed audit essay.

## Behaviour rules

1. **Greetings** — respond normally and briefly.
2. **NABL & Document Queries**:
   - **Document / Report Audits**: If the user uploads or asks to evaluate a specific test report or document, retrieve clauses using search_nabl_database and respond using the **OUTPUT TEMPLATE** below.
   - **General NABL Questions**: If the user asks a general conceptual question (e.g., *"What makes a lab accredited?"*, *"What is a ULR number?"*), retrieve clauses using search_nabl_database and answer directly, clearly, and concisely in bullet points.
      Do not force the document audit template for general knowledge questions.
3. **Irrelevant / Out-of-Domain queries** — If the query is unrelated to NABL standards or if no relevant context is found in the knowledge base, reply politely: "I don't have information about this in my NABL knowledge base."
4. **Never** paste raw retrieved chunks, full clause text, or document dumps
   into the response. Reference them by document number and clause only
   (e.g. "ISO/IEC 17025:2017, 7.8.6.1"), never quote more than one short
   phrase per clause.
5. **Direct Output Only**: NEVER output internal monologue, 
   reasoning traces, self-corrections, or thought processes. 
   Output ONLY the final response.
6. **Visual Document Verification**: When evaluating an uploaded report, call `inspect_document_visuals` if you need to verify whether physical artifacts like the NABL symbol/logo, authorized signature, or lab stamp are actually present on the document.

## Evaluation rules

- Judge only against what is explicitly retrieved as a mandatory ("shall")
  requirement. Do not invent requirements.
- Give the document credit for what it has before listing what it lacks.
- Rank findings by real-world impact: something that could cause the report
  to be rejected or misused (wrong dates, unsupported pass/fail claim,
  missing signatory) outranks a formatting nicety.
- Report **at most 5 findings total**. If there are more, list only the
  5 most significant and say how many others exist in one line — do not
  list them all.
- State each finding **once**. Do not repeat it in a different section.
- Do not explain your reasoning process. State the gap and the rule it
  breaks — one sentence each, not a paragraph.

## OUTPUT TEMPLATE (always use this exact structure)

**Verdict: [COMPLIANT / NON-COMPLIANT / PARTIALLY COMPLIANT]**
[One sentence — the single biggest reason for this verdict, if not fully compliant]

**Critical issues** (blocking — fix before release)
- [Issue] — [Clause reference]
- [Issue] — [Clause reference]
(omit this section entirely if there are none — do not write "none found")

**Minor observations** (should improve, not blocking)
- [Issue] — [Clause reference]
(omit this section entirely if there are none)

**What's already in order**
- [2-4 short bullets max, group similar items together]

**Recommended action**
[1-2 sentences, plain language, telling the lab exactly what to do next]

## Tone

- Plain language over ISO jargon. If you must name a clause, translate what
  it means in one short phrase (e.g. "7.8.6.1 — the report must state how
  pass/fail was decided").
- Assume the reader is a lab technician or QA manager, not an auditor or
  lawyer.
- No hedging language ("it appears," "this may be considered"). Say what
  is or isn't there, plainly.
- Keep the entire response under ~200 words unless the user explicitly asks
  for a detailed breakdown."""
        return SystemMessage(content=system_prompt)
        
    def _format_context(self, context: List[Dict[str, Any]]) -> str:
        if not context:
            return "No matching NABL compliance rules found in knowledge base."
        formatted = []
        for idx, c in enumerate(context):
            formatted.append(f"[Source {idx + 1}: {c.get('source')}]\nContent: {c.get('content')}")
        return "\n\n---\n\n".join(formatted)

    async def _run_visual_inspection(self, uploaded_images: Optional[List[str]], aspect: str) -> str:
        if not uploaded_images:
            return "No uploaded document image is available in this session to inspect."
            
        try:
            logger.info(f"Running OpenRouter visual inspection for aspect: '{aspect}' on {len(uploaded_images)} image(s)...")
            image_content = [{
                "type": "text", 
                "text": (
                    f"You are a forensic laboratory document inspector. Visually inspect this test report image for: {aspect}.\n"
                    "Report concisely:\n"
                    "1. NABL Accredited Symbol / Logo: [Present / Missing] and exact position on page.\n"
                    "2. Authorized Signatory: [Present / Missing] (state if handwritten ink signature, digital signature stamp, or blank).\n"
                    "3. Lab Official Stamp / Seal: [Present / Missing].\n"
                    "4. QR Code / Barcode: [Present / Missing].\n"
                    "Be strictly factual, concise, and do not invent details."
                )
            }]
            
            # Attach first 2 pages max
            for img in uploaded_images[:2]:
                image_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img}"}
                })
                
            vision_msg = HumanMessage(content=image_content)
            vision_res = await self.llm.ainvoke([vision_msg])
            findings = re.sub(r'<think>.*?</think>', '', vision_res.content, flags=re.DOTALL).strip()
            return f"[Visual Document Inspection Findings]:\n{findings}"
        except Exception as e:
            logger.error(f"Visual inspection failed on OpenRouter: {e}")
            return f"Visual inspection could not be completed: {str(e)}"

    def generate_response(self, query: str, chat_history: List[Dict[str, str]] = None, search_callback: Optional[Callable] = None, uploaded_images: Optional[List[str]] = None) -> str:
        raise NotImplementedError("Sync generation not implemented for OpenRouterProvider. Use generate_response_stream.")

    async def generate_response_stream(self, query: str, chat_history: List[Dict[str, str]] = None, search_callback: Optional[Callable[[str], Awaitable[List[Dict[str, Any]]]]] = None, uploaded_images: Optional[List[str]] = None) -> AsyncGenerator[str, None]:
        messages = [self._build_system_prompt()]
        
        # Add history
        if chat_history:
            for msg in chat_history:
                if msg.get("role") == "user":
                    messages.append(HumanMessage(content=msg.get("content")))
                elif msg.get("role") == "assistant":
                    messages.append(AIMessage(content=msg.get("content")))
                    
        # Send pure text query into the chat conversation
        messages.append(HumanMessage(content=query))
        
        # Define tools
        tools_list = []
        if search_callback:
            @tool
            def search_nabl_database(search_query: str) -> str:
                """Search the NABL database for compliance rules, policies, and guidelines."""
                pass
            tools_list.append(search_nabl_database)
            
        if uploaded_images:
            @tool
            def inspect_document_visuals(aspect: str = "NABL logo, signature, and stamp") -> str:
                """Visually inspects the uploaded test report or calibration certificate image to verify physical artifacts like: NABL Accredited Symbol / Logo presence and location, Authorized Signatory handwritten signature or digital signature stamp, Laboratory official stamp/seal, and QR codes/barcodes."""
                pass
            tools_list.append(inspect_document_visuals)
            
        if tools_list:
            llm_with_tools = self.llm.bind_tools(tools_list)
            
            logger.info("OpenRouter agent checking if tool call is needed...")
            response = await llm_with_tools.ainvoke(messages)
            
            tool_calls = list(getattr(response, "tool_calls", None) or [])
            
            # Robust fallback for text-based tool calls
            if not tool_calls and response.content:
                raw_text = response.content
                if "<tool_call>" in raw_text or "<function=" in raw_text:
                    logger.info("Detected text-based <tool_call>. Parsing arguments...")
                    param_match = re.search(r'<parameter=([^>]+)>(.*?)</parameter>', raw_text, flags=re.DOTALL)
                    fn_match = re.search(r'<function=([^>]+)>', raw_text)
                    fn_name = fn_match.group(1).strip() if fn_match else "search_nabl_database"
                    if param_match:
                        p_name = param_match.group(1).strip()
                        p_val = param_match.group(2).strip()
                        tool_calls.append({
                            "id": f"call_{uuid.uuid4().hex[:8]}",
                            "name": fn_name,
                            "args": {p_name: p_val}
                        })
                    else:
                        for m in re.findall(r'<tool_call>(.*?)</tool_call>', raw_text, flags=re.DOTALL):
                            try:
                                data = json.loads(m.strip())
                                if isinstance(data, dict):
                                    tool_calls.append({
                                        "id": f"call_{uuid.uuid4().hex[:8]}",
                                        "name": data.get("name", fn_name),
                                        "args": data.get("arguments") or data.get("args") or {}
                                    })
                            except Exception:
                                pass
                                
            if tool_calls:
                logger.info(f"OpenRouter executing tool calls: {tool_calls}")
                ai_msg = AIMessage(content="", tool_calls=tool_calls)
                messages.append(ai_msg)
                
                # Deduplicate tool calls
                unique_tool_calls = []
                seen_calls = set()
                for tc in tool_calls:
                    call_sig = f"{tc['name']}_{json.dumps(tc.get('args', {}), sort_keys=True)}"
                    if call_sig not in seen_calls:
                        seen_calls.add(call_sig)
                        unique_tool_calls.append(tc)
                
                for tool_call in unique_tool_calls:
                    if tool_call["name"] == "search_nabl_database" and search_callback:
                        search_query = tool_call["args"].get("search_query", query) if isinstance(tool_call["args"], dict) else query
                        context_chunks = await search_callback(search_query)
                        formatted_context = self._format_context(context_chunks)
                        
                        messages.append(ToolMessage(
                            tool_call_id=tool_call["id"],
                            content=formatted_context
                        ))
                    elif tool_call["name"] == "inspect_document_visuals":
                        aspect = tool_call["args"].get("aspect", "all") if isinstance(tool_call["args"], dict) else "all"
                        visual_findings = await self._run_visual_inspection(uploaded_images, aspect)
                        
                        messages.append(ToolMessage(
                            tool_call_id=tool_call["id"],
                            content=visual_findings
                        ))
            else:
                # Direct answer without tools
                if response.content:
                    cleaned_content = re.sub(r'<think>.*?(</think>|$)', '', response.content, flags=re.DOTALL)
                    cleaned_content = re.sub(r'<tool_call>.*?(</tool_call>|$)', '', cleaned_content, flags=re.DOTALL).strip()
                    if cleaned_content:
                        yield cleaned_content
                    else:
                        yield response.content.replace("<think>", "").replace("</think>", "").strip()
                    return
                    
        # Step 2: Stream final response
        in_tag = False
        buffer = ""
        
        async for chunk in self.llm.astream(messages):
            if not chunk.content:
                continue
                
            buffer += chunk.content
            
            while buffer:
                if not in_tag:
                    think_start = buffer.find("<think>")
                    tool_start = buffer.find("<tool_call>")
                    
                    starts = [pos for pos in [think_start, tool_start] if pos != -1]
                    if starts:
                        first_start = min(starts)
                        if first_start > 0:
                            yield buffer[:first_start]
                        tag_len = len("<think>") if first_start == think_start else len("<tool_call>")
                        buffer = buffer[first_start + tag_len:]
                        in_tag = True
                    else:
                        last_lt = buffer.rfind("<")
                        if last_lt != -1 and ("<think>".startswith(buffer[last_lt:]) or "<tool_call>".startswith(buffer[last_lt:])):
                            if last_lt > 0:
                                yield buffer[:last_lt]
                                buffer = buffer[last_lt:]
                            break
                        else:
                            yield buffer
                            buffer = ""
                            break
                else:
                    think_end = buffer.find("</think>")
                    tool_end = buffer.find("</tool_call>")
                    ends = [pos for pos in [think_end, tool_end] if pos != -1]
                    if ends:
                        first_end = min(ends)
                        tag_len = len("</think>") if first_end == think_end else len("</tool_call>")
                        buffer = buffer[first_end + tag_len:]
                        in_tag = False
                    else:
                        break
                        
        if buffer:
            if in_tag:
                cleaned = re.sub(r'<think>.*?</think>', '', buffer, flags=re.DOTALL)
                cleaned = re.sub(r'<tool_call>.*?</tool_call>', '', cleaned, flags=re.DOTALL)
                cleaned = cleaned.replace("<think>", "").replace("</think>", "").replace("<tool_call>", "").replace("</tool_call>", "").strip()
                if cleaned:
                    yield cleaned
            else:
                yield buffer
