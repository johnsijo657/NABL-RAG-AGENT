import chainlit as cl
from typing import Dict, Optional, List, Any
import sys
import logging
from pathlib import Path

# Ensure project root is in sys.path
project_root = str(Path(__file__).parent.parent.absolute())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

logger = logging.getLogger(__name__)

from src.database import SessionLocal
from src.api.auth import verify_password, get_user
from src.retrieval.hybrid_search import hybrid_search
from src.retrieval.reranker import rerank_results
from src.retrieval.tracing import RAGPipelineTracer
from src.generation.provider_factory import get_llm_provider
from src.models import QueryLog, ChatSession, User, Feedback
from src.config import settings
import uuid
from src.ingestion.loader import load_pdf, PDFTooLargeError
import time
import shutil
from datetime import datetime

from chainlit.data import BaseDataLayer
from chainlit.types import Pagination, ThreadFilter, PaginatedResponse
from chainlit.auth.cookie import OAuth2PasswordBearerWithCookie
from fastapi.openapi.models import OAuth2, OAuthFlows, OAuthFlowPassword

# Safeguard: prevent Chainlit's internal FastAPI app from crashing if /docs or /openapi.json is accessed on port 8002
if not hasattr(OAuth2PasswordBearerWithCookie, "model"):
    OAuth2PasswordBearerWithCookie.model = OAuth2(
        flows=OAuthFlows(password=OAuthFlowPassword(tokenUrl="/login"))
    )

class DummyDataLayer(BaseDataLayer):

    """
    A minimal data layer to satisfy Chainlit's internal requirements 
    when authentication is enabled, preventing 'Error fetching threads'.
    """
    async def get_user(self, identifier: str):
        return cl.PersistedUser(id=identifier, identifier=identifier, createdAt="2024-01-01T00:00:00Z")
    
    async def create_user(self, user: cl.User):
        return cl.PersistedUser(id=user.identifier, identifier=user.identifier, createdAt="2024-01-01T00:00:00Z")
        
    async def list_threads(self, pagination: Pagination, filter: ThreadFilter):
        return PaginatedResponse(data=[], pageInfo={"hasNextPage": False, "endCursor": None, "startCursor": None})

    async def build_debug_url(self) -> str: return ""
    async def close(self) -> None: pass
    async def create_element(self, element) -> None: pass
    async def create_step(self, step_dict) -> None: pass
    async def delete_element(self, element_id, thread_id=None) -> None: pass
    async def delete_feedback(self, feedback_id) -> None: pass
    async def delete_step(self, step_id) -> None: pass
    async def delete_thread(self, thread_id) -> None: pass
    async def get_element(self, thread_id, element_id): return None
    async def get_favorite_steps(self, user_id): return []
    async def get_thread(self, thread_id): return None
    async def get_thread_author(self, thread_id) -> str: return ""
    async def update_step(self, step_dict) -> None: pass
    async def update_thread(self, thread_id, name=None, user_id=None, metadata=None, tags=None) -> None: pass
    async def upsert_feedback(self, feedback) -> None:
        db = SessionLocal()
        try:
            val = 1 if getattr(feedback, "value", 1) == 1 else -1
            fb_record = Feedback(
                for_id=getattr(feedback, "forId", "unknown") or "unknown",
                value=val,
                comment=getattr(feedback, "comment", None)
            )
            db.add(fb_record)
            db.commit()
            logger.info(f"Saved user feedback: value={val}, for_id={fb_record.for_id}")
        except Exception as e:
            logger.error(f"Failed to persist user feedback: {e}")
        finally:
            db.close()
@cl.data_layer
def get_data_layer():
    return DummyDataLayer()

@cl.password_auth_callback
def auth(username: str, password: str) -> Optional[cl.User]:
    """Authenticates the user using our PostgreSQL database."""
    db = SessionLocal()
    try:
        user = get_user(db, username=username)
        if user and verify_password(password, user.hashed_password):
            return cl.User(identifier=username)
    finally:
        db.close()
    return None

@cl.set_chat_profiles
async def chat_profile():
    return [
        cl.ChatProfile(
            name="OpenRouter",
            markdown_description="Fast, multi-model cloud access (Gemini 2.0 Flash / Llama 3.3 / Claude) via OpenRouter.",
        ),
        cl.ChatProfile(
            name="Groq",
            markdown_description="Fast Groq LPU inference (Qwen / Llama).",
        ),
        cl.ChatProfile(
            name="Ollama",
            markdown_description="Private, local GPU/CPU execution via Ollama.",
        ),
    ]

@cl.on_chat_start
async def on_chat_start():
    welcome_msg = """
    ## Welcome
    I am ready to answer your questions regarding NABL compliance and guidelines.
    """
    await cl.Message(content=welcome_msg).send()
    
    # Initialize chat history
    cl.user_session.set("history", [])
    
    # Log session start
    db = SessionLocal()
    try:
        user_identity = cl.user_session.get("user").identifier if cl.user_session.get("user") else "Unknown"
        db_user = db.query(User).filter(User.username == user_identity).first()
        if db_user:
            session_id = cl.user_session.get("id") or str(uuid.uuid4())
            new_session = ChatSession(user_id=db_user.id, session_id=session_id)
            db.add(new_session)
            db.commit()
    except Exception as e:
        logger.error(f"Failed to log chat session: {e}")
    finally:
        db.close()

@cl.on_message
async def main(message: cl.Message):
    # This function handles the incoming user message
    query = message.content
    user_identity = cl.user_session.get("user").identifier if cl.user_session.get("user") else "Unknown"
    
    # Check chat limit
    history = cl.user_session.get("history") or []
    if len(history) >= settings.MAX_CHAT_MESSAGES:
        await cl.Message(
            content=f"⚠️ You have reached the maximum of **{settings.MAX_CHAT_MESSAGES} messages** for this session. Please start a **New Chat** to continue."
        ).send()
        return
    
    # Create a thinking message
    msg = cl.Message(content="")
    await msg.send()
    
    start_time = time.time()
    
    db = SessionLocal()
    try:
        # 0. Handle File Uploads
        uploaded_text = ""
        uploaded_images = []
        uploaded_files_list = []
        
        if message.elements:
            upload_dir = Path("uploaded_files")
            upload_dir.mkdir(exist_ok=True)
            
            for element in message.elements:
                if element.path and element.name.endswith(".pdf"):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    safe_name = f"{user_identity}_{timestamp}_{element.name}"
                    perm_path = upload_dir / safe_name
                    
                    shutil.copy2(element.path, perm_path)
                    
                    try:
                        pages = load_pdf(str(perm_path), extract_images=True)
                        uploaded_text += f"\n\n--- Content from uploaded file: {element.name} ---\n"
                        for p in pages:
                            uploaded_text += p['content'] + " "
                            if "image_base64" in p:
                                uploaded_images.append(p["image_base64"])
                        uploaded_files_list.append(element.name)
                    except PDFTooLargeError as e:
                        msg.content = f"❌ {str(e)}"
                        await msg.update()
                        return
            
            # Persist uploaded document context in session for follow-up questions
            cl.user_session.set("uploaded_text", uploaded_text)
            cl.user_session.set("uploaded_images", uploaded_images)
            cl.user_session.set("uploaded_files_list", uploaded_files_list)
        else:
            # No new file uploaded — retrieve previously stored document context
            uploaded_text = cl.user_session.get("uploaded_text") or ""
            uploaded_images = cl.user_session.get("uploaded_images") or []
            uploaded_files_list = cl.user_session.get("uploaded_files_list") or []

        # 1. Define the search callback for the LLM Provider
        async def run_search(search_query: str) -> List[Dict[str, Any]]:
            tracer = RAGPipelineTracer(search_query)
            hybrid_results = hybrid_search(db, search_query, top_k=10, tracer=tracer)
            reranked_results = rerank_results(search_query, hybrid_results, top_k=settings.RERANK_TOP_N, tracer=tracer)
            
            # Persist trace data so it can be saved to QueryLog later
            cl.user_session.set("last_trace_data", tracer.stages)
            
            # Store sources globally so we can log them later
            cl.user_session.set("last_sources", [res['source'] for res in reranked_results])
            return reranked_results
            
        # Reset last_sources before generation
        cl.user_session.set("last_sources", [])
        
        # Resolve active provider dynamically from user session profile or settings
        selected_profile = cl.user_session.get("chat_profile")
        active_provider = get_llm_provider(selected_profile or settings.LLM_PROVIDER)
        
        # Build effective query containing the uploaded document text for the LLM
        effective_query = query
        if uploaded_text:
            files_label = f" ({', '.join(uploaded_files_list)})" if uploaded_files_list else ""
            effective_query = (
                f"{query}\n\n"
                f"--- Content from uploaded document{files_label} ---\n"
                f"{uploaded_text.strip()}"
            )
        
        # 2. Generate Answer (Streaming)
        async for chunk in active_provider.generate_response_stream(
            query=effective_query, 
            chat_history=history, 
            search_callback=run_search,
            uploaded_images=uploaded_images
        ):
            await msg.stream_token(chunk)
            
        await msg.update()
        
        # Save to history
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": msg.content})
        cl.user_session.set("history", history)
        
        # 3. Log to DB
        latency_ms = (time.time() - start_time) * 1000
        db_user = db.query(User).filter(User.username == user_identity).first()
        user_id = db_user.id if db_user else None
        
        sources_list = cl.user_session.get("last_sources") or []
        trace_data = cl.user_session.get("last_trace_data")
        
        q_log = QueryLog(
            user_id=user_id,
            query=query,
            response=msg.content,
            latency_ms=latency_ms,
            sources_used="; ".join(sources_list) if sources_list else "None",
            model_used=active_provider.model_name,
            trace_data=trace_data
        )
        db.add(q_log)
        db.commit()
        
    except Exception as e:
         logger.error(f"Error handling user message: {str(e)}", exc_info=True)
         err_str = str(e).lower()
         if "429" in err_str or "rate limit" in err_str:
            msg.content = "⚠️ The system is currently experiencing high volume. Please wait a few seconds and try again."
         elif "404" in err_str or "model_not_found" in err_str:
            msg.content = "⚠️ Please wait a few seconds and try again."
         else:
            msg.content = "⚠️ I encountered an unexpected error while processing your request. Please try again in a moment."
         await msg.update()
    finally:
        db.close()
