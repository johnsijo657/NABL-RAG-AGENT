import chainlit as cl
from typing import Dict, Optional
import sys
from pathlib import Path

# Ensure project root is in sys.path
project_root = str(Path(__file__).parent.parent.absolute())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.database import SessionLocal
from src.api.auth import verify_password, get_user
from src.retrieval.hybrid_search import hybrid_search
from src.retrieval.reranker import rerank_results
from src.generation.ollama_provider import OllamaProvider
from src.models import QueryLog, ChatSession, User
import uuid
from src.ingestion.loader import load_pdf
import time
import shutil
from datetime import datetime

from chainlit.data import BaseDataLayer
from chainlit.types import Pagination, ThreadFilter, PaginatedResponse

# Initialize our LLM Provider once globally
llm_provider = OllamaProvider()

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
    async def upsert_feedback(self, feedback) -> None: pass
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
        print(f"Failed to log chat session: {e}")
    finally:
        db.close()

@cl.on_message
async def main(message: cl.Message):
    # This function handles the incoming user message
    query = message.content
    user_identity = cl.user_session.get("user").identifier if cl.user_session.get("user") else "Unknown"
    
    # Create a thinking message
    msg = cl.Message(content="")
    await msg.send()
    
    start_time = time.time()
    
    db = SessionLocal()
    try:
        # 0. Handle File Uploads
        uploaded_text = ""
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
                    
                    pages = load_pdf(str(perm_path))
                    uploaded_text += f"\n\n--- Content from uploaded file: {element.name} ---\n"
                    for p in pages:
                        uploaded_text += p['content'] + " "
                    uploaded_files_list.append(element.name)

        # 1. Route Query
        intent = await llm_provider.route_query(query)
        
        # If user uploaded a file, force SEARCH intent to analyze the file
        if uploaded_text:
            intent = "SEARCH"
            
        reranked_results = []
        if intent == "SEARCH":
            # Retrieve and Rank Context
            search_query = query
            if uploaded_text:
                search_query = await llm_provider.expand_search_query(query, uploaded_text)
                print(f"Agentic Query Expansion: '{query}' -> '{search_query}'")
                
            hybrid_results = hybrid_search(db, search_query, top_k=10)
            reranked_results = rerank_results(search_query, hybrid_results, top_k=5)
            
            # Inject uploaded file text into context if it exists
            if uploaded_text:
                reranked_results.insert(0, {
                    "content": uploaded_text,
                    "source": f"User Uploaded File: {', '.join(uploaded_files_list)}",
                    "page_number": 1,
                    "score": 1.0,
                    "type": "upload"
                })

        # 2. Generate Answer (Streaming)
        history = cl.user_session.get("history")
        
        async for chunk in llm_provider.generate_response_stream(query, reranked_results, history):
            await msg.stream_token(chunk)
            
        await msg.update()
        
        # Save to history
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": msg.content})
        cl.user_session.set("history", history)
        
        # 4. Log to DB
        latency_ms = (time.time() - start_time) * 1000
        from src.models import User
        db_user = db.query(User).filter(User.username == user_identity).first()
        user_id = db_user.id if db_user else None
        
        sources_list = [res['source'] for res in reranked_results]
        
        q_log = QueryLog(
            user_id=user_id,
            query=query,
            response=msg.content,
            latency_ms=latency_ms,
            sources_used="; ".join(sources_list) if sources_list else "None"
        )
        db.add(q_log)
        db.commit()
        
    except Exception as e:
        msg.content = f"An error occurred: {str(e)}"
        await msg.update()
    finally:
        db.close()
