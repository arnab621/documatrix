"""Chat service for handling conversations with GraphRAG."""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from uuid import UUID, uuid4

from ..models.chat import ChatSession, ChatMessage, MessageRole
from ..models.knowledge_store import KnowledgeStore
from ..core.config import settings
from .knowledge_store_service import knowledge_store_service
from .graphrag_service import graphrag_manager

logger = logging.getLogger(__name__)


class ChatService:
    """Service for managing chat sessions and conversations."""
    
    def __init__(self):
        """Initialize chat service."""
        self._chat_sessions: Dict[UUID, ChatSession] = {}
        self._sessions_dir = Path(settings.knowledge_stores_dir) / "chat_sessions"
        self._sessions_dir.mkdir(parents=True, exist_ok=True)
        self._load_existing_sessions()
    
    def _load_existing_sessions(self) -> None:
        """Load existing chat sessions from disk."""
        try:
            for session_file in self._sessions_dir.glob("*.json"):
                try:
                    with open(session_file, 'r', encoding='utf-8') as f:
                        session_data = json.load(f)
                    
                    chat_session = ChatSession(**session_data)
                    self._chat_sessions[chat_session.id] = chat_session
                    
                except Exception as e:
                    logger.error(f"Error loading chat session from {session_file}: {e}")
            
            logger.info(f"Loaded {len(self._chat_sessions)} chat sessions")
            
        except Exception as e:
            logger.error(f"Error loading chat sessions: {e}")
    
    def _save_chat_session(self, chat_session: ChatSession) -> None:
        """Save chat session to disk."""
        try:
            session_file = self._sessions_dir / f"{chat_session.id}.json"
            
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(chat_session.dict(), f, indent=2, default=str)
            
        except Exception as e:
            logger.error(f"Error saving chat session {chat_session.id}: {e}")
    
    def create_chat_session(
        self, 
        knowledge_store_id: Optional[UUID] = None,
        title: Optional[str] = None
    ) -> ChatSession:
        """Create a new chat session."""
        try:
            chat_session = ChatSession(
                knowledge_store_id=knowledge_store_id,
                title=title
            )
            
            self._chat_sessions[chat_session.id] = chat_session
            self._save_chat_session(chat_session)
            
            logger.info(f"Created chat session {chat_session.id}")
            return chat_session
            
        except Exception as e:
            logger.error(f"Error creating chat session: {e}")
            raise
    
    def get_chat_session(self, session_id: UUID) -> Optional[ChatSession]:
        """Get a chat session by ID."""
        return self._chat_sessions.get(session_id)
    
    def list_chat_sessions(
        self, 
        knowledge_store_id: Optional[UUID] = None
    ) -> List[ChatSession]:
        """List chat sessions, optionally filtered by knowledge store."""
        sessions = list(self._chat_sessions.values())
        
        if knowledge_store_id:
            sessions = [s for s in sessions if s.knowledge_store_id == knowledge_store_id]
        
        # Sort by updated_at descending
        sessions.sort(key=lambda x: x.updated_at, reverse=True)
        
        return sessions
    
    def delete_chat_session(self, session_id: UUID) -> bool:
        """Delete a chat session."""
        try:
            if session_id not in self._chat_sessions:
                return False
            
            # Remove from memory
            del self._chat_sessions[session_id]
            
            # Remove from disk
            session_file = self._sessions_dir / f"{session_id}.json"
            if session_file.exists():
                session_file.unlink()
            
            logger.info(f"Deleted chat session {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting chat session {session_id}: {e}")
            return False
    
    async def process_message(
        self, 
        session_id: UUID, 
        user_message: str
    ) -> Dict[str, Any]:
        """Process a user message and generate a response."""
        try:
            chat_session = self.get_chat_session(session_id)
            if not chat_session:
                raise ValueError(f"Chat session {session_id} not found")
            
            # Add user message to session
            user_msg = ChatMessage(
                role=MessageRole.USER,
                content=user_message
            )
            chat_session.add_message(user_msg)
            
            # Generate response
            response_data = await self._generate_response(chat_session, user_message)
            
            # Add assistant response to session
            assistant_msg = ChatMessage(
                role=MessageRole.ASSISTANT,
                content=response_data["response"],
                metadata=response_data.get("metadata", {})
            )
            chat_session.add_message(assistant_msg)
            
            # Save updated session
            self._save_chat_session(chat_session)
            
            return {
                "session_id": str(session_id),
                "user_message": user_message,
                "assistant_response": response_data["response"],
                "context": response_data.get("context", {}),
                "metadata": response_data.get("metadata", {}),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error processing message in session {session_id}: {e}")
            return {
                "session_id": str(session_id),
                "user_message": user_message,
                "assistant_response": "I apologize, but I encountered an error while processing your message. Please try again.",
                "context": {},
                "metadata": {"error": str(e)},
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def _generate_response(
        self, 
        chat_session: ChatSession, 
        user_message: str
    ) -> Dict[str, Any]:
        """Generate response using GraphRAG or general LLM."""
        try:
            # If session has a knowledge store, use GraphRAG
            if chat_session.knowledge_store_id:
                knowledge_store = knowledge_store_service.get_knowledge_store(
                    chat_session.knowledge_store_id
                )
                
                if knowledge_store:
                    return await self._generate_graphrag_response(
                        knowledge_store, 
                        user_message, 
                        chat_session.get_conversation_history(limit=10)
                    )
            
            # Otherwise, use general LLM response
            return await self._generate_general_response(
                user_message, 
                chat_session.get_conversation_history(limit=10)
            )
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                "response": "I apologize, but I encountered an error while generating a response. Please try again.",
                "context": {},
                "metadata": {"error": str(e)}
            }
    
    async def _generate_graphrag_response(
        self, 
        knowledge_store: KnowledgeStore, 
        user_message: str,
        conversation_history: List[ChatMessage]
    ) -> Dict[str, Any]:
        """Generate response using GraphRAG."""
        try:
            graphrag_service = graphrag_manager.get_service(knowledge_store)
            
            result = await graphrag_service.query(
                user_message,
                conversation_history=conversation_history
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating GraphRAG response: {e}")
            raise
    
    async def _generate_general_response(
        self, 
        user_message: str,
        conversation_history: List[ChatMessage]
    ) -> Dict[str, Any]:
        """Generate general response without knowledge store context."""
        try:
            from .llm_service import llm_service
            
            system_prompt = """
            You are a helpful AI assistant specialized in contract analysis and legal documents.
            You can help users understand contract concepts, legal terminology, and provide general guidance.
            
            However, you don't have access to any specific documents in this conversation.
            If users ask about specific contracts or documents, let them know they need to create a knowledge store first.
            """
            
            # Convert conversation history to messages
            messages = []
            for msg in conversation_history[-6:]:  # Last 6 messages
                messages.append({
                    "role": msg.role.value,
                    "content": msg.content
                })
            
            # Add current message
            messages.append({
                "role": "user",
                "content": user_message
            })
            
            response = await llm_service.generate_response(messages, system_prompt, use_for="chat")
            
            return {
                "response": response,
                "context": {},
                "metadata": {
                    "response_type": "general",
                    "knowledge_store_id": None
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating general response: {e}")
            raise
    
    def set_session_knowledge_store(
        self, 
        session_id: UUID, 
        knowledge_store_id: UUID
    ) -> bool:
        """Set the knowledge store for a chat session."""
        try:
            chat_session = self.get_chat_session(session_id)
            if not chat_session:
                return False
            
            # Verify knowledge store exists
            knowledge_store = knowledge_store_service.get_knowledge_store(knowledge_store_id)
            if not knowledge_store:
                return False
            
            chat_session.knowledge_store_id = knowledge_store_id
            chat_session.updated_at = datetime.utcnow()
            
            # Add system message about knowledge store change
            system_msg = ChatMessage(
                role=MessageRole.SYSTEM,
                content=f"Knowledge store changed to: {knowledge_store.name}",
                metadata={"knowledge_store_id": str(knowledge_store_id)}
            )
            chat_session.add_message(system_msg)
            
            self._save_chat_session(chat_session)
            
            logger.info(f"Set knowledge store {knowledge_store_id} for session {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error setting knowledge store for session {session_id}: {e}")
            return False
    
    def get_session_summary(self, session_id: UUID) -> Dict[str, Any]:
        """Get a summary of a chat session."""
        try:
            chat_session = self.get_chat_session(session_id)
            if not chat_session:
                return {}
            
            knowledge_store = None
            if chat_session.knowledge_store_id:
                knowledge_store = knowledge_store_service.get_knowledge_store(
                    chat_session.knowledge_store_id
                )
            
            return {
                "session_id": str(chat_session.id),
                "title": chat_session.title,
                "knowledge_store": {
                    "id": str(knowledge_store.id) if knowledge_store else None,
                    "name": knowledge_store.name if knowledge_store else None
                },
                "message_count": len(chat_session.messages),
                "created_at": chat_session.created_at.isoformat(),
                "updated_at": chat_session.updated_at.isoformat(),
                "last_message": chat_session.messages[-1].content if chat_session.messages else None
            }
            
        except Exception as e:
            logger.error(f"Error getting session summary {session_id}: {e}")
            return {}


# Global chat service instance
chat_service = ChatService()
