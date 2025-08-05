"""Chat-related data models."""

from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class MessageRole(str, Enum):
    """Message role enumeration."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ChatMessage(BaseModel):
    """Chat message model."""
    id: UUID = Field(default_factory=uuid4)
    role: MessageRole
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            UUID: str,
            datetime: lambda v: v.isoformat()
        }


class ChatSession(BaseModel):
    """Chat session model."""
    id: UUID = Field(default_factory=uuid4)
    knowledge_store_id: Optional[UUID] = None
    messages: List[ChatMessage] = Field(default_factory=list)
    title: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            UUID: str,
            datetime: lambda v: v.isoformat()
        }
    
    def add_message(self, message: ChatMessage) -> None:
        """Add a message to the session."""
        self.messages.append(message)
        self.updated_at = datetime.utcnow()
        
        # Auto-generate title from first user message
        if not self.title and message.role == MessageRole.USER and len(self.messages) <= 2:
            self.title = message.content[:50] + "..." if len(message.content) > 50 else message.content
    
    def get_conversation_history(self, limit: Optional[int] = None) -> List[ChatMessage]:
        """Get conversation history with optional limit."""
        messages = self.messages
        if limit:
            messages = messages[-limit:]
        return messages
