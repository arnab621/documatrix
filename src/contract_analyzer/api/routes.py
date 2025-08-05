"""FastAPI routes for Contract Analyzer."""

import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ..services.knowledge_store_service import knowledge_store_service
from ..services.chat_service import chat_service
from ..core.config import settings

logger = logging.getLogger(__name__)

# Create router
router = APIRouter()


# Request/Response models
class CreateKnowledgeStoreRequest(BaseModel):
    name: str
    description: Optional[str] = None


class CreateChatSessionRequest(BaseModel):
    knowledge_store_id: Optional[UUID] = None
    title: Optional[str] = None


class ChatMessageRequest(BaseModel):
    message: str


class SetKnowledgeStoreRequest(BaseModel):
    knowledge_store_id: UUID


# Knowledge Store endpoints
@router.post("/knowledge-stores", response_model=Dict[str, Any])
async def create_knowledge_store(request: CreateKnowledgeStoreRequest):
    """Create a new knowledge store."""
    try:
        knowledge_store = knowledge_store_service.create_knowledge_store(
            name=request.name,
            description=request.description
        )
        
        return {
            "id": str(knowledge_store.id),
            "name": knowledge_store.name,
            "description": knowledge_store.description,
            "created_at": knowledge_store.created_at.isoformat(),
            "status": "created"
        }
        
    except Exception as e:
        logger.error(f"Error creating knowledge store: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/knowledge-stores", response_model=List[Dict[str, Any]])
async def list_knowledge_stores():
    """List all knowledge stores."""
    try:
        stores = knowledge_store_service.list_knowledge_stores()
        
        return [
            {
                "id": str(store.id),
                "name": store.name,
                "description": store.description,
                "document_count": len(store.documents),
                "entity_count": len(store.entities),
                "relationship_count": len(store.relationships),
                "created_at": store.created_at.isoformat(),
                "updated_at": store.updated_at.isoformat()
            }
            for store in stores
        ]
        
    except Exception as e:
        logger.error(f"Error listing knowledge stores: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/knowledge-stores/{store_id}", response_model=Dict[str, Any])
async def get_knowledge_store(store_id: UUID):
    """Get a specific knowledge store."""
    try:
        knowledge_store = knowledge_store_service.get_knowledge_store(store_id)
        if not knowledge_store:
            raise HTTPException(status_code=404, detail="Knowledge store not found")
        
        return {
            "id": str(knowledge_store.id),
            "name": knowledge_store.name,
            "description": knowledge_store.description,
            "documents": [
                {
                    "id": str(doc.id),
                    "title": doc.title,
                    "file_type": doc.file_type,
                    "created_at": doc.created_at.isoformat()
                }
                for doc in knowledge_store.documents
            ],
            "statistics": knowledge_store_service.get_store_statistics(store_id),
            "created_at": knowledge_store.created_at.isoformat(),
            "updated_at": knowledge_store.updated_at.isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting knowledge store {store_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/knowledge-stores/{store_id}")
async def delete_knowledge_store(store_id: UUID):
    """Delete a knowledge store."""
    try:
        success = knowledge_store_service.delete_knowledge_store(store_id)
        if not success:
            raise HTTPException(status_code=404, detail="Knowledge store not found")
        
        return {"status": "deleted", "id": str(store_id)}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting knowledge store {store_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/knowledge-stores/{store_id}/documents")
async def upload_document(
    store_id: UUID,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: Optional[str] = Form(None)
):
    """Upload a document to a knowledge store."""
    try:
        # Verify knowledge store exists
        knowledge_store = knowledge_store_service.get_knowledge_store(store_id)
        if not knowledge_store:
            raise HTTPException(status_code=404, detail="Knowledge store not found")
        
        # Save uploaded file
        upload_dir = Path("./data/uploads")
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = upload_dir / f"{store_id}_{file.filename}"
        
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Process document in background
        background_tasks.add_task(
            _process_document_background,
            store_id,
            str(file_path),
            title or file.filename
        )
        
        return {
            "status": "uploaded",
            "filename": file.filename,
            "file_path": str(file_path),
            "message": "Document uploaded and processing started"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading document to store {store_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _process_document_background(store_id: UUID, file_path: str, title: str):
    """Background task to process uploaded document."""
    try:
        await knowledge_store_service.add_document_to_store(store_id, file_path, title)
        logger.info(f"Successfully processed document {title} for store {store_id}")
        
    except Exception as e:
        logger.error(f"Error processing document {title} for store {store_id}: {e}")


# Chat endpoints
@router.post("/chat/sessions", response_model=Dict[str, Any])
async def create_chat_session(request: CreateChatSessionRequest):
    """Create a new chat session."""
    try:
        chat_session = chat_service.create_chat_session(
            knowledge_store_id=request.knowledge_store_id,
            title=request.title
        )
        
        return {
            "id": str(chat_session.id),
            "knowledge_store_id": str(chat_session.knowledge_store_id) if chat_session.knowledge_store_id else None,
            "title": chat_session.title,
            "created_at": chat_session.created_at.isoformat(),
            "status": "created"
        }
        
    except Exception as e:
        logger.error(f"Error creating chat session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/chat/sessions", response_model=List[Dict[str, Any]])
async def list_chat_sessions(knowledge_store_id: Optional[UUID] = None):
    """List chat sessions."""
    try:
        sessions = chat_service.list_chat_sessions(knowledge_store_id)
        
        return [
            chat_service.get_session_summary(session.id)
            for session in sessions
        ]
        
    except Exception as e:
        logger.error(f"Error listing chat sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/chat/sessions/{session_id}", response_model=Dict[str, Any])
async def get_chat_session(session_id: UUID):
    """Get a specific chat session."""
    try:
        chat_session = chat_service.get_chat_session(session_id)
        if not chat_session:
            raise HTTPException(status_code=404, detail="Chat session not found")
        
        return {
            "id": str(chat_session.id),
            "knowledge_store_id": str(chat_session.knowledge_store_id) if chat_session.knowledge_store_id else None,
            "title": chat_session.title,
            "messages": [
                {
                    "id": str(msg.id),
                    "role": msg.role.value,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat(),
                    "metadata": msg.metadata
                }
                for msg in chat_session.messages
            ],
            "created_at": chat_session.created_at.isoformat(),
            "updated_at": chat_session.updated_at.isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting chat session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/sessions/{session_id}/messages", response_model=Dict[str, Any])
async def send_message(session_id: UUID, request: ChatMessageRequest):
    """Send a message to a chat session."""
    try:
        result = await chat_service.process_message(session_id, request.message)
        return result
        
    except Exception as e:
        logger.error(f"Error sending message to session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/chat/sessions/{session_id}/knowledge-store")
async def set_session_knowledge_store(session_id: UUID, request: SetKnowledgeStoreRequest):
    """Set the knowledge store for a chat session."""
    try:
        success = chat_service.set_session_knowledge_store(
            session_id, 
            request.knowledge_store_id
        )
        
        if not success:
            raise HTTPException(status_code=404, detail="Session or knowledge store not found")
        
        return {"status": "updated", "knowledge_store_id": str(request.knowledge_store_id)}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error setting knowledge store for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/chat/sessions/{session_id}")
async def delete_chat_session(session_id: UUID):
    """Delete a chat session."""
    try:
        success = chat_service.delete_chat_session(session_id)
        if not success:
            raise HTTPException(status_code=404, detail="Chat session not found")
        
        return {"status": "deleted", "id": str(session_id)}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting chat session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Health check endpoint
@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "contract-analyzer",
        "version": "0.1.0"
    }
