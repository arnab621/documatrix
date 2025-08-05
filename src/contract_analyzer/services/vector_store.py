"""Vector store service using ChromaDB."""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from uuid import UUID

import chromadb
from chromadb.config import Settings as ChromaSettings

from ..models.knowledge_store import Document
from ..core.config import settings
from .llm_service import llm_service

logger = logging.getLogger(__name__)


class VectorStore:
    """Vector store using ChromaDB."""
    
    def __init__(self, knowledge_store_id: UUID):
        """Initialize vector store for a specific knowledge store."""
        self.knowledge_store_id = knowledge_store_id
        self.client = None
        self.collection = None
        self._initialize_client()
    
    def _initialize_client(self) -> None:
        """Initialize ChromaDB client."""
        try:
            self.client = chromadb.PersistentClient(
                path=settings.chroma_persist_directory,
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Create or get collection for this knowledge store
            collection_name = f"knowledge_store_{str(self.knowledge_store_id).replace('-', '_')}"
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"knowledge_store_id": str(self.knowledge_store_id)}
            )
            
            logger.info(f"Initialized vector store for knowledge store {self.knowledge_store_id}")
            
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            raise
    
    async def add_document(self, document: Document) -> None:
        """Add a document to the vector store."""
        try:
            # Split document into chunks
            chunks = self._split_document(document)
            
            if not chunks:
                logger.warning(f"No chunks created for document {document.id}")
                return
            
            # Generate embeddings for chunks
            chunk_texts = [chunk["content"] for chunk in chunks]
            embeddings = await llm_service.generate_embeddings(chunk_texts)
            
            # Prepare data for ChromaDB
            ids = [chunk["id"] for chunk in chunks]
            metadatas = [chunk["metadata"] for chunk in chunks]
            
            # Add to collection
            await asyncio.to_thread(
                self.collection.add,
                embeddings=embeddings,
                documents=chunk_texts,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Added document {document.id} with {len(chunks)} chunks to vector store")
            
        except Exception as e:
            logger.error(f"Error adding document {document.id} to vector store: {e}")
            raise
    
    async def search(
        self, 
        query: str, 
        n_results: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar documents."""
        try:
            # Generate query embedding
            query_embedding = await llm_service.generate_embedding(query)
            
            # Search in collection
            results = await asyncio.to_thread(
                self.collection.query,
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=filter_metadata
            )
            
            # Format results
            search_results = []
            if results["documents"] and results["documents"][0]:
                for i, doc in enumerate(results["documents"][0]):
                    result = {
                        "id": results["ids"][0][i],
                        "content": doc,
                        "metadata": results["metadatas"][0][i],
                        "distance": results["distances"][0][i] if results["distances"] else None
                    }
                    search_results.append(result)
            
            logger.info(f"Found {len(search_results)} results for query: {query[:50]}...")
            return search_results
            
        except Exception as e:
            logger.error(f"Error searching vector store: {e}")
            return []
    
    async def get_document_chunks(self, document_id: UUID) -> List[Dict[str, Any]]:
        """Get all chunks for a specific document."""
        try:
            results = await asyncio.to_thread(
                self.collection.get,
                where={"document_id": str(document_id)}
            )
            
            chunks = []
            if results["documents"]:
                for i, doc in enumerate(results["documents"]):
                    chunk = {
                        "id": results["ids"][i],
                        "content": doc,
                        "metadata": results["metadatas"][i]
                    }
                    chunks.append(chunk)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error getting chunks for document {document_id}: {e}")
            return []
    
    async def delete_document(self, document_id: UUID) -> None:
        """Delete all chunks for a document."""
        try:
            await asyncio.to_thread(
                self.collection.delete,
                where={"document_id": str(document_id)}
            )
            
            logger.info(f"Deleted document {document_id} from vector store")
            
        except Exception as e:
            logger.error(f"Error deleting document {document_id} from vector store: {e}")
            raise
    
    def _split_document(self, document: Document) -> List[Dict[str, Any]]:
        """Split document into chunks for vector storage."""
        content = document.content
        chunk_size = settings.chunk_size
        overlap = settings.chunk_overlap
        
        if len(content) <= chunk_size:
            return [{
                "id": f"{document.id}_chunk_0",
                "content": content,
                "metadata": {
                    "document_id": str(document.id),
                    "document_title": document.title,
                    "chunk_index": 0,
                    "file_type": document.file_type
                }
            }]
        
        chunks = []
        start = 0
        chunk_index = 0
        
        while start < len(content):
            end = start + chunk_size
            
            # Try to break at sentence boundary
            if end < len(content):
                for i in range(end, max(start + chunk_size // 2, end - 200), -1):
                    if content[i] in '.!?\n':
                        end = i + 1
                        break
            
            chunk_content = content[start:end].strip()
            if chunk_content:
                chunks.append({
                    "id": f"{document.id}_chunk_{chunk_index}",
                    "content": chunk_content,
                    "metadata": {
                        "document_id": str(document.id),
                        "document_title": document.title,
                        "chunk_index": chunk_index,
                        "file_type": document.file_type,
                        "start_pos": start,
                        "end_pos": end
                    }
                })
                chunk_index += 1
            
            start = end - overlap
            if start >= len(content):
                break
        
        return chunks


class VectorStoreManager:
    """Manager for multiple vector stores."""
    
    def __init__(self):
        """Initialize the vector store manager."""
        self._stores: Dict[UUID, VectorStore] = {}
    
    def get_store(self, knowledge_store_id: UUID) -> VectorStore:
        """Get or create a vector store for a knowledge store."""
        if knowledge_store_id not in self._stores:
            self._stores[knowledge_store_id] = VectorStore(knowledge_store_id)
        return self._stores[knowledge_store_id]
    
    def remove_store(self, knowledge_store_id: UUID) -> None:
        """Remove a vector store."""
        if knowledge_store_id in self._stores:
            del self._stores[knowledge_store_id]


# Global vector store manager
vector_store_manager = VectorStoreManager()
