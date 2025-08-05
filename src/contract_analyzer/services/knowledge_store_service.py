"""Knowledge store service for managing document processing and storage."""

import asyncio
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from uuid import UUID, uuid4

from ..models.knowledge_store import KnowledgeStore, Document, Entity, Relationship
from ..core.config import settings
from .document_service import document_service
from .vector_store import vector_store_manager
from .graph_service import graph_service_manager

logger = logging.getLogger(__name__)


class KnowledgeStoreService:
    """Service for managing knowledge stores."""
    
    def __init__(self):
        """Initialize knowledge store service."""
        self._knowledge_stores: Dict[UUID, KnowledgeStore] = {}
        self._load_existing_stores()
    
    def _load_existing_stores(self) -> None:
        """Load existing knowledge stores from disk."""
        try:
            stores_dir = Path(settings.knowledge_stores_dir)
            if not stores_dir.exists():
                return
            
            for store_file in stores_dir.glob("*.json"):
                try:
                    with open(store_file, 'r', encoding='utf-8') as f:
                        store_data = json.load(f)
                    
                    knowledge_store = KnowledgeStore(**store_data)
                    self._knowledge_stores[knowledge_store.id] = knowledge_store
                    
                    logger.info(f"Loaded knowledge store: {knowledge_store.name}")
                    
                except Exception as e:
                    logger.error(f"Error loading knowledge store from {store_file}: {e}")
            
            logger.info(f"Loaded {len(self._knowledge_stores)} knowledge stores")
            
        except Exception as e:
            logger.error(f"Error loading knowledge stores: {e}")
    
    def _save_knowledge_store(self, knowledge_store: KnowledgeStore) -> None:
        """Save knowledge store to disk."""
        try:
            stores_dir = Path(settings.knowledge_stores_dir)
            stores_dir.mkdir(parents=True, exist_ok=True)
            
            store_file = stores_dir / f"{knowledge_store.id}.json"
            
            with open(store_file, 'w', encoding='utf-8') as f:
                json.dump(knowledge_store.dict(), f, indent=2, default=str)
            
            logger.info(f"Saved knowledge store: {knowledge_store.name}")
            
        except Exception as e:
            logger.error(f"Error saving knowledge store {knowledge_store.id}: {e}")
            raise
    
    def create_knowledge_store(self, name: str, description: Optional[str] = None) -> KnowledgeStore:
        """Create a new knowledge store."""
        try:
            knowledge_store = KnowledgeStore(
                name=name,
                description=description or f"Knowledge store for {name}"
            )
            
            self._knowledge_stores[knowledge_store.id] = knowledge_store
            self._save_knowledge_store(knowledge_store)
            
            logger.info(f"Created knowledge store: {name} ({knowledge_store.id})")
            return knowledge_store
            
        except Exception as e:
            logger.error(f"Error creating knowledge store {name}: {e}")
            raise
    
    def get_knowledge_store(self, store_id: UUID) -> Optional[KnowledgeStore]:
        """Get a knowledge store by ID."""
        return self._knowledge_stores.get(store_id)
    
    def list_knowledge_stores(self) -> List[KnowledgeStore]:
        """List all knowledge stores."""
        return list(self._knowledge_stores.values())
    
    def delete_knowledge_store(self, store_id: UUID) -> bool:
        """Delete a knowledge store."""
        try:
            if store_id not in self._knowledge_stores:
                return False
            
            knowledge_store = self._knowledge_stores[store_id]
            
            # Remove from memory
            del self._knowledge_stores[store_id]
            
            # Remove from disk
            store_file = Path(settings.knowledge_stores_dir) / f"{store_id}.json"
            if store_file.exists():
                store_file.unlink()
            
            # Clean up vector store and graph
            vector_store_manager.remove_store(store_id)
            graph_service_manager.remove_graph(store_id)
            
            logger.info(f"Deleted knowledge store: {knowledge_store.name}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting knowledge store {store_id}: {e}")
            return False
    
    async def add_document_to_store(
        self, 
        store_id: UUID, 
        file_path: str, 
        title: Optional[str] = None
    ) -> Optional[Document]:
        """Add a document to a knowledge store and process it."""
        try:
            knowledge_store = self.get_knowledge_store(store_id)
            if not knowledge_store:
                raise ValueError(f"Knowledge store {store_id} not found")
            
            # Process the document
            document = await document_service.process_file(file_path, title)
            
            # Add to knowledge store
            knowledge_store.add_document(document)
            
            # Process document for entities and relationships
            await self._process_document_for_graph(knowledge_store, document)
            
            # Add to vector store
            vector_store = vector_store_manager.get_store(store_id)
            await vector_store.add_document(document)
            
            # Update graph
            graph_service = graph_service_manager.get_graph(store_id)
            for entity in knowledge_store.entities:
                if document.id in entity.document_ids:
                    graph_service.add_entity(entity)
            
            for relationship in knowledge_store.relationships:
                if document.id in relationship.document_ids:
                    graph_service.add_relationship(relationship)
            
            # Save updated knowledge store
            self._save_knowledge_store(knowledge_store)
            
            logger.info(f"Added document {document.title} to knowledge store {knowledge_store.name}")
            return document
            
        except Exception as e:
            logger.error(f"Error adding document to store {store_id}: {e}")
            raise
    
    async def _process_document_for_graph(
        self, 
        knowledge_store: KnowledgeStore, 
        document: Document
    ) -> None:
        """Process document to extract entities and relationships for the graph."""
        try:
            # Extract entities
            entities = await document_service.extract_entities_from_document(document)
            
            # Merge with existing entities
            existing_entity_lookup = {
                (e.name.lower(), e.entity_type): e for e in knowledge_store.entities
            }
            
            new_entities = []
            for entity in entities:
                key = (entity.name.lower(), entity.entity_type)
                if key in existing_entity_lookup:
                    # Merge with existing entity
                    existing_entity = existing_entity_lookup[key]
                    if document.id not in existing_entity.document_ids:
                        existing_entity.document_ids.append(document.id)
                else:
                    # Add new entity
                    new_entities.append(entity)
                    knowledge_store.add_entity(entity)
            
            # Extract relationships
            all_entities = knowledge_store.entities
            relationships = await document_service.extract_relationships_from_document(
                document, 
                all_entities
            )
            
            # Add relationships to knowledge store
            for relationship in relationships:
                knowledge_store.add_relationship(relationship)
            
            logger.info(f"Processed document {document.title}: {len(new_entities)} new entities, {len(relationships)} relationships")
            
        except Exception as e:
            logger.error(f"Error processing document {document.id} for graph: {e}")
            raise
    
    async def process_multiple_files(
        self, 
        store_id: UUID, 
        file_paths: List[str],
        progress_callback: Optional[callable] = None
    ) -> List[Document]:
        """Process multiple files and add them to a knowledge store."""
        try:
            processed_documents = []
            total_files = len(file_paths)
            
            for i, file_path in enumerate(file_paths):
                try:
                    if progress_callback:
                        progress_callback(i, total_files, f"Processing {Path(file_path).name}")
                    
                    document = await self.add_document_to_store(store_id, file_path)
                    if document:
                        processed_documents.append(document)
                    
                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {e}")
                    continue
            
            if progress_callback:
                progress_callback(total_files, total_files, "Processing complete")
            
            logger.info(f"Processed {len(processed_documents)} out of {total_files} files")
            return processed_documents
            
        except Exception as e:
            logger.error(f"Error processing multiple files: {e}")
            return []
    
    def get_store_statistics(self, store_id: UUID) -> Dict[str, Any]:
        """Get statistics for a knowledge store."""
        try:
            knowledge_store = self.get_knowledge_store(store_id)
            if not knowledge_store:
                return {}
            
            # Get graph statistics
            graph_service = graph_service_manager.get_graph(store_id)
            graph_stats = graph_service.get_graph_statistics()
            
            return {
                "knowledge_store": {
                    "id": str(knowledge_store.id),
                    "name": knowledge_store.name,
                    "description": knowledge_store.description,
                    "created_at": knowledge_store.created_at.isoformat(),
                    "updated_at": knowledge_store.updated_at.isoformat()
                },
                "documents": {
                    "count": len(knowledge_store.documents),
                    "file_types": {}
                },
                "entities": {
                    "count": len(knowledge_store.entities),
                    "types": {}
                },
                "relationships": {
                    "count": len(knowledge_store.relationships),
                    "types": {}
                },
                "graph": graph_stats
            }
            
        except Exception as e:
            logger.error(f"Error getting statistics for store {store_id}: {e}")
            return {}


# Global knowledge store service instance
knowledge_store_service = KnowledgeStoreService()
