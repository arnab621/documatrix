"""Document processing service."""

import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from uuid import UUID

from markitdown import MarkItDown

from ..models.knowledge_store import Document, Entity, Relationship
from ..core.config import settings
from .llm_service import llm_service

logger = logging.getLogger(__name__)


class DocumentService:
    """Service for processing documents."""
    
    def __init__(self):
        """Initialize the document service."""
        self.markitdown = MarkItDown()
    
    async def process_file(self, file_path: str, title: Optional[str] = None) -> Document:
        """Process a file and extract content."""
        try:
            file_path_obj = Path(file_path)
            
            if not file_path_obj.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Use markitdown to extract content
            result = await asyncio.to_thread(
                self.markitdown.convert,
                file_path
            )
            
            content = result.text_content
            
            # Create document
            document = Document(
                title=title or file_path_obj.stem,
                content=content,
                file_path=file_path,
                file_type=file_path_obj.suffix.lower(),
                metadata={
                    "file_size": file_path_obj.stat().st_size,
                    "original_filename": file_path_obj.name
                }
            )
            
            logger.info(f"Processed document: {document.title}")
            return document
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            raise
    
    async def extract_entities_from_document(self, document: Document) -> List[Entity]:
        """Extract entities from a document."""
        try:
            # Split content into chunks for processing
            chunks = self._split_text(document.content)
            all_entities = []
            
            for chunk in chunks:
                entities_data = await llm_service.extract_entities(chunk)
                
                for entity_data in entities_data:
                    entity = Entity(
                        name=entity_data.get("name", ""),
                        entity_type=entity_data.get("type", "UNKNOWN"),
                        description=entity_data.get("description"),
                        document_ids=[document.id],
                        properties={"source_chunk": chunk[:200]}
                    )
                    all_entities.append(entity)
            
            # Deduplicate entities by name and type
            unique_entities = self._deduplicate_entities(all_entities)
            
            logger.info(f"Extracted {len(unique_entities)} entities from document {document.title}")
            return unique_entities
            
        except Exception as e:
            logger.error(f"Error extracting entities from document {document.id}: {e}")
            return []
    
    async def extract_relationships_from_document(
        self, 
        document: Document, 
        entities: List[Entity]
    ) -> List[Relationship]:
        """Extract relationships from a document given its entities."""
        try:
            # Split content into chunks for processing
            chunks = self._split_text(document.content)
            all_relationships = []
            
            # Create entity lookup
            entity_lookup = {entity.name.lower(): entity for entity in entities}
            
            for chunk in chunks:
                # Only process chunks that contain multiple entities
                chunk_entities = [
                    entity for entity in entities 
                    if entity.name.lower() in chunk.lower()
                ]
                
                if len(chunk_entities) < 2:
                    continue
                
                relationships_data = await llm_service.extract_relationships(
                    chunk, 
                    [{"name": e.name, "type": e.entity_type} for e in chunk_entities]
                )
                
                for rel_data in relationships_data:
                    source_name = rel_data.get("source", "").lower()
                    target_name = rel_data.get("target", "").lower()
                    
                    source_entity = entity_lookup.get(source_name)
                    target_entity = entity_lookup.get(target_name)
                    
                    if source_entity and target_entity:
                        relationship = Relationship(
                            source_entity_id=source_entity.id,
                            target_entity_id=target_entity.id,
                            relationship_type=rel_data.get("relationship", "RELATED_TO"),
                            description=rel_data.get("description"),
                            document_ids=[document.id],
                            properties={"source_chunk": chunk[:200]}
                        )
                        all_relationships.append(relationship)
            
            logger.info(f"Extracted {len(all_relationships)} relationships from document {document.title}")
            return all_relationships
            
        except Exception as e:
            logger.error(f"Error extracting relationships from document {document.id}: {e}")
            return []
    
    def _split_text(self, text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
        """Split text into chunks."""
        chunk_size = chunk_size or settings.chunk_size
        overlap = overlap or settings.chunk_overlap
        
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings
                for i in range(end, max(start + chunk_size // 2, end - 200), -1):
                    if text[i] in '.!?':
                        end = i + 1
                        break
            
            chunks.append(text[start:end])
            start = end - overlap
            
            if start >= len(text):
                break
        
        return chunks
    
    def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """Deduplicate entities by name and type."""
        seen = set()
        unique_entities = []
        
        for entity in entities:
            key = (entity.name.lower(), entity.entity_type)
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)
            else:
                # Merge document IDs
                existing = next(e for e in unique_entities if (e.name.lower(), e.entity_type) == key)
                existing.document_ids.extend(entity.document_ids)
                existing.document_ids = list(set(existing.document_ids))
        
        return unique_entities


# Global document service instance
document_service = DocumentService()
