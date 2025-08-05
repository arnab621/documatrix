"""Knowledge store data models."""

from datetime import datetime
from typing import Dict, List, Optional, Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class Document(BaseModel):
    """Document model."""
    id: UUID = Field(default_factory=uuid4)
    title: str
    content: str
    file_path: str
    file_type: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    processed_at: Optional[datetime] = None
    
    class Config:
        json_encoders = {
            UUID: str,
            datetime: lambda v: v.isoformat()
        }


class Entity(BaseModel):
    """Entity model for graph nodes."""
    id: UUID = Field(default_factory=uuid4)
    name: str
    entity_type: str
    description: Optional[str] = None
    properties: Dict[str, Any] = Field(default_factory=dict)
    document_ids: List[UUID] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            UUID: str,
            datetime: lambda v: v.isoformat()
        }


class Relationship(BaseModel):
    """Relationship model for graph edges."""
    id: UUID = Field(default_factory=uuid4)
    source_entity_id: UUID
    target_entity_id: UUID
    relationship_type: str
    description: Optional[str] = None
    properties: Dict[str, Any] = Field(default_factory=dict)
    weight: float = Field(default=1.0)
    document_ids: List[UUID] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            UUID: str,
            datetime: lambda v: v.isoformat()
        }


class KnowledgeStore(BaseModel):
    """Knowledge store model."""
    id: UUID = Field(default_factory=uuid4)
    name: str
    description: Optional[str] = None
    documents: List[Document] = Field(default_factory=list)
    entities: List[Entity] = Field(default_factory=list)
    relationships: List[Relationship] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            UUID: str,
            datetime: lambda v: v.isoformat()
        }
    
    def add_document(self, document: Document) -> None:
        """Add a document to the knowledge store."""
        self.documents.append(document)
        self.updated_at = datetime.utcnow()
    
    def add_entity(self, entity: Entity) -> None:
        """Add an entity to the knowledge store."""
        self.entities.append(entity)
        self.updated_at = datetime.utcnow()
    
    def add_relationship(self, relationship: Relationship) -> None:
        """Add a relationship to the knowledge store."""
        self.relationships.append(relationship)
        self.updated_at = datetime.utcnow()
    
    def get_document_by_id(self, doc_id: UUID) -> Optional[Document]:
        """Get document by ID."""
        return next((doc for doc in self.documents if doc.id == doc_id), None)
    
    def get_entity_by_id(self, entity_id: UUID) -> Optional[Entity]:
        """Get entity by ID."""
        return next((entity for entity in self.entities if entity.id == entity_id), None)
    
    def get_relationship_by_id(self, rel_id: UUID) -> Optional[Relationship]:
        """Get relationship by ID."""
        return next((rel for rel in self.relationships if rel.id == rel_id), None)
