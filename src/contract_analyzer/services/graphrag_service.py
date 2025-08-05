"""GraphRAG service combining vector search and graph traversal."""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from uuid import UUID

from ..models.knowledge_store import KnowledgeStore
from ..models.chat import ChatMessage, MessageRole
from .vector_store import vector_store_manager
from .graph_service import graph_service_manager
from .llm_service import llm_service

logger = logging.getLogger(__name__)


class GraphRAGService:
    """GraphRAG service for enhanced retrieval and generation."""
    
    def __init__(self, knowledge_store: KnowledgeStore):
        """Initialize GraphRAG service for a knowledge store."""
        self.knowledge_store = knowledge_store
        self.vector_store = vector_store_manager.get_store(knowledge_store.id)
        self.graph_service = graph_service_manager.get_graph(knowledge_store.id)
        
        # Build graph if not already built
        if self.graph_service.graph.number_of_nodes() == 0:
            graph_service_manager.build_graph_from_knowledge_store(knowledge_store)
    
    async def query(
        self, 
        question: str, 
        conversation_history: Optional[List[ChatMessage]] = None,
        max_context_chunks: int = 5,
        max_graph_hops: int = 2
    ) -> Dict[str, Any]:
        """Process a query using GraphRAG approach."""
        try:
            # Step 1: Vector search for relevant content
            vector_results = await self._vector_search(question, max_context_chunks)
            
            # Step 2: Extract entities from question and find related entities in graph
            graph_context = await self._graph_search(question, max_graph_hops)
            
            # Step 3: Combine and rank context
            combined_context = self._combine_context(vector_results, graph_context)
            
            # DEBUG: Print the retrieved context
            print("\n" + "="*80)
            print("🔍 CONTEXT RETRIEVED FOR QUERY")
            print("="*80)
            print(f"📝 Query: {question}")
            print(f"\n📊 Vector Search Results: {len(vector_results)} chunks")
            for i, result in enumerate(vector_results[:3], 1):  # Show first 3
                doc_info = result.get('document_info', {})
                title = doc_info.get('title', 'Unknown')
                content_preview = result['content'][:100] + "..." if len(result['content']) > 100 else result['content']
                print(f"  {i}. From '{title}': {content_preview}")
            
            print(f"\n🕸️ Graph Context:")
            entities = combined_context.get('entities', [])
            relationships = combined_context.get('relationships', [])
            print(f"  - Entities found: {len(entities)}")
            for entity in entities[:5]:  # Show first 5
                entity_data = entity.get('entity_data', {})
                name = entity_data.get('name', 'Unknown')
                entity_type = entity_data.get('entity_type', 'Unknown')
                print(f"    • {name} ({entity_type})")
            
            print(f"  - Relationships found: {len(relationships)}")
            for rel in relationships[:5]:  # Show first 5
                source = rel.get('source_entity', {}).get('name', 'Unknown')
                target = rel.get('target_entity', {}).get('name', 'Unknown')
                rel_type = rel.get('relationship', {}).get('relationship_type', 'related to')
                print(f"    • {source} {rel_type} {target}")
            print("="*80)
            
            # Step 4: Generate response using LLM
            MAX_CONTEXT_TOKENS = 30000  # Increased budget for GPT-4 with 32K response tokens
            response = await self._generate_response(
                question, 
                combined_context, 
                conversation_history
            )
            
            return {
                "response": response,
                "context": {
                    "vector_results": vector_results,
                    "graph_context": graph_context,
                    "combined_context": combined_context
                },
                "metadata": {
                    "knowledge_store_id": str(self.knowledge_store.id),
                    "knowledge_store_name": self.knowledge_store.name
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing GraphRAG query: {e}")
            return {
                "response": "I apologize, but I encountered an error while processing your question. Please try again.",
                "context": {},
                "metadata": {"error": str(e)}
            }
    
    async def _vector_search(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Perform vector search for relevant content."""
        try:
            results = await self.vector_store.search(query, n_results=max_results)
            
            # Enhance results with document information
            enhanced_results = []
            for result in results:
                doc_id = result["metadata"].get("document_id")
                if doc_id:
                    document = self.knowledge_store.get_document_by_id(UUID(doc_id))
                    if document:
                        result["document_info"] = {
                            "title": document.title,
                            "file_type": document.file_type
                        }
                
                enhanced_results.append(result)
            
            logger.info(f"Vector search found {len(enhanced_results)} relevant chunks")
            return enhanced_results
            
        except Exception as e:
            logger.error(f"Error in vector search: {e}")
            return []
    
    async def _graph_search(self, query: str, max_hops: int) -> Dict[str, Any]:
        """Perform graph search for related entities and relationships."""
        try:
            # Extract entities from the query
            query_entities = await self._extract_query_entities(query)
            
            graph_context = {
                "query_entities": query_entities,
                "related_entities": [],
                "relationships": [],
                "entity_paths": []
            }
            
            if not query_entities:
                return graph_context
            
            # Find matching entities in the graph
            matched_entities = []
            for query_entity in query_entities:
                # Search by name
                name_matches = self.graph_service.search_entities_by_name(query_entity["name"])
                matched_entities.extend(name_matches)
                
                # Search by type if specified
                if query_entity.get("type"):
                    type_matches = self.graph_service.search_entities_by_type(query_entity["type"])
                    matched_entities.extend(type_matches)
            
            # Remove duplicates
            unique_entities = {}
            for entity in matched_entities:
                entity_id = entity["entity_id"]
                if entity_id not in unique_entities:
                    unique_entities[entity_id] = entity
            
            matched_entities = list(unique_entities.values())
            
            # Get neighbors and relationships for matched entities
            for entity in matched_entities:
                entity_id = UUID(entity["entity_id"])
                
                # Get neighbors
                neighbors = self.graph_service.get_entity_neighbors(entity_id, max_hops)
                graph_context["related_entities"].extend(neighbors)
                
                # Get relationships
                relationships = self.graph_service.get_entity_relationships(entity_id)
                graph_context["relationships"].extend(relationships)
                
                # Find paths between matched entities
                for other_entity in matched_entities:
                    if entity["entity_id"] != other_entity["entity_id"]:
                        path = self.graph_service.find_shortest_path(
                            entity_id, 
                            UUID(other_entity["entity_id"])
                        )
                        if path:
                            graph_context["entity_paths"].append(path)
            
            logger.info(f"Graph search found {len(matched_entities)} matched entities, {len(graph_context['related_entities'])} related entities")
            return graph_context
            
        except Exception as e:
            logger.error(f"Error in graph search: {e}")
            return {
                "query_entities": [],
                "related_entities": [],
                "relationships": [],
                "entity_paths": []
            }
    
    async def _extract_query_entities(self, query: str) -> List[Dict[str, Any]]:
        """Extract entities from the user query."""
        try:
            system_prompt = """
            Extract key entities from the user's question that might be relevant for searching a contract knowledge graph.
            Focus on entities like company names, person names, contract terms, dates, amounts, locations, etc.
            
            Return a JSON list of entities in this format:
            [
                {
                    "name": "Entity Name",
                    "type": "ENTITY_TYPE"
                }
            ]
            
            Entity types include: PERSON, ORGANIZATION, DATE, MONEY, LOCATION, CONTRACT_TERM, OBLIGATION, etc.
            If you can't determine the type, use "UNKNOWN".
            """
            
            messages = [{"role": "user", "content": f"Extract entities from this question: {query}"}]
            response = await llm_service.generate_response(messages, system_prompt, use_for="graph")
            
            import json
            entities = json.loads(response)
            return entities
            
        except Exception as e:
            logger.error(f"Error extracting query entities: {e}")
            return []
    
    def _combine_context(
        self, 
        vector_results: List[Dict[str, Any]], 
        graph_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Combine vector search results with graph context."""
        try:
            combined = {
                "document_chunks": vector_results,
                "entities": [],
                "relationships": [],
                "entity_connections": []
            }
            
            # Add unique entities from graph context
            seen_entities = set()
            for entity in graph_context.get("related_entities", []):
                entity_id = entity["entity_id"]
                if entity_id not in seen_entities:
                    seen_entities.add(entity_id)
                    combined["entities"].append(entity)
            
            # Add relationships
            combined["relationships"] = graph_context.get("relationships", [])
            
            # Add entity paths as connections
            combined["entity_connections"] = graph_context.get("entity_paths", [])
            
            # Rank and limit context to prevent overwhelming the LLM
            combined = self._rank_and_limit_context(combined)
            
            return combined
            
        except Exception as e:
            logger.error(f"Error combining context: {e}")
            return {
                "document_chunks": vector_results,
                "entities": [],
                "relationships": [],
                "entity_connections": []
            }
    
    def _rank_and_limit_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Rank and limit context to fit within token limits."""
        # Limit document chunks (already ranked by similarity)
        context["document_chunks"] = context["document_chunks"][:5]
        
        # Limit entities (keep most connected ones)
        entities = context["entities"]
        if len(entities) > 10:
            # Sort by number of relationships (if available)
            entities.sort(key=lambda x: len(x.get("entity_data", {}).get("document_ids", [])), reverse=True)
            context["entities"] = entities[:10]
        
        # Limit relationships
        context["relationships"] = context["relationships"][:15]
        
        # Limit entity connections
        context["entity_connections"] = context["entity_connections"][:5]
        
        return context
    
    async def _generate_response(
        self, 
        question: str, 
        context: Dict[str, Any],
        conversation_history: Optional[List[ChatMessage]] = None,
        max_history_tokens: int = 10000
    ) -> str:
        """Generate response using LLM with context."""
        try:
            # Build system prompt with context
            system_prompt = self._build_system_prompt(context)
            
            # Build conversation messages
            messages = []
            
            # Add conversation history if available
            if conversation_history:
                for msg in conversation_history[-6:]:  # Last 6 messages for context
                    messages.append({
                        "role": msg.role.value,
                        "content": msg.content
                    })
            
            # Add current question
            messages.append({
                "role": "user",
                "content": question
            })
            
            # DEBUG: Print the final prompt and messages sent to LLM
            print("\n" + "="*80)
            print("🤖 FINAL PROMPT SENT TO LLM")
            print("="*80)
            print("\n📋 SYSTEM PROMPT:")
            print("-" * 40)
            print(system_prompt)
            print("\n💬 CONVERSATION MESSAGES:")
            print("-" * 40)
            for i, msg in enumerate(messages, 1):
                role = msg["role"].upper()
                content = msg["content"][:200] + "..." if len(msg["content"]) > 200 else msg["content"]
                print(f"{i}. {role}: {content}")
            print("\n" + "="*80)
            print("🔄 SENDING TO LLM...")
            print("="*80)
            
            response = await llm_service.generate_response(messages, system_prompt, use_for="graph")
            
            # DEBUG: Print the LLM response
            print("\n" + "="*80)
            print("✅ LLM RESPONSE RECEIVED")
            print("="*80)
            response_preview = response[:300] + "..." if len(response) > 300 else response
            print(f"📤 Response: {response_preview}")
            print("="*80 + "\n")
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I apologize, but I encountered an error while generating a response. Please try again."
    
    def _build_system_prompt(self, context: Dict[str, Any]) -> str:
        """Build system prompt with context information."""
        prompt_parts = [
            "You are an AI assistant specialized in analyzing contract documents.",
            "You have access to a knowledge base of contract documents and their extracted entities and relationships.",
            "Use the provided context to answer questions accurately and cite specific information when possible.",
            "",
            "CONTEXT INFORMATION:",
            ""
        ]
        
        # Add document chunks
        if context.get("document_chunks"):
            prompt_parts.append("RELEVANT DOCUMENT SECTIONS:")
            for i, chunk in enumerate(context["document_chunks"], 1):
                doc_info = chunk.get("document_info", {})
                title = doc_info.get("title", "Unknown Document")
                content = chunk["content"][:500] + "..." if len(chunk["content"]) > 500 else chunk["content"]
                prompt_parts.append(f"{i}. From '{title}':")
                prompt_parts.append(f"   {content}")
                prompt_parts.append("")
        
        # Add entities
        if context.get("entities"):
            prompt_parts.append("RELEVANT ENTITIES:")
            for entity in context["entities"][:8]:  # Limit to avoid token overflow
                entity_data = entity.get("entity_data", {})
                name = entity_data.get("name", "Unknown")
                entity_type = entity_data.get("entity_type", "Unknown")
                description = entity_data.get("description", "")
                prompt_parts.append(f"- {name} ({entity_type}): {description}")
            prompt_parts.append("")
        
        # Add relationships
        if context.get("relationships"):
            prompt_parts.append("RELEVANT RELATIONSHIPS:")
            for rel in context["relationships"][:10]:  # Limit to avoid token overflow
                source = rel.get("source_entity", {}).get("name", "Unknown")
                target = rel.get("target_entity", {}).get("name", "Unknown")
                rel_type = rel.get("relationship", {}).get("relationship_type", "related to")
                description = rel.get("relationship", {}).get("description", "")
                prompt_parts.append(f"- {source} {rel_type} {target}")
                if description:
                    prompt_parts.append(f"  Description: {description}")
            prompt_parts.append("")
        
        prompt_parts.extend([
            "INSTRUCTIONS:",
            "- Answer questions based on the provided context",
            "- Cite specific documents or sections when possible",
            "- If information is not available in the context, say so clearly",
            "- Be precise and factual in your responses",
            "- Explain relationships between entities when relevant"
        ])
        
        return "\n".join(prompt_parts)


class GraphRAGManager:
    """Manager for GraphRAG services."""
    
    def __init__(self):
        """Initialize GraphRAG manager."""
        self._services: Dict[UUID, GraphRAGService] = {}
    
    def get_service(self, knowledge_store: KnowledgeStore) -> GraphRAGService:
        """Get or create GraphRAG service for a knowledge store."""
        if knowledge_store.id not in self._services:
            self._services[knowledge_store.id] = GraphRAGService(knowledge_store)
        return self._services[knowledge_store.id]
    
    def remove_service(self, knowledge_store_id: UUID) -> None:
        """Remove GraphRAG service."""
        if knowledge_store_id in self._services:
            del self._services[knowledge_store_id]


# Global GraphRAG manager
graphrag_manager = GraphRAGManager()
