"""Graph service using NetworkX."""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Set, Tuple
from uuid import UUID

import networkx as nx
import json

from ..models.knowledge_store import Entity, Relationship, KnowledgeStore
from ..core.config import settings

logger = logging.getLogger(__name__)


class GraphService:
    """Graph service using NetworkX."""
    
    def __init__(self, knowledge_store_id: UUID):
        """Initialize graph service for a specific knowledge store."""
        self.knowledge_store_id = knowledge_store_id
        self.graph = nx.MultiDiGraph()  # Directed graph allowing multiple edges
        self._entity_lookup: Dict[UUID, Entity] = {}
        self._relationship_lookup: Dict[UUID, Relationship] = {}
    
    def add_entity(self, entity: Entity) -> None:
        """Add an entity to the graph."""
        try:
            self._entity_lookup[entity.id] = entity
            
            # Add node to graph
            self.graph.add_node(
                str(entity.id),
                name=entity.name,
                entity_type=entity.entity_type,
                description=entity.description or "",
                properties=entity.properties,
                document_ids=[str(doc_id) for doc_id in entity.document_ids],
                created_at=entity.created_at.isoformat()
            )
            
            logger.debug(f"Added entity {entity.name} to graph")
            
        except Exception as e:
            logger.error(f"Error adding entity {entity.id} to graph: {e}")
            raise
    
    def add_relationship(self, relationship: Relationship) -> None:
        """Add a relationship to the graph."""
        try:
            self._relationship_lookup[relationship.id] = relationship
            
            source_id = str(relationship.source_entity_id)
            target_id = str(relationship.target_entity_id)
            
            # Ensure both entities exist in graph
            if source_id not in self.graph.nodes:
                logger.warning(f"Source entity {source_id} not found in graph")
                return
            
            if target_id not in self.graph.nodes:
                logger.warning(f"Target entity {target_id} not found in graph")
                return
            
            # Add edge to graph
            self.graph.add_edge(
                source_id,
                target_id,
                key=str(relationship.id),
                relationship_type=relationship.relationship_type,
                description=relationship.description or "",
                weight=relationship.weight,
                properties=relationship.properties,
                document_ids=[str(doc_id) for doc_id in relationship.document_ids],
                created_at=relationship.created_at.isoformat()
            )
            
            logger.debug(f"Added relationship {relationship.relationship_type} between {source_id} and {target_id}")
            
        except Exception as e:
            logger.error(f"Error adding relationship {relationship.id} to graph: {e}")
            raise
    
    def get_entity_neighbors(
        self, 
        entity_id: UUID, 
        max_hops: int = 2,
        relationship_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Get neighboring entities within max_hops."""
        try:
            start_node = str(entity_id)
            if start_node not in self.graph.nodes:
                return []
            
            neighbors = []
            visited = set()
            
            # BFS to find neighbors within max_hops
            queue = [(start_node, 0)]  # (node, distance)
            visited.add(start_node)
            
            while queue:
                current_node, distance = queue.pop(0)
                
                if distance >= max_hops:
                    continue
                
                # Get all neighbors (both incoming and outgoing)
                for neighbor in self.graph.neighbors(current_node):
                    if neighbor not in visited:
                        # Check relationship type filter
                        edge_data = self.graph.get_edge_data(current_node, neighbor)
                        if relationship_types:
                            valid_edge = False
                            for edge_key, edge_attrs in edge_data.items():
                                if edge_attrs.get("relationship_type") in relationship_types:
                                    valid_edge = True
                                    break
                            if not valid_edge:
                                continue
                        
                        neighbors.append({
                            "entity_id": neighbor,
                            "entity_data": self.graph.nodes[neighbor],
                            "distance": distance + 1,
                            "path_length": distance + 1
                        })
                        
                        visited.add(neighbor)
                        queue.append((neighbor, distance + 1))
                
                # Also check incoming edges
                for predecessor in self.graph.predecessors(current_node):
                    if predecessor not in visited:
                        # Check relationship type filter
                        edge_data = self.graph.get_edge_data(predecessor, current_node)
                        if relationship_types:
                            valid_edge = False
                            for edge_key, edge_attrs in edge_data.items():
                                if edge_attrs.get("relationship_type") in relationship_types:
                                    valid_edge = True
                                    break
                            if not valid_edge:
                                continue
                        
                        neighbors.append({
                            "entity_id": predecessor,
                            "entity_data": self.graph.nodes[predecessor],
                            "distance": distance + 1,
                            "path_length": distance + 1
                        })
                        
                        visited.add(predecessor)
                        queue.append((predecessor, distance + 1))
            
            return neighbors
            
        except Exception as e:
            logger.error(f"Error getting neighbors for entity {entity_id}: {e}")
            return []
    
    def find_shortest_path(
        self, 
        source_entity_id: UUID, 
        target_entity_id: UUID
    ) -> Optional[List[Dict[str, Any]]]:
        """Find shortest path between two entities."""
        try:
            source_node = str(source_entity_id)
            target_node = str(target_entity_id)
            
            if source_node not in self.graph.nodes or target_node not in self.graph.nodes:
                return None
            
            try:
                path = nx.shortest_path(self.graph, source_node, target_node)
                
                path_info = []
                for i, node in enumerate(path):
                    node_info = {
                        "entity_id": node,
                        "entity_data": self.graph.nodes[node],
                        "step": i
                    }
                    
                    # Add relationship info for edges
                    if i < len(path) - 1:
                        next_node = path[i + 1]
                        edge_data = self.graph.get_edge_data(node, next_node)
                        if edge_data:
                            # Get the first edge (in case of multiple edges)
                            first_edge = list(edge_data.values())[0]
                            node_info["relationship_to_next"] = first_edge
                    
                    path_info.append(node_info)
                
                return path_info
                
            except nx.NetworkXNoPath:
                return None
            
        except Exception as e:
            logger.error(f"Error finding path between {source_entity_id} and {target_entity_id}: {e}")
            return None
    
    def get_entity_relationships(self, entity_id: UUID) -> List[Dict[str, Any]]:
        """Get all relationships for an entity."""
        try:
            node = str(entity_id)
            if node not in self.graph.nodes:
                return []
            
            relationships = []
            
            # Outgoing relationships
            for target in self.graph.neighbors(node):
                edge_data = self.graph.get_edge_data(node, target)
                for edge_key, edge_attrs in edge_data.items():
                    relationships.append({
                        "direction": "outgoing",
                        "source_entity_id": node,
                        "target_entity_id": target,
                        "source_entity": self.graph.nodes[node],
                        "target_entity": self.graph.nodes[target],
                        "relationship": edge_attrs
                    })
            
            # Incoming relationships
            for source in self.graph.predecessors(node):
                edge_data = self.graph.get_edge_data(source, node)
                for edge_key, edge_attrs in edge_data.items():
                    relationships.append({
                        "direction": "incoming",
                        "source_entity_id": source,
                        "target_entity_id": node,
                        "source_entity": self.graph.nodes[source],
                        "target_entity": self.graph.nodes[node],
                        "relationship": edge_attrs
                    })
            
            return relationships
            
        except Exception as e:
            logger.error(f"Error getting relationships for entity {entity_id}: {e}")
            return []
    
    def search_entities_by_type(self, entity_type: str) -> List[Dict[str, Any]]:
        """Search entities by type."""
        try:
            matching_entities = []
            
            for node_id, node_data in self.graph.nodes(data=True):
                if node_data.get("entity_type", "").lower() == entity_type.lower():
                    matching_entities.append({
                        "entity_id": node_id,
                        "entity_data": node_data
                    })
            
            return matching_entities
            
        except Exception as e:
            logger.error(f"Error searching entities by type {entity_type}: {e}")
            return []
    
    def search_entities_by_name(self, name_query: str) -> List[Dict[str, Any]]:
        """Search entities by name (partial match)."""
        try:
            matching_entities = []
            name_query_lower = name_query.lower()
            
            for node_id, node_data in self.graph.nodes(data=True):
                entity_name = node_data.get("name", "").lower()
                if name_query_lower in entity_name:
                    matching_entities.append({
                        "entity_id": node_id,
                        "entity_data": node_data,
                        "relevance_score": len(name_query_lower) / len(entity_name) if entity_name else 0
                    })
            
            # Sort by relevance score (descending)
            matching_entities.sort(key=lambda x: x["relevance_score"], reverse=True)
            
            return matching_entities
            
        except Exception as e:
            logger.error(f"Error searching entities by name {name_query}: {e}")
            return []
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get graph statistics."""
        try:
            stats = {
                "num_entities": self.graph.number_of_nodes(),
                "num_relationships": self.graph.number_of_edges(),
                "entity_types": {},
                "relationship_types": {},
                "connected_components": 0,
                "density": 0.0
            }
            
            # Calculate connected components and density safely
            try:
                if isinstance(self.graph, (nx.MultiGraph, nx.MultiDiGraph)):
                    # For MultiGraph, convert to simple graph for analysis
                    simple_graph = nx.Graph(self.graph) if not self.graph.is_directed() else nx.DiGraph(self.graph)
                    stats["connected_components"] = nx.number_connected_components(simple_graph) if not simple_graph.is_directed() else nx.number_weakly_connected_components(simple_graph)
                    if simple_graph.number_of_nodes() > 1:
                        stats["density"] = nx.density(simple_graph)
                else:
                    stats["connected_components"] = nx.number_weakly_connected_components(self.graph)
                    if self.graph.number_of_nodes() > 1:
                        stats["density"] = nx.density(self.graph)
            except Exception:
                pass  # Keep default values
            
            # Count entity types
            for node_id, node_data in self.graph.nodes(data=True):
                entity_type = node_data.get("entity_type", "UNKNOWN")
                stats["entity_types"][entity_type] = stats["entity_types"].get(entity_type, 0) + 1
            
            # Count relationship types
            for source, target, edge_data in self.graph.edges(data=True):
                rel_type = edge_data.get("relationship_type", "UNKNOWN")
                stats["relationship_types"][rel_type] = stats["relationship_types"].get(rel_type, 0) + 1
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting graph statistics: {e}")
            return {}
    
    def export_graph_data(self) -> Dict[str, Any]:
        """Export graph data for visualization."""
        try:
            nodes = []
            edges = []
            
            # Export nodes
            for node_id, node_data in self.graph.nodes(data=True):
                nodes.append({
                    "id": node_id,
                    **node_data
                })
            
            # Export edges
            for source, target, edge_data in self.graph.edges(data=True):
                edges.append({
                    "source": source,
                    "target": target,
                    **edge_data
                })
            
            return {
                "nodes": nodes,
                "edges": edges,
                "statistics": self.get_graph_statistics()
            }
            
        except Exception as e:
            logger.error(f"Error exporting graph data: {e}")
            return {"nodes": [], "edges": [], "statistics": {}}


class GraphServiceManager:
    """Manager for multiple graph services."""
    
    def __init__(self):
        """Initialize the graph service manager."""
        self._graphs: Dict[UUID, GraphService] = {}
    
    def get_graph(self, knowledge_store_id: UUID) -> GraphService:
        """Get or create a graph service for a knowledge store."""
        if knowledge_store_id not in self._graphs:
            self._graphs[knowledge_store_id] = GraphService(knowledge_store_id)
        return self._graphs[knowledge_store_id]
    
    def remove_graph(self, knowledge_store_id: UUID) -> None:
        """Remove a graph service."""
        if knowledge_store_id in self._graphs:
            del self._graphs[knowledge_store_id]
    
    def build_graph_from_knowledge_store(self, knowledge_store: KnowledgeStore) -> GraphService:
        """Build a graph from a knowledge store."""
        graph_service = self.get_graph(knowledge_store.id)
        
        # Add all entities
        for entity in knowledge_store.entities:
            graph_service.add_entity(entity)
        
        # Add all relationships
        for relationship in knowledge_store.relationships:
            graph_service.add_relationship(relationship)
        
        logger.info(f"Built graph for knowledge store {knowledge_store.id} with {len(knowledge_store.entities)} entities and {len(knowledge_store.relationships)} relationships")
        
        return graph_service


# Global graph service manager
graph_service_manager = GraphServiceManager()
