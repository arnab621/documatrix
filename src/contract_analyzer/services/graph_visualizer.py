"""Graph visualization service for Contract Analyzer."""

import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from uuid import UUID

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_hex
import numpy as np

from .graph_service import graph_service_manager
from ..models.knowledge_store import KnowledgeStore

logger = logging.getLogger(__name__)


class GraphVisualizer:
    """Service for visualizing knowledge graphs."""
    
    def __init__(self):
        """Initialize the graph visualizer."""
        self.entity_colors = {
            'PERSON': '#FF6B6B',
            'ORGANIZATION': '#4ECDC4', 
            'DATE': '#45B7D1',
            'MONEY': '#96CEB4',
            'LOCATION': '#FFEAA7',
            'CONTRACT_TERM': '#DDA0DD',
            'OBLIGATION': '#F4A460',
            'RIGHT': '#98D8C8',
            'UNKNOWN': '#BDC3C7'
        }
        
        self.relationship_colors = {
            'PARTY_TO': '#E74C3C',
            'OBLIGATED_TO': '#3498DB',
            'PAYS': '#2ECC71',
            'RECEIVES': '#F39C12',
            'LOCATED_IN': '#9B59B6',
            'GOVERNED_BY': '#1ABC9C',
            'RELATED_TO': '#95A5A6'
        }
    
    def create_interactive_graph_data(self, knowledge_store_id: UUID) -> Dict[str, Any]:
        """Create interactive graph data for web visualization."""
        try:
            graph_service = graph_service_manager.get_graph(knowledge_store_id)
            graph_data = graph_service.export_graph_data()
            
            # Convert to format suitable for web visualization (e.g., vis.js, D3.js)
            nodes = []
            edges = []
            
            # Process nodes
            for node in graph_data['nodes']:
                entity_type = node.get('entity_type', 'UNKNOWN')
                nodes.append({
                    'id': node['id'],
                    'label': node.get('name', 'Unknown'),
                    'title': f"Type: {entity_type}\nDescription: {node.get('description', 'N/A')}",
                    'color': self.entity_colors.get(entity_type, self.entity_colors['UNKNOWN']),
                    'size': min(20 + len(node.get('document_ids', [])) * 5, 50),
                    'entity_type': entity_type,
                    'description': node.get('description', ''),
                    'document_count': len(node.get('document_ids', []))
                })
            
            # Process edges
            for edge in graph_data['edges']:
                rel_type = edge.get('relationship_type', 'RELATED_TO')
                edges.append({
                    'from': edge['source'],
                    'to': edge['target'],
                    'label': rel_type,
                    'title': edge.get('description', ''),
                    'color': self.relationship_colors.get(rel_type, self.relationship_colors['RELATED_TO']),
                    'width': max(1, edge.get('weight', 1) * 2),
                    'relationship_type': rel_type,
                    'description': edge.get('description', ''),
                    'weight': edge.get('weight', 1.0)
                })
            
            return {
                'nodes': nodes,
                'edges': edges,
                'statistics': graph_data['statistics'],
                'legend': {
                    'entity_types': self.entity_colors,
                    'relationship_types': self.relationship_colors
                }
            }
            
        except Exception as e:
            logger.error(f"Error creating interactive graph data: {e}")
            return {'nodes': [], 'edges': [], 'statistics': {}, 'legend': {}}
    
    def generate_matplotlib_visualization(
        self, 
        knowledge_store_id: UUID, 
        output_path: str = None,
        figsize: Tuple[int, int] = (15, 10)
    ) -> str:
        """Generate a matplotlib visualization of the graph."""
        try:
            graph_service = graph_service_manager.get_graph(knowledge_store_id)
            G = graph_service.graph
            
            if G.number_of_nodes() == 0:
                return "No entities found in the knowledge graph."
            
            # Create figure
            plt.figure(figsize=figsize)
            plt.clf()
            
            # Use spring layout for better visualization
            pos = nx.spring_layout(G, k=3, iterations=50, seed=42)
            
            # Separate nodes by type for coloring
            node_colors = []
            node_sizes = []
            node_labels = {}
            
            for node_id in G.nodes():
                node_data = G.nodes[node_id]
                entity_type = node_data.get('entity_type', 'UNKNOWN')
                node_colors.append(self.entity_colors.get(entity_type, self.entity_colors['UNKNOWN']))
                
                # Size based on number of connections
                degree = G.degree(node_id)
                node_sizes.append(max(300, degree * 100))
                
                # Label with entity name
                name = node_data.get('name', 'Unknown')
                node_labels[node_id] = name[:15] + '...' if len(name) > 15 else name
            
            # Draw nodes
            nx.draw_networkx_nodes(
                G, pos, 
                node_color=node_colors,
                node_size=node_sizes,
                alpha=0.8,
                linewidths=2,
                edgecolors='black'
            )
            
            # Draw edges with different colors for relationship types
            edge_colors = []
            edge_widths = []
            
            for source, target, edge_data in G.edges(data=True):
                rel_type = edge_data.get('relationship_type', 'RELATED_TO')
                edge_colors.append(self.relationship_colors.get(rel_type, self.relationship_colors['RELATED_TO']))
                edge_widths.append(max(1, edge_data.get('weight', 1) * 2))
            
            nx.draw_networkx_edges(
                G, pos,
                edge_color=edge_colors,
                width=edge_widths,
                alpha=0.6,
                arrows=True,
                arrowsize=20,
                arrowstyle='->'
            )
            
            # Draw labels
            nx.draw_networkx_labels(
                G, pos, 
                labels=node_labels,
                font_size=8,
                font_weight='bold'
            )
            
            # Create legend for entity types
            entity_legend = [
                mpatches.Patch(color=color, label=entity_type)
                for entity_type, color in self.entity_colors.items()
                if any(G.nodes[node].get('entity_type') == entity_type for node in G.nodes())
            ]
            
            # Create legend for relationship types
            relationship_legend = [
                mpatches.Patch(color=color, label=rel_type)
                for rel_type, color in self.relationship_colors.items()
                if any(edge_data.get('relationship_type') == rel_type for _, _, edge_data in G.edges(data=True))
            ]
            
            # Add legends
            if entity_legend:
                plt.legend(
                    handles=entity_legend,
                    title="Entity Types",
                    loc='upper left',
                    bbox_to_anchor=(0, 1)
                )
            
            if relationship_legend:
                plt.legend(
                    handles=relationship_legend,
                    title="Relationship Types", 
                    loc='upper right',
                    bbox_to_anchor=(1, 1)
                )
            
            plt.title(f"Knowledge Graph Visualization\n{G.number_of_nodes()} Entities, {G.number_of_edges()} Relationships", 
                     fontsize=16, fontweight='bold')
            plt.axis('off')
            plt.tight_layout()
            
            # Save or return path
            if output_path is None:
                output_path = f"./data/graph_visualization_{knowledge_store_id}.png"
            
            plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating matplotlib visualization: {e}")
            return f"Error generating visualization: {str(e)}"
    
    def get_graph_summary(self, knowledge_store_id: UUID) -> Dict[str, Any]:
        """Get a summary of the graph structure."""
        try:
            graph_service = graph_service_manager.get_graph(knowledge_store_id)
            stats = graph_service.get_graph_statistics()
            
            # Add more detailed analysis
            graph = graph_service.graph
            
            # Find most connected entities
            if graph.number_of_nodes() > 0:
                degree_centrality = nx.degree_centrality(graph)
                most_connected = sorted(
                    [(node, centrality) for node, centrality in degree_centrality.items()],
                    key=lambda x: x[1],
                    reverse=True
                )[:5]
                
                # Get entity names for most connected
                most_connected_entities = []
                for node_id, centrality in most_connected:
                    node_data = graph.nodes[node_id]
                    most_connected_entities.append({
                        'name': node_data.get('name', 'Unknown'),
                        'type': node_data.get('entity_type', 'Unknown'),
                        'centrality': round(centrality, 3),
                        'connections': graph.degree(node_id)
                    })
                
                stats['most_connected_entities'] = most_connected_entities
                
                # Calculate clustering coefficient
                if graph.number_of_edges() > 0:
                    # Convert to undirected for clustering calculation
                    graph_undirected = graph.to_undirected()
                    clustering = nx.average_clustering(graph_undirected)
                    stats['average_clustering'] = round(clustering, 3)
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting graph summary: {e}")
            return {}
    
    def export_graph_formats(self, knowledge_store_id: UUID, output_dir: str = "./data/exports") -> Dict[str, str]:
        """Export graph in multiple formats."""
        try:
            import os
            from pathlib import Path
            os.makedirs(output_dir, exist_ok=True)
            
            graph_service = graph_service_manager.get_graph(knowledge_store_id)
            graph = graph_service.graph
            
            base_path = Path(output_dir)
            exports = {}
            
            # GraphML format - clean node attributes first
            try:
                graphml_path = base_path / f"{knowledge_store_id}_graph.graphml"
                
                # Create a clean graph for export
                clean_graph = nx.Graph() if not graph.is_directed() else nx.DiGraph()
                
                # Add nodes with string attributes only
                for node, data in graph.nodes(data=True):
                    clean_data = {}
                    for key, value in data.items():
                        if isinstance(value, (str, int, float, bool)):
                            clean_data[key] = str(value)
                    clean_graph.add_node(node, **clean_data)
                
                # Add edges with string attributes only
                for source, target, data in graph.edges(data=True):
                    clean_data = {}
                    for key, value in data.items():
                        if isinstance(value, (str, int, float, bool)):
                            clean_data[key] = str(value)
                    clean_graph.add_edge(source, target, **clean_data)
                
                nx.write_graphml(clean_graph, graphml_path)
                exports["graphml"] = str(graphml_path)
            except Exception as e:
                logger.error(f"Error exporting GraphML: {e}")
            
            # Export as GML
            try:
                gml_path = base_path / f"{knowledge_store_id}_graph.gml"
                nx.write_gml(graph, gml_path)
                exports['gml'] = str(gml_path)
            except Exception as e:
                logger.error(f"Error exporting GML: {e}")
            
            # Export as JSON
            try:
                json_path = base_path / f"{knowledge_store_id}_graph.json"
                graph_data = graph_service.export_graph_data()
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(graph_data, f, indent=2, default=str)
                exports['json'] = str(json_path)
            except Exception as e:
                logger.error(f"Error exporting JSON: {e}")
            
            # Export as DOT (Graphviz format)
            try:
                dot_path = base_path / f"{knowledge_store_id}_graph.dot"
                nx.drawing.nx_pydot.write_dot(graph, dot_path)
                exports['dot'] = str(dot_path)
            except Exception as e:
                logger.error(f"Error exporting DOT: {e}")
            
            return exports
            
        except Exception as e:
            logger.error(f"Error exporting graph formats: {e}")
            return {}


# Global graph visualizer instance
graph_visualizer = GraphVisualizer()
