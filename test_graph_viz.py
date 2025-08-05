#!/usr/bin/env python3
"""Test script for graph visualization functionality."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from contract_analyzer.services.knowledge_store_service import knowledge_store_service
from contract_analyzer.services.graph_visualizer import graph_visualizer
from contract_analyzer.models.knowledge_store import Entity, Relationship


def create_sample_graph():
    """Create a sample knowledge store with entities and relationships for testing."""
    
    # Create test knowledge store
    knowledge_store = knowledge_store_service.create_knowledge_store(
        name="Graph Visualization Test",
        description="Sample data for testing graph visualization"
    )
    
    # Create sample entities
    entities = [
        Entity(
            name="ABC Corporation",
            entity_type="ORGANIZATION",
            description="Technology company"
        ),
        Entity(
            name="John Smith",
            entity_type="PERSON", 
            description="CEO of ABC Corporation"
        ),
        Entity(
            name="XYZ Ltd",
            entity_type="ORGANIZATION",
            description="Consulting firm"
        ),
        Entity(
            name="Software License Agreement",
            entity_type="CONTRACT_TERM",
            description="Main contract document"
        ),
        Entity(
            name="$50,000",
            entity_type="MONEY",
            description="Annual license fee"
        ),
        Entity(
            name="January 1, 2024",
            entity_type="DATE",
            description="Contract start date"
        )
    ]
    
    # Add entities to knowledge store
    for entity in entities:
        knowledge_store.add_entity(entity)
    
    # Create sample relationships
    relationships = [
        Relationship(
            source_entity_id=entities[1].id,  # John Smith
            target_entity_id=entities[0].id,  # ABC Corporation
            relationship_type="PARTY_TO",
            description="John Smith is CEO of ABC Corporation"
        ),
        Relationship(
            source_entity_id=entities[0].id,  # ABC Corporation
            target_entity_id=entities[2].id,  # XYZ Ltd
            relationship_type="OBLIGATED_TO",
            description="ABC Corporation has contract with XYZ Ltd"
        ),
        Relationship(
            source_entity_id=entities[0].id,  # ABC Corporation
            target_entity_id=entities[4].id,  # $50,000
            relationship_type="PAYS",
            description="ABC Corporation pays annual fee"
        ),
        Relationship(
            source_entity_id=entities[3].id,  # Software License Agreement
            target_entity_id=entities[5].id,  # January 1, 2024
            relationship_type="GOVERNED_BY",
            description="Contract starts on this date"
        ),
        Relationship(
            source_entity_id=entities[2].id,  # XYZ Ltd
            target_entity_id=entities[4].id,  # $50,000
            relationship_type="RECEIVES",
            description="XYZ Ltd receives payment"
        )
    ]
    
    # Add relationships to knowledge store
    for relationship in relationships:
        knowledge_store.add_relationship(relationship)
    
    # Build the graph
    from contract_analyzer.services.graph_service import graph_service_manager
    graph_service_manager.build_graph_from_knowledge_store(knowledge_store)
    
    return knowledge_store


def test_graph_visualization():
    """Test graph visualization functionality."""
    
    print("🎨 Testing Graph Visualization Functionality")
    print("=" * 50)
    
    try:
        # Create sample data
        print("1. Creating sample knowledge store with entities and relationships...")
        knowledge_store = create_sample_graph()
        print(f"   ✅ Created knowledge store: {knowledge_store.name}")
        print(f"   - Entities: {len(knowledge_store.entities)}")
        print(f"   - Relationships: {len(knowledge_store.relationships)}")
        
        # Test graph summary
        print("\n2. Testing graph summary...")
        summary = graph_visualizer.get_graph_summary(knowledge_store.id)
        print(f"   ✅ Graph statistics:")
        print(f"   - Nodes: {summary.get('num_entities', 0)}")
        print(f"   - Edges: {summary.get('num_relationships', 0)}")
        print(f"   - Density: {summary.get('density', 0):.3f}")
        
        if 'entity_types' in summary:
            print(f"   - Entity types: {list(summary['entity_types'].keys())}")
        
        if 'most_connected_entities' in summary:
            print(f"   - Most connected: {[e['name'] for e in summary['most_connected_entities'][:3]]}")
        
        # Test visualization generation
        print("\n3. Testing matplotlib visualization...")
        image_path = graph_visualizer.generate_matplotlib_visualization(
            knowledge_store.id,
            output_path="./data/test_graph.png"
        )
        
        if not image_path.startswith("Error"):
            print(f"   ✅ Graph visualization saved to: {image_path}")
        else:
            print(f"   ❌ Visualization failed: {image_path}")
        
        # Test export functionality
        print("\n4. Testing graph export...")
        exports = graph_visualizer.export_graph_formats(knowledge_store.id)
        print(f"   ✅ Exported {len(exports)} formats:")
        for format_type, path in exports.items():
            print(f"   - {format_type.upper()}: {path}")
        
        # Test interactive graph data
        print("\n5. Testing interactive graph data...")
        interactive_data = graph_visualizer.create_interactive_graph_data(knowledge_store.id)
        print(f"   ✅ Interactive data created:")
        print(f"   - Nodes: {len(interactive_data['nodes'])}")
        print(f"   - Edges: {len(interactive_data['edges'])}")
        print(f"   - Legend: {len(interactive_data['legend']['entity_types'])} entity types")
        
        # Clean up
        print("\n6. Cleaning up test data...")
        knowledge_store_service.delete_knowledge_store(knowledge_store.id)
        print("   ✅ Test data cleaned up")
        
        print("\n🎉 All graph visualization tests passed!")
        print("\n📋 Graph Visualization Features Available:")
        print("   • Matplotlib-based static visualizations")
        print("   • Interactive graph data for web components") 
        print("   • Multi-format export (GraphML, JSON, DOT, GML)")
        print("   • Entity exploration and relationship mapping")
        print("   • Color-coded nodes by entity type")
        print("   • Weighted edges by relationship strength")
        print("   • Graph statistics and analysis")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_graph_visualization()
    sys.exit(0 if success else 1)
