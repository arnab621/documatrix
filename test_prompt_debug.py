#!/usr/bin/env python3
"""Test script to demonstrate the debug output showing prompts sent to LLM."""

import sys
import asyncio
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from contract_analyzer.services.knowledge_store_service import knowledge_store_service
from contract_analyzer.services.graphrag_service import graphrag_manager
from contract_analyzer.models.knowledge_store import Entity, Relationship, Document


async def test_prompt_debug():
    """Test the debug output for prompts sent to LLM."""
    
    print("🔍 Testing Prompt Debug Output")
    print("=" * 50)
    
    try:
        # Create a simple knowledge store with sample data
        knowledge_store = knowledge_store_service.create_knowledge_store(
            name="Prompt Debug Test",
            description="Test knowledge store for debugging prompts"
        )
        
        # Add a sample document
        sample_document = Document(
            title="Sample Contract",
            content="""
            This is a sample software license agreement between ABC Corporation and XYZ Solutions.
            
            ABC Corporation, represented by CEO John Smith, agrees to pay XYZ Solutions $50,000 
            annually for software licensing services. The contract is effective from January 1, 2024
            and includes a service level agreement guaranteeing 99.9% uptime.
            
            Payment terms: Monthly payments of $4,167 are due on the first of each month.
            Late payments incur a penalty of $500.
            """,
            file_path="/fake/path/sample_contract.txt",
            file_type=".txt"
        )
        
        # Add document to knowledge store
        knowledge_store.add_document(sample_document)
        
        # Create sample entities
        entities = [
            Entity(
                name="ABC Corporation",
                entity_type="ORGANIZATION",
                description="Software company"
            ),
            Entity(
                name="XYZ Solutions", 
                entity_type="ORGANIZATION",
                description="Software licensing provider"
            ),
            Entity(
                name="John Smith",
                entity_type="PERSON",
                description="CEO of ABC Corporation"
            ),
            Entity(
                name="$50,000",
                entity_type="MONEY", 
                description="Annual licensing fee"
            ),
            Entity(
                name="January 1, 2024",
                entity_type="DATE",
                description="Contract effective date"
            )
        ]
        
        # Add entities
        for entity in entities:
            knowledge_store.add_entity(entity)
        
        # Create sample relationships
        relationships = [
            Relationship(
                source_entity_id=entities[2].id,  # John Smith
                target_entity_id=entities[0].id,  # ABC Corporation
                relationship_type="CEO_OF",
                description="John Smith is CEO of ABC Corporation"
            ),
            Relationship(
                source_entity_id=entities[0].id,  # ABC Corporation
                target_entity_id=entities[1].id,  # XYZ Solutions
                relationship_type="CONTRACTS_WITH",
                description="ABC Corporation has contract with XYZ Solutions"
            ),
            Relationship(
                source_entity_id=entities[0].id,  # ABC Corporation
                target_entity_id=entities[3].id,  # $50,000
                relationship_type="PAYS",
                description="ABC Corporation pays annual fee"
            )
        ]
        
        # Add relationships
        for relationship in relationships:
            knowledge_store.add_relationship(relationship)
        
        # Build the graph
        from contract_analyzer.services.graph_service import graph_service_manager
        graph_service_manager.build_graph_from_knowledge_store(knowledge_store)
        
        # Add some document chunks to vector store for demonstration
        from contract_analyzer.services.vector_store import vector_store_manager
        vector_store = vector_store_manager.get_store(knowledge_store.id)
        
        # Add some sample chunks
        await vector_store.add_documents([
            {
                "content": "ABC Corporation agrees to pay XYZ Solutions $50,000 annually for software licensing services.",
                "metadata": {
                    "document_id": str(sample_document.id),
                    "chunk_index": 0
                }
            },
            {
                "content": "John Smith, CEO of ABC Corporation, is the authorized signatory for this agreement.",
                "metadata": {
                    "document_id": str(sample_document.id), 
                    "chunk_index": 1
                }
            },
            {
                "content": "Payment terms: Monthly payments of $4,167 are due on the first of each month.",
                "metadata": {
                    "document_id": str(sample_document.id),
                    "chunk_index": 2
                }
            }
        ])
        
        # Get GraphRAG service
        graphrag_service = graphrag_manager.get_service(knowledge_store)
        
        # Test with a sample query
        test_query = "Who is responsible for payments in the ABC Corporation contract?"
        
        print(f"\n🤖 Testing GraphRAG Query: '{test_query}'")
        print("=" * 60)
        print("📋 This will show:")
        print("  1. Context retrieved (vector + graph)")
        print("  2. Final system prompt sent to LLM")
        print("  3. Conversation messages")
        print("  4. LLM response")
        print("\n" + "🚀 Starting GraphRAG Query..." + "\n")
        
        # Execute the query (this will trigger our debug output)
        result = await graphrag_service.query(test_query)
        
        print("\n✅ Query completed!")
        print(f"📄 Final response: {result['response'][:200]}...")
        
        # Cleanup
        knowledge_store_service.delete_knowledge_store(knowledge_store.id)
        
    except Exception as e:
        print(f"❌ Error in test: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_prompt_debug())
