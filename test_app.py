#!/usr/bin/env python3
"""Test script for Contract Analyzer application."""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from contract_analyzer.core.config import settings
from contract_analyzer.services.knowledge_store_service import knowledge_store_service
from contract_analyzer.services.chat_service import chat_service


async def test_basic_functionality():
    """Test basic functionality of the Contract Analyzer."""
    
    print("🧪 Testing Contract Analyzer Basic Functionality")
    print("=" * 50)
    
    try:
        # Test 1: Configuration
        print("✅ 1. Configuration loaded successfully")
        print(f"   - Azure OpenAI Endpoint: {settings.azure_openai_endpoint}")
        print(f"   - Gradio Port: {settings.gradio_port}")
        print(f"   - Debug Mode: {settings.debug}")
        
        # Test 2: Knowledge Store Service
        print("\n✅ 2. Knowledge Store Service")
        stores = knowledge_store_service.list_knowledge_stores()
        print(f"   - Found {len(stores)} existing knowledge stores")
        
        # Create a test knowledge store
        test_store = knowledge_store_service.create_knowledge_store(
            name="Test Store",
            description="Test knowledge store for verification"
        )
        print(f"   - Created test store: {test_store.name} ({test_store.id})")
        
        # Test 3: Chat Service
        print("\n✅ 3. Chat Service")
        sessions = chat_service.list_chat_sessions()
        print(f"   - Found {len(sessions)} existing chat sessions")
        
        # Create a test chat session
        test_session = chat_service.create_chat_session(
            knowledge_store_id=test_store.id,
            title="Test Chat Session"
        )
        print(f"   - Created test session: {test_session.title} ({test_session.id})")
        
        # Test 4: Basic chat (without documents)
        print("\n✅ 4. Basic Chat Test")
        response = await chat_service.process_message(
            test_session.id,
            "Hello, can you help me understand contract analysis?"
        )
        print(f"   - Chat response received: {response['assistant_response'][:100]}...")
        
        # Test 5: Clean up
        print("\n✅ 5. Cleanup")
        chat_service.delete_chat_session(test_session.id)
        knowledge_store_service.delete_knowledge_store(test_store.id)
        print("   - Test data cleaned up")
        
        print("\n🎉 All tests passed! Contract Analyzer is working correctly.")
        print("\n📋 Next Steps:")
        print("   1. Open the Gradio UI at http://localhost:7860")
        print("   2. Create a knowledge store")
        print("   3. Upload contract documents (PDF, DOCX, TXT, MD)")
        print("   4. Start chatting with your documents!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function."""
    print("Contract Analyzer - Test Suite")
    print("=" * 50)
    
    # Run async tests
    success = asyncio.run(test_basic_functionality())
    
    if success:
        print("\n✅ All systems operational!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
