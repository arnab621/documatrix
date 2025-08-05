"""LLM service using Azure OpenAI."""

import asyncio
import logging
from typing import List, Dict, Any, Optional

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.schema import HumanMessage, SystemMessage, AIMessage

from ..core.config import settings

logger = logging.getLogger(__name__)


class LLMService:
    """Azure OpenAI LLM service."""
    
    def __init__(self):
        """Initialize the LLM service."""
        self.chat_models = {}  # Store different chat models
        self.embedding_model = None
        self.current_graph_model = settings.default_graph_model
        self.current_chat_model = settings.default_chat_model
        self._initialize_models()
    
    def _initialize_models(self) -> None:
        """Initialize Azure OpenAI models."""
        try:
            # Initialize all available chat models
            available_models = settings.get_available_models()
            for model_name in available_models:
                self.chat_models[model_name] = AzureChatOpenAI(
                    azure_endpoint=settings.azure_openai_endpoint,
                    azure_deployment=model_name,
                    api_version=settings.azure_openai_api_version,
                    api_key=settings.azure_openai_api_key,
                    temperature=settings.temperature,
                    max_tokens=settings.max_tokens,
                )
            
            # Initialize embedding model
            self.embedding_model = AzureOpenAIEmbeddings(
                azure_endpoint=settings.azure_openai_embedding_endpoint,
                azure_deployment=settings.azure_openai_embedding_deployment_name,
                api_version=settings.azure_openai_embedding_api_version,
                api_key=settings.azure_openai_embedding_api_key,
            )
            
            logger.info(f"Azure OpenAI models initialized successfully: {list(self.chat_models.keys())}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Azure OpenAI models: {e}")
            raise
    
    def get_available_models(self) -> List[str]:
        """Get list of available models."""
        return list(self.chat_models.keys())
    
    def get_current_models(self) -> Dict[str, str]:
        """Get current model selections."""
        return {
            "graph_model": self.current_graph_model,
            "chat_model": self.current_chat_model
        }
    
    def set_graph_model(self, model_name: str) -> bool:
        """Set the model for graph operations."""
        if model_name in self.chat_models:
            self.current_graph_model = model_name
            logger.info(f"Graph model set to: {model_name}")
            return True
        else:
            logger.error(f"Model {model_name} not available")
            return False
    
    def set_chat_model(self, model_name: str) -> bool:
        """Set the model for chat operations."""
        if model_name in self.chat_models:
            self.current_chat_model = model_name
            logger.info(f"Chat model set to: {model_name}")
            return True
        else:
            logger.error(f"Model {model_name} not available")
            return False
    
    async def generate_response(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        model_name: Optional[str] = None,
        use_for: str = "chat"  # "chat" or "graph"
    ) -> str:
        """Generate a response using the specified chat model."""
        try:
            # Determine which model to use
            if model_name:
                if model_name not in self.chat_models:
                    raise ValueError(f"Model {model_name} not available")
                selected_model = self.chat_models[model_name]
            else:
                # Use default model based on use case
                if use_for == "graph":
                    selected_model = self.chat_models[self.current_graph_model]
                else:
                    selected_model = self.chat_models[self.current_chat_model]
            
            # Convert messages to LangChain format
            langchain_messages = []
            
            if system_prompt:
                langchain_messages.append(SystemMessage(content=system_prompt))
            
            for msg in messages:
                if msg["role"] == "user":
                    langchain_messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    langchain_messages.append(AIMessage(content=msg["content"]))
            
            # Generate response
            response = await asyncio.to_thread(
                selected_model.invoke,
                langchain_messages
            )
            
            return response.content
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise
    
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        try:
            embeddings = await asyncio.to_thread(
                self.embedding_model.embed_documents,
                texts
            )
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        try:
            embedding = await asyncio.to_thread(
                self.embedding_model.embed_query,
                text
            )
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise
    
    async def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities from text using LLM."""
        system_prompt = """
        You are an expert at extracting entities from legal and contract documents.
        Extract key entities like parties, dates, amounts, locations, contract terms, etc.
        
        Return a JSON list of entities in this format:
        [
            {
                "name": "Entity Name",
                "type": "ENTITY_TYPE",
                "description": "Brief description of the entity"
            }
        ]
        
        Entity types include: PERSON, ORGANIZATION, DATE, MONEY, LOCATION, CONTRACT_TERM, OBLIGATION, RIGHT, etc.
        """
        
        messages = [{"role": "user", "content": f"Extract entities from this text:\n\n{text}"}]
        
        try:
            response = await self.generate_response(messages, system_prompt, use_for="graph")
            # Parse JSON response
            import json
            entities = json.loads(response)
            return entities
            
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            return []
    
    async def extract_relationships(self, text: str, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract relationships between entities."""
        entity_names = [entity["name"] for entity in entities]
        
        system_prompt = f"""
        You are an expert at identifying relationships between entities in legal and contract documents.
        Given these entities: {entity_names}
        
        Extract relationships between them from the text.
        
        Return a JSON list of relationships in this format:
        [
            {{
                "source": "Source Entity Name",
                "target": "Target Entity Name", 
                "relationship": "RELATIONSHIP_TYPE",
                "description": "Description of the relationship"
            }}
        ]
        
        Relationship types include: PARTY_TO, OBLIGATED_TO, PAYS, RECEIVES, LOCATED_IN, GOVERNED_BY, etc.
        """
        
        messages = [{"role": "user", "content": f"Extract relationships from this text:\n\n{text}"}]
        
        try:
            response = await self.generate_response(messages, system_prompt, use_for="graph")
            # Parse JSON response
            import json
            relationships = json.loads(response)
            return relationships
            
        except Exception as e:
            logger.error(f"Error extracting relationships: {e}")
            return []


# Global LLM service instance
llm_service = LLMService()
