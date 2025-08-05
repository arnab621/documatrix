"""FastAPI application for Contract Analyzer."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .routes import router
from ..core.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO if not settings.debug else logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting Contract Analyzer API")
    
    # Initialize services
    try:
        from ..services.llm_service import llm_service
        logger.info("LLM service initialized")
        
        from ..services.knowledge_store_service import knowledge_store_service
        logger.info("Knowledge store service initialized")
        
        from ..services.chat_service import chat_service
        logger.info("Chat service initialized")
        
    except Exception as e:
        logger.error(f"Error initializing services: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Contract Analyzer API")


# Create FastAPI app
app = FastAPI(
    title="Contract Analyzer API",
    description="GraphRAG-powered contract analysis API",
    version="0.1.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(router, prefix="/api/v1")


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Contract Analyzer API",
        "version": "0.1.0",
        "docs": "/docs"
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "contract_analyzer.api.app:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info"
    )
