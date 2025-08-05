"""Main entry point for DocuMatrix."""

import asyncio
import logging
import multiprocessing
import signal
import sys
from pathlib import Path

import uvicorn

from .core.config import settings
from .ui.app import launch_app


def run_api_server():
    """Run the FastAPI server."""
    try:
        uvicorn.run(
            "contract_analyzer.api.app:app",
            host=settings.host,
            port=settings.port,
            reload=False,  # Disable reload in production
            log_level="info",
            access_log=True
        )
    except Exception as e:
        logging.error(f"Error running API server: {e}")
        sys.exit(1)


def run_gradio_app():
    """Run the Gradio application."""
    try:
        launch_app()
    except Exception as e:
        logging.error(f"Error running Gradio app: {e}")
        sys.exit(1)


def main():
    """Main function to run both API and UI."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO if not settings.debug else logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Starting DocuMatrix")
    
    # Ensure data directories exist
    Path(settings.chroma_persist_directory).mkdir(parents=True, exist_ok=True)
    Path(settings.knowledge_stores_dir).mkdir(parents=True, exist_ok=True)
    Path("./data/uploads").mkdir(parents=True, exist_ok=True)
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "api":
            logger.info("Running API server only")
            run_api_server()
        elif sys.argv[1] == "ui":
            logger.info("Running Gradio UI only")
            run_gradio_app()
        else:
            logger.error(f"Unknown command: {sys.argv[1]}")
            logger.info("Usage: python -m contract_analyzer.main [api|ui]")
            sys.exit(1)
    else:
        # Run both API and UI in separate processes
        logger.info("Starting both API server and Gradio UI")
        
        # Start API server process
        api_process = multiprocessing.Process(target=run_api_server, name="API-Server")
        api_process.start()
        
        # Start Gradio UI process
        ui_process = multiprocessing.Process(target=run_gradio_app, name="Gradio-UI")
        ui_process.start()
        
        def signal_handler(signum, frame):
            logger.info("Received shutdown signal, terminating processes...")
            api_process.terminate()
            ui_process.terminate()
            api_process.join(timeout=5)
            ui_process.join(timeout=5)
            sys.exit(0)
        
        # Register signal handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        try:
            # Wait for processes
            api_process.join()
            ui_process.join()
        except KeyboardInterrupt:
            signal_handler(signal.SIGINT, None)


if __name__ == "__main__":
    main()
