"""Contract Analyzer - GraphRAG-powered contract analysis application."""

__version__ = "0.1.0"

def main() -> None:
    """Main entry point for the application."""
    from .ui.app import launch_app
    launch_app()
