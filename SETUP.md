# DocuMatrix Setup Guide

## Prerequisites

### System Requirements
- **Python**: 3.11 or higher
- **Package Manager**: UV (recommended) or pip
- **Memory**: 4GB RAM minimum, 8GB recommended
- **Storage**: 2GB free space for dependencies and data

### Azure OpenAI Requirements
- Azure OpenAI resource with API access
- Deployed models:
  - `gpt-4.1` (or your preferred GPT-4 deployment)
  - `gpt-4.1-mini` (optional, for cost optimization)
  - `text-embedding-ada-002` (for embeddings)

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/documatrix.git
cd documatrix
```

### 2. Install UV Package Manager (Recommended)
```bash
# On macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or using pip
pip install uv
```

### 3. Install Dependencies
```bash
# Using UV (recommended)
uv sync

# Or using pip
pip install -r requirements.txt
```

### 4. Environment Configuration

1. **Copy the example environment file:**
   ```bash
   cp .env.example .env
   ```

2. **Edit `.env` with your Azure OpenAI credentials:**
   ```bash
   # Required: Azure OpenAI Configuration
   AZURE_OPENAI_API_KEY=your_actual_api_key_here
   AZURE_OPENAI_ENDPOINT=https://your-resource.cognitiveservices.azure.com
   AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4.1
   
   # Required: Embeddings Configuration
   AZURE_OPENAI_EMBEDDING_API_KEY=your_actual_api_key_here
   AZURE_OPENAI_EMBEDDING_ENDPOINT=https://your-resource.cognitiveservices.azure.com
   AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME=text-embedding-ada-002
   
   # Optional: Model Selection
   AZURE_OPENAI_MODELS=gpt-4.1,gpt-4.1-mini
   DEFAULT_GRAPH_MODEL=gpt-4.1
   DEFAULT_CHAT_MODEL=gpt-4.1
   ```

## Running the Application

### Development Mode
```bash
# Start both API and UI
uv run python -m contract_analyzer.main

# Or start components separately
uv run python -m contract_analyzer.main api    # API only
uv run python -m contract_analyzer.main ui     # UI only
```

### Production Mode (Docker)
```bash
# Build and run with Docker Compose
docker-compose up --build

# Or build manually
docker build -t documatrix .
docker run -p 7860:7860 -p 8000:8000 documatrix
```

## Access Points

- **DocuMatrix UI**: http://localhost:7860
- **API Documentation**: http://localhost:8000/docs
- **API Health Check**: http://localhost:8000/health

## First Steps

1. **Open DocuMatrix**: Navigate to http://localhost:7860
2. **Configure Models**: Go to ⚙️ Settings tab to select your preferred models
3. **Create Knowledge Store**: Go to 📚 Knowledge Stores tab
4. **Upload Documents**: Add your PDF, DOCX, TXT, or MD files
5. **Start Chatting**: Go to 💬 Chat tab and ask questions about your documents

## Troubleshooting

### Common Issues

**1. Azure OpenAI Connection Errors**
- Verify your API key and endpoint in `.env`
- Check that your Azure OpenAI resource is active
- Ensure your deployment names match exactly

**2. Model Not Found Errors**
- Verify deployment names in Azure OpenAI Studio
- Check that models are deployed and active
- Update `AZURE_OPENAI_MODELS` in `.env` if needed

**3. Memory Issues**
- Reduce `chunk_size` in settings for large documents
- Use `gpt-4.1-mini` for memory-intensive operations
- Restart the application if memory usage is high

**4. Port Already in Use**
- Change `PORT` and `GRADIO_PORT` in `.env`
- Kill existing processes: `lsof -ti:7860 | xargs kill -9`

### Performance Optimization

**Model Selection Strategy:**
- **High Accuracy**: Use `gpt-4.1` for both graph and chat
- **Balanced**: Use `gpt-4.1` for graph, `gpt-4.1-mini` for chat
- **Cost Optimized**: Use `gpt-4.1-mini` for both

**Document Processing:**
- Process documents in smaller batches
- Use appropriate chunk sizes (1000-2000 characters)
- Monitor memory usage during large uploads

## Development

### Project Structure
```
src/contract_analyzer/
├── api/           # FastAPI routes and app
├── core/          # Configuration and settings
├── models/        # Pydantic data models
├── services/      # Business logic services
├── ui/            # Gradio interface
└── main.py        # Application entry point
```

### Running Tests
```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_llm_service.py

# Run with coverage
uv run pytest --cov=contract_analyzer
```

### Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review existing GitHub issues
3. Create a new issue with detailed information
4. Include logs and error messages when reporting bugs
