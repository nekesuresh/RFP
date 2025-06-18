# Multi-Agent RFP Assistant

A sophisticated RAG (Retrieval-Augmented Generation) system enhanced with multi-agent capabilities for RFP (Request for Proposal) review and improvement. This system uses two specialized agents to provide intelligent document analysis and content enhancement.

## Features

### Multi-Agent System
- Retriever Agent: Performs semantic search on ChromaDB to find relevant documents
- RFP Editor Agent: Analyzes content and provides improvements based on RFP best practices

### Human-in-the-Loop Feedback
- Accept suggestions
- Reject and get rephrased versions
- Edit suggestions manually
- Track feedback history

### RFP Best Practices Integration
- Clarity & Scope requirements
- Measurable outcomes and KPIs
- Stakeholder needs identification
- Vendor responsibilities
- Timeline & Budget considerations
- Technical specifications
- Real-world examples and context

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit UI  │◄──►│   FastAPI       │◄──►│   ChromaDB      │
│   (Frontend)    │    │   (Backend)     │    │   (Vector DB)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │   Ollama        │
                       │   (LLM)         │
                       └─────────────────┘
```

## Quick Start

### Prerequisites
- Python 3.8+
- Ollama installed and running
- A compatible Ollama model (e.g., `llama2-uncensored:7b`, `mistral`, `phi3`)

### Installation

1. **Clone and install dependencies:**
```bash
git clone <repository-url>
cd RAG_Ollama-main
pip install -r requirements.txt
```

2. **Start Ollama and pull a model:**
```bash
ollama pull llama2-uncensored:7b
# or
ollama pull mistral
# or
ollama pull phi3
```

3. **Start the backend server:**
```bash
python start_backend.py
```

4. **Start the frontend (in a new terminal):**
```bash
python start_frontend.py
```

5. **Access the application:**
- Frontend: http://localhost:8501
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

## Project Structure

```
RAG_Ollama-main/
├── backend/
│   ├── main.py          # FastAPI backend with multi-agent endpoints
│   ├── agents.py        # Multi-agent system implementation
│   └── config.py        # Configuration management
├── streamlit_ui/
│   └── app.py           # Streamlit frontend with agent interaction
├── data/                # PDF upload directory
├── chroma_data/         # ChromaDB persistent storage
├── start_backend.py     # Backend startup script
├── start_frontend.py    # Frontend startup script
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Configuration

### Environment Variables
You can configure the system using environment variables:

```bash
# Ollama model configuration
export OLLAMA_MODEL="llama2-uncensored:7b"
export OLLAMA_EMBEDDING_MODEL="llama2-uncensored:7b"

# Vector database configuration
export CHROMA_DB_PATH="./chroma_data"
export COLLECTION_NAME="rag_collection"

# RAG configuration
export CHUNK_SIZE="500"
export TOP_K_RESULTS="3"

# Agent configuration
export TEMPERATURE="0.7"
```

### Supported Ollama Models
- `llama2-uncensored:7b` (default)
- `mistral`
- `phi3`
- `llama3`
- Any other Ollama-compatible model

## Usage

### 1. Upload Documents
- Use the sidebar in the Streamlit UI to upload PDF files
- Documents are automatically processed and indexed in ChromaDB

### 2. Ask Questions
- Enter RFP-related questions or improvement requests
- The system will:
  1. Use the Retriever Agent to find relevant documents
  2. Use the RFP Editor Agent to analyze and improve content
  3. Display both original and improved responses

### 3. Provide Feedback
- Accept: Use the suggestion as-is
- Reject: Provide feedback and get a rephrased version
- Edit: Modify the suggestion manually

### 4. Track Progress
- View agent execution logs
- Review feedback history
- Monitor applied best practices

## API Endpoints

### Core Endpoints
- `POST /upload-pdf/` - Upload and process PDF documents
- `POST /ask/` - Process queries through the multi-agent system
- `POST /feedback/` - Handle user feedback and generate revisions

### Utility Endpoints
- `GET /ping` - Health check
- `GET /config` - View current configuration
- `GET /ask/` - Legacy simple RAG endpoint

## Multi-Agent Workflow

1. **Query Processing**
   ```
   User Query → Retriever Agent → Document Retrieval → RFP Editor Agent → Improved Content
   ```

2. **Feedback Loop**
   ```
   User Feedback → RFP Editor Agent → Revised Content → User Review
   ```

3. **Best Practices Application**
   ```
   Content Analysis → Best Practices Check → Improvement Suggestions → User Validation
   ```

## UI Features

### Main Interface
- Query Input: Large text area for RFP questions
- Results Display: Shows both retrieval and improvement results
- Feedback Interface: Accept/Reject/Edit buttons

### Sidebar
- File Upload: PDF document upload
- API Status: Connection health check
- Configuration: View current settings

### Agent Log
- Step-by-step execution: See what each agent is doing
- Document retrieval: View found documents
- Improvement details: See applied best practices

## Example Queries

- "Help me improve the project scope section"
- "Review this technical requirements section"
- "Make the deliverables more measurable"
- "Add stakeholder requirements to this section"
- "Improve the timeline and budget clarity"

## Development

### Adding New Agents
1. Create a new agent class in `backend/agents.py`
2. Implement the required methods
3. Add the agent to the `MultiAgentRFPAssistant` class

### Customizing Best Practices
1. Modify the `_get_rfp_best_practices()` method in `RFPEditorAgent`
2. Update the `_extract_applied_practices()` method for new practices

### Extending the UI
1. Add new components to `streamlit_ui/app.py`
2. Create new API endpoints in `backend/main.py`
3. Update the frontend-backend communication

## Troubleshooting

### Common Issues

1. Ollama not running
   ```bash
   ollama serve
   ```

2. Model not found
   ```bash
   ollama pull llama2-uncensored:7b
   ```

3. Port conflicts
   - Backend: Change port in `start_backend.py`
   - Frontend: Change port in `start_frontend.py`

4. ChromaDB issues
   - Delete `chroma_data/` directory to reset
   - Check file permissions

### Logs
- Backend logs are displayed in the terminal
- Frontend logs are in the Streamlit interface
- Agent execution logs are shown in the UI

## License

This project is open source and available under the MIT License.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the API documentation at `/docs`
3. Open an issue on GitHub

Built with love using FastAPI, Streamlit, Ollama, and ChromaDB
