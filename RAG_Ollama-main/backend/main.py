from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
import uuid
import os
import logging
import sys
import ollama

# Add the parent directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_pipeline import add_to_vector_db, query_vector_db
from pdf_load import extract_text_from_pdf, split_pdf_into_chunks_with_metadata
from backend.agents import MultiAgentRFPAssistant
from backend.config import Config

app = FastAPI(title="Multi-Agent RFP Assistant", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Initialize the multi-agent system
multi_agent_assistant = MultiAgentRFPAssistant(query_vector_db)

# Pydantic models for request/response
class QueryRequest(BaseModel):
    query: str

class FeedbackRequest(BaseModel):
    query: str
    feedback: str
    original_suggestion: str

class QueryResponse(BaseModel):
    status: str
    query: str
    retrieval_result: Dict[str, Any]
    improvement_result: Dict[str, Any]
    agent_log: list

class HelpingAgentRequest(BaseModel):
    query: str

class HelpingAgentResponse(BaseModel):
    answer: str

def process_pdf_sync(file_path: str, task_id: str):
    """Process PDF file and add to vector database"""
    logging.info(f"Task {task_id}: Started processing {file_path}")
    try:
        paragraphs = extract_text_from_pdf(file_path)
        logging.info(f"Task {task_id}: Extracted text from PDF")
        max_tokens = Config.get_chunk_size_tokens()
        overlap_tokens = Config.get_overlap_tokens()
        chunks = split_pdf_into_chunks_with_metadata(paragraphs, max_tokens=max_tokens, overlap_tokens=overlap_tokens)
        texts = [chunk['text'] for chunk in chunks]
        ids = [str(uuid.uuid4()) for _ in chunks]
        metadatas = [{'page': chunk['page'], 'para': chunk['para'], 'tokens': chunk['tokens']} for chunk in chunks]
        add_to_vector_db(texts, ids, metadatas)
        logging.info(f"Task {task_id}: Successfully added {len(chunks)} chunks to vector DB")
        logging.info(f"Task {task_id}: Total tokens: {sum(chunk['tokens'] for chunk in chunks)}")
    except Exception as e:
        logging.error(f"Task {task_id}: Error processing PDF: {e}")
        raise

@app.post("/upload-pdf/")
async def upload_pdf(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Upload a PDF file for processing and indexing
    
    This endpoint processes the PDF in the background and adds it to the vector database
    for later retrieval by the multi-agent system.
    """
    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Create upload directory if it doesn't exist
    upload_dir = Config.UPLOAD_DIR
    os.makedirs(upload_dir, exist_ok=True)
    
    file_path = os.path.join(upload_dir, file.filename)
    
    # Save the uploaded file
    with open(file_path, "wb") as f:
        f.write(await file.read())

    task_id = str(uuid.uuid4())
    logging.info(f"Received upload, assigned task_id: {task_id}")

    # Process the PDF in the background
    background_tasks.add_task(process_pdf_sync, file_path, task_id)

    return {
        "message": f"{file.filename} is being processed.",
        "task_id": task_id,
        "status": "processing"
    }

@app.post("/ask/", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    """
    Process a query through the multi-agent RFP review system
    
    This endpoint:
    1. Uses Agent A (Retriever) to find relevant documents
    2. Uses Agent B (RFP Editor) to analyze and improve content
    3. Returns both original and improved responses with agent logs
    """
    try:
        logger.info(f"Processing query: {request.query}")
        
        # Process through multi-agent system
        result = multi_agent_assistant.process_query(request.query)
        
        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result.get("error", "Unknown error"))
        
        return QueryResponse(**result)
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.post("/feedback/", response_model=QueryResponse)
async def handle_feedback(request: FeedbackRequest):
    """
    Handle user feedback and generate revised suggestions
    
    This endpoint allows users to:
    - Reject a suggestion and get a rephrased version
    - Provide specific feedback for improvement
    """
    try:
        logger.info(f"Handling feedback for query: {request.query}")
        
        # Process feedback through multi-agent system
        result = multi_agent_assistant.handle_feedback(
            request.query,
            request.feedback,
            request.original_suggestion
        )
        
        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result.get("error", "Unknown error"))
        
        return QueryResponse(**result)
        
    except Exception as e:
        logger.error(f"Error handling feedback: {e}")
        raise HTTPException(status_code=500, detail=f"Error handling feedback: {str(e)}")

@app.get("/ask/")
async def ask_question_legacy(q: str):
    """
    Legacy endpoint for backward compatibility
    
    This endpoint provides the original simple RAG functionality
    """
    try:
        context_docs = query_vector_db(q)
        context = "\n".join(context_docs) if context_docs else ""

        prompt = f"Answer the question using the context below.\n\nContext:\n{context}\n\nQuestion: {q}"

        response = ollama.chat(
            model=Config.get_ollama_model(), 
            messages=[{"role": "user", "content": prompt}]
        )

        return {"response": response['message']['content']}
        
    except Exception as e:
        logger.error(f"Error in legacy ask endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/ping")
async def ping():
    """Health check endpoint"""
    return {"status": "pong", "service": "Multi-Agent RFP Assistant"}

@app.get("/config")
async def get_config():
    """Get current configuration settings"""
    return {
        "ollama_model": Config.get_ollama_model(),
        "embedding_model": Config.get_embedding_model(),
        "chroma_path": Config.get_chroma_path(),
        "collection_name": Config.get_collection_name(),
        "chunk_size": Config.CHUNK_SIZE,
        "top_k_results": Config.TOP_K_RESULTS,
        "temperature": Config.TEMPERATURE
    }

@app.post("/helping-agent/", response_model=HelpingAgentResponse)
async def helping_agent(request: HelpingAgentRequest):
    """
    RFP knowledge chatbot endpoint with document context.
    Answers any RFP-related question using the Ollama model, leveraging both general RFP knowledge and the indexed PDFs.
    """
    try:
        top_k = Config.TOP_K_RESULTS
        context_chunks = query_vector_db(request.query, n_results=top_k)
        logger.info(f"Helping Agent context_chunks: {context_chunks}")
        # Defensive: ensure context_chunks is a list of dicts with 'text' key and not None
        if not isinstance(context_chunks, list):
            context_chunks = []
        filtered_chunks = []
        if context_chunks:
            for chunk in context_chunks:
                if isinstance(chunk, dict) and 'text' in chunk and chunk['text'] is not None:
                    filtered_chunks.append(chunk)
        context = "\n".join([chunk['text'] for chunk in filtered_chunks]) if filtered_chunks else ""

        prompt = f"""
        You are an expert in writing, reviewing, and consulting on Requests for Proposal (RFPs). Answer the user's question with clear, accurate, and practical advice. Use both your general RFP knowledge and the provided document context below. If the context is relevant, cite it in your answer. If not, answer from your expertise.

        USER QUESTION: {request.query}

        DOCUMENT CONTEXT:
        {context}
        """
        response = ollama.chat(
            model=Config.get_ollama_model(),
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": Config.TEMPERATURE}
        )
        answer = response['message']['content']
        return HelpingAgentResponse(answer=answer)
    except Exception as e:
        logger.error(f"Error in helping agent: {e}")
        raise HTTPException(status_code=500, detail=f"Error in helping agent: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 