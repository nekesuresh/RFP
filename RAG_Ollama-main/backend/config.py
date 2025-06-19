import os
from typing import Optional

class Config:
    """Configuration settings for the RAG system"""
    
    # Ollama settings
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:0.6b")
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))
    
    # Vector database settings
    CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_data")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    
    # Chunking settings - now token-based
    CHUNK_SIZE_TOKENS = int(os.getenv("CHUNK_SIZE_TOKENS", "500"))
    OVERLAP_TOKENS = int(os.getenv("OVERLAP_TOKENS", "50"))
    MAX_CHUNK_SIZE_TOKENS = int(os.getenv("MAX_CHUNK_SIZE_TOKENS", "1000"))
    MIN_CHUNK_SIZE_TOKENS = int(os.getenv("MIN_CHUNK_SIZE_TOKENS", "100"))
    
    # Legacy character-based settings (for backward compatibility)
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
    OVERLAP = int(os.getenv("OVERLAP", "50"))
    
    # Retrieval settings
    TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "3"))
    SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))
    
    # API settings
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "8000"))
    
    # File upload settings
    UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploads")
    MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", "52428800"))  # 50MB
    ALLOWED_EXTENSIONS = {".pdf", ".txt", ".docx"}
    
    # Logging settings
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    @classmethod
    def get_ollama_model(cls) -> str:
        """Get the configured Ollama model"""
        return cls.OLLAMA_MODEL
    
    @classmethod
    def get_chunk_size_tokens(cls) -> int:
        """Get the configured chunk size in tokens"""
        return cls.CHUNK_SIZE_TOKENS
    
    @classmethod
    def get_overlap_tokens(cls) -> int:
        """Get the configured overlap size in tokens"""
        return cls.OVERLAP_TOKENS
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate configuration settings"""
        if cls.CHUNK_SIZE_TOKENS <= 0:
            raise ValueError("CHUNK_SIZE_TOKENS must be positive")
        if cls.OVERLAP_TOKENS >= cls.CHUNK_SIZE_TOKENS:
            raise ValueError("OVERLAP_TOKENS must be less than CHUNK_SIZE_TOKENS")
        if cls.TOP_K_RESULTS <= 0:
            raise ValueError("TOP_K_RESULTS must be positive")
        if cls.TEMPERATURE < 0 or cls.TEMPERATURE > 2:
            raise ValueError("TEMPERATURE must be between 0 and 2")
        return True 