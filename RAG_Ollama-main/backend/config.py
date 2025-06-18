import os
from typing import Optional

class Config:
    # Ollama model configuration
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama2-uncensored:7b")
    OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "llama2-uncensored:7b")
    
    # Vector database configuration
    CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_data")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "rag_collection")
    
    # RAG configuration
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
    TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "3"))
    
    # Agent configuration
    MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
    
    # File paths
    DATA_DIR = os.getenv("DATA_DIR", "./data")
    
    @classmethod
    def get_ollama_model(cls) -> str:
        return cls.OLLAMA_MODEL
    
    @classmethod
    def get_embedding_model(cls) -> str:
        return cls.OLLAMA_EMBEDDING_MODEL
    
    @classmethod
    def get_chroma_path(cls) -> str:
        return cls.CHROMA_DB_PATH
    
    @classmethod
    def get_collection_name(cls) -> str:
        return cls.COLLECTION_NAME 