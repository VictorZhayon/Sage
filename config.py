import os
from dataclasses import dataclass

@dataclass
class Config:
    # Gemini API
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = "gemini-1.5-flash"
    
    # FAISS
    FAISS_PERSIST_DIR: str = "./faiss_db"
    COLLECTION_NAME: str = "documents"
    
    # Chunking
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    
    # Retrieval
    TOP_K_RESULTS: int = 5
    
    # UI
    MAX_FILE_SIZE_MB: int = 10