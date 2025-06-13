import PyPDF2
import docx
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Tuple
import hashlib

class DocumentProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def extract_text(self, file_path: str, file_type: str) -> str:
        """Extract text from different file types"""
        if file_type == "pdf":
            return self._extract_pdf(file_path)
        elif file_type == "docx":
            return self._extract_docx(file_path)
        elif file_type == "txt":
            return self._extract_txt(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
    
    def _extract_pdf(self, file_path: str) -> str:
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text
    
    def _extract_docx(self, file_path: str) -> str:
        doc = docx.Document(file_path)
        return "\n".join([paragraph.text for paragraph in doc.paragraphs])
    
    def _extract_txt(self, file_path: str) -> str:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    
    def chunk_text(self, text: str, filename: str) -> List[Tuple[str, dict]]:
        """Split text into chunks with metadata"""
        chunks = self.splitter.split_text(text)
        
        # Create metadata for each chunk
        chunk_data = []
        for i, chunk in enumerate(chunks):
            metadata = {
                "filename": filename,
                "chunk_id": i,
                "chunk_hash": hashlib.md5(chunk.encode()).hexdigest()
            }
            chunk_data.append((chunk, metadata))
        
        return chunk_data
    
    def get_file_hash(self, file_content: bytes) -> str:
        """Generate hash for file content to detect duplicates"""
        return hashlib.md5(file_content).hexdigest()