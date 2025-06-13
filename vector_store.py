import faiss
import numpy as np
import pickle
import os
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Dict
import uuid

class VectorStore:
    def __init__(self, persist_dir: str, collection_name: str):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embedding_dim = 384  # all-MiniLM-L6-v2 embedding dimension
        
        # Create persist directory
        os.makedirs(persist_dir, exist_ok=True)
        
        # File paths
        self.index_path = os.path.join(persist_dir, f"{collection_name}.index")
        self.metadata_path = os.path.join(persist_dir, f"{collection_name}_metadata.pkl")
        self.documents_path = os.path.join(persist_dir, f"{collection_name}_documents.pkl")
        
        # Initialize or load index
        self._load_or_create_index()
    
    def _load_or_create_index(self):
        """Load existing index or create new one"""
        if os.path.exists(self.index_path):
            # Load existing index
            self.index = faiss.read_index(self.index_path)
            with open(self.metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
            with open(self.documents_path, 'rb') as f:
                self.documents = pickle.load(f)
        else:
            # Create new index
            self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for cosine similarity
            self.metadata = []
            self.documents = []
    
    def _save_index(self):
        """Save index and metadata to disk"""
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)
        with open(self.documents_path, 'wb') as f:
            pickle.dump(self.documents, f)
    
    def add_documents(self, chunks: List[Tuple[str, dict]]) -> None:
        """Add document chunks to vector store"""
        texts = [chunk[0] for chunk in chunks]
        metadatas = [chunk[1] for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(texts)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add to index
        self.index.add(embeddings.astype('float32'))
        
        # Store metadata and documents
        for i, (text, metadata) in enumerate(chunks):
            metadata['id'] = str(uuid.uuid4())
            self.metadata.append(metadata)
            self.documents.append(text)
        
        # Save to disk
        self._save_index()
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for similar documents"""
        if self.index.ntotal == 0:
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding.astype('float32'), min(top_k, self.index.ntotal))
        
        # Format results
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx != -1:  # -1 indicates no result
                results.append({
                    "text": self.documents[idx],
                    "metadata": self.metadata[idx],
                    "distance": 1 - score  # Convert similarity to distance
                })
        
        return results
    
    def file_exists(self, filename: str) -> bool:
        """Check if a file already exists in the collection"""
        return any(meta.get('filename') == filename for meta in self.metadata)
    
    def delete_file(self, filename: str) -> None:
        """Delete all chunks from a specific file"""
        # Find indices to delete
        indices_to_delete = [i for i, meta in enumerate(self.metadata) if meta.get('filename') == filename]
        
        if not indices_to_delete:
            return
        
        # Create new index without deleted items
        remaining_indices = [i for i in range(len(self.metadata)) if i not in indices_to_delete]
        
        if remaining_indices:
            # Get embeddings for remaining documents
            remaining_texts = [self.documents[i] for i in remaining_indices]
            embeddings = self.embedding_model.encode(remaining_texts)
            faiss.normalize_L2(embeddings)
            
            # Create new index
            new_index = faiss.IndexFlatIP(self.embedding_dim)
            new_index.add(embeddings.astype('float32'))
            
            # Update data structures
            self.index = new_index
            self.metadata = [self.metadata[i] for i in remaining_indices]
            self.documents = [self.documents[i] for i in remaining_indices]
        else:
            # No documents left
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            self.metadata = []
            self.documents = []
        
        # Save changes
        self._save_index()
    
    def reset_database(self) -> None:
        """Clear all data from the collection"""
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.metadata = []
        self.documents = []
        
        # Remove files
        for path in [self.index_path, self.metadata_path, self.documents_path]:
            if os.path.exists(path):
                os.remove(path)
    
    def get_stats(self) -> Dict:
        """Get database statistics"""
        unique_files = set(meta.get('filename', '') for meta in self.metadata)
        unique_files.discard('')  # Remove empty filenames
        
        return {
            "total_chunks": len(self.documents),
            "unique_files": len(unique_files),
            "files": list(unique_files)
        }