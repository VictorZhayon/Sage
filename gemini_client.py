import google.generativeai as genai
from typing import List, Dict

class GeminiClient:
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash"):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
    
    def generate_answer(self, query: str, context_docs: List[Dict]) -> str:
        """Generate answer using query and retrieved context"""
        
        # Prepare context
        context = "\n\n".join([
            f"Source: {doc['metadata']['filename']}\n{doc['text']}"
            for doc in context_docs
        ])
        
        # Create prompt
        prompt = f"""You are Sage AI, a helpful AI assistant. Answer the user's question based on the provided context documents.

            Context Documents:
            {context}

            Question: {query}

            Instructions:
            1. Answer the question based on the provided context
            2. If the context doesn't contain enough information, say so
            3. Cite the source documents when relevant
            4. Be concise but thorough
            5. If the user asks for a summary, provide a brief overview of the main points
            6. If the question is not a question, respond appropriately

            Answer:    
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def test_connection(self) -> bool:
        """Test if the API key is valid"""
        try:
            self.model.generate_content("Hello")
            return True
        except:
            return False