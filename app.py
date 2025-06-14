import streamlit as st
import tempfile
import os
from config import Config
from document_processor import DocumentProcessor
from vector_store import VectorStore
from gemini_client import GeminiClient

# Page config
st.set_page_config(
    page_title="Sage AI Assistant",
    page_icon="🧙🏿‍♂️",
    layout="wide"
)

@st.cache_resource
def initialize_components():
    """Initialize all components"""
    config = Config()
    
    # Initialize components
    doc_processor = DocumentProcessor(config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    vector_store = VectorStore(config.FAISS_PERSIST_DIR, config.COLLECTION_NAME)
    
    return config, doc_processor, vector_store

def main():
    st.title("🧙🏿‍♂️ Sage AI Assistant")
    st.markdown("Present your manuscripts and seek wisdom from their pages!")
    
    # Initialize components
    config, doc_processor, vector_store = initialize_components()
    
    # Sidebar for API key
    with st.sidebar:
        st.header("Council of Settings")
        api_key = st.text_input("Gemini API Key", type="password", value=config.GEMINI_API_KEY)
        
        if api_key:
            gemini_client = GeminiClient(api_key, config.GEMINI_MODEL)
            if gemini_client.test_connection():
                st.success("🔮 The Gemini API is attuned.")
            else:
                st.error("❌ The spirits do not recognize this key.")
        else:
            st.warning("⚠️ Kindly provide your Gemini API key to commune with the oracle.")
            st.markdown("[Seek your API key here](https://aistudio.google.com/app/apikey)")
        
        st.divider()
        
        # Database stats
        st.header("Tome of Knowledge")
        stats = vector_store.get_stats()
        st.metric("Manuscripts", stats["unique_files"])
        st.metric("Passages", stats["total_chunks"])
        
        # if stats["files"]:
        #     st.write("Manuscripts in the tome:")
        #     for file in stats["files"]:
        #         st.write(f"• {file}")
        
        # Reset button
        if st.button("🗑️ Purge Tome of Knowledge", type="secondary"):
            vector_store.reset_database()
            st.success("The tome has been cleansed of all manuscripts.")
            st.rerun()
    
    # Main content area - tabs
    tab1, tab2 = st.tabs(["📁 Offer Manuscripts", "💬 Seek Wisdom"])
    
    with tab1:
        st.header("Offer Manuscripts")
        
        uploaded_files = st.file_uploader(
            "Present your manuscripts to the Sage",
            type=['pdf', 'txt', 'docx'],
            accept_multiple_files=True,
            help=f"Maximum manuscript size: {config.MAX_FILE_SIZE_MB}MB"
        )
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                if uploaded_file.size > config.MAX_FILE_SIZE_MB * 1024 * 1024:
                    st.error(f"Manuscript {uploaded_file.name} is too voluminous for the tome!")
                    continue
                
                # Check if file already exists
                if vector_store.file_exists(uploaded_file.name):
                    st.warning(f"Manuscript {uploaded_file.name} already resides in the tome.")
                    if st.button(f"Transcribe Anew {uploaded_file.name}"):
                        vector_store.delete_file(uploaded_file.name)
                        st.info(f"The prior manuscript has been retired.")
                    else:
                        continue
                
                # Process file
                with st.spinner(f"Contemplating {uploaded_file.name}..."):
                    try:
                        # Save to temp file
                        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_path = tmp_file.name
                        
                        # Extract text
                        file_type = uploaded_file.name.split('.')[-1].lower()
                        text = doc_processor.extract_text(tmp_path, file_type)
                        
                        # Chunk text
                        chunks = doc_processor.chunk_text(text, uploaded_file.name)
                        
                        # Add to vector store
                        vector_store.add_documents(chunks)
                        
                        # Clean up
                        os.unlink(tmp_path)
                        
                        st.success(f"✅ {uploaded_file.name} has been inscribed into the tome! ({len(chunks)} passages)")
                        st.rerun()  # Immediately refresh to update sidebar
                        
                    except Exception as e:
                        st.error(f"Alas, an error befell {uploaded_file.name}: {str(e)}")
    
    with tab2:
        st.header("Seek Wisdom")
        
        if not api_key:
            st.warning("The Sage awaits your Gemini API key in the Council of Settings.")
            return
        
        if stats["total_chunks"] == 0:
            st.info("The tome is empty. Present manuscripts before seeking wisdom.")
            return
        
        # top_k = st.slider("How many passages shall the Sage consult?", 1, config.TOP_K_RESULTS)
        
        # Query input
        user_query = st.text_area("What knowledge dost thou seek from the manuscripts?", height=100)
        
        search_button = st.button("🔍 Divine Answer", type="primary")
       
            
        if search_button and user_query:
            with st.spinner("The Sage ponders your query..."):
                try:
                    # Search for relevant documents
                    relevant_docs = vector_store.search(user_query, config.TOP_K_RESULTS)
                    
                    if not relevant_docs:
                        st.warning("The Sage finds no wisdom for your query.")
                        return
                    
                    # Generate answer
                    gemini_client = GeminiClient(api_key, config.GEMINI_MODEL)
                    answer = gemini_client.generate_answer(user_query, relevant_docs)
                    
                    # Display results
                    st.subheader("Sage's Counsel")
                    st.write(answer)
                    
                    # Show sources
                    # st.subheader("Sources Consulted")
                    # for i, doc in enumerate(relevant_docs):
                    #     with st.expander(f"Source {i+1}: {doc['metadata']['filename']} (Similarity: {1-doc['distance']:.3f})"):
                    #         st.write(doc['text'])
                
                except Exception as e:
                    st.error(f"An error has clouded the Sage's vision: {str(e)}")

if __name__ == "__main__":
    main()