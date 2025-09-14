import streamlit as st
import os
from pathlib import Path
import tempfile
from typing import List, Dict
import pickle

# Core libraries for PDF processing and RAG
import PyPDF2
from sentence_transformers import SentenceTransformer
import faiss
import openai
from openai import OpenAI

# Alternative: Use Ollama for local LLM (uncomment if preferred)
# import ollama


class PDFProcessor:
    """Handle PDF loading and text extraction"""

    @staticmethod
    def extract_text_from_pdf(pdf_file) -> list:
        """Extract text from uploaded PDF file, return list of (page_num, text)"""
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        page_texts = []
        for i, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            page_texts.append((i + 1, page_text))  # 1-based page number
        return page_texts

    @staticmethod
    def chunk_text_with_page(text: str, page_num: int, chunk_size: int = 1000, overlap: int = 200) -> list:
        """Split text into overlapping chunks, return list of (chunk, page_num)"""
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            if end < len(text):
                for i in range(end, max(start + chunk_size - 200, start), -1):
                    if text[i] in '.!?':
                        end = i + 1
                        break
            chunk = text[start:end].strip()
            if chunk:
                chunks.append((chunk, page_num))
            start = end - overlap
        return chunks


class RAGChatbot:
    """Main RAG chatbot class using FAISS for vector search"""
    def __init__(self):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        self.index = None
        self.embeddings = []
        self.documents = []
        self.metadatas = []
        self.openai_client = None

    def setup_openai(self, api_key: str):
        """Setup OpenAI client"""
        self.openai_client = OpenAI(api_key=api_key)

    def create_collection(self, collection_name: str = "pdf_documents"):
        """Initialize FAISS index (if not already)"""
        if self.index is None:
            # Use dimension of embedding model
            dim = self.embedding_model.get_sentence_embedding_dimension()
            self.index = faiss.IndexFlatL2(dim)
            st.success(f"Initialized FAISS index for: {collection_name}")

    def add_documents(self, documents: List[str], metadatas: List[Dict]):
        """Add documents to the FAISS index"""
        if self.index is None:
            self.create_collection()
        # Compute embeddings
        new_embeddings = self.embedding_model.encode(documents, show_progress_bar=True)
        self.index.add(new_embeddings)
        self.embeddings.extend(new_embeddings)
        self.documents.extend(documents)
        self.metadatas.extend(metadatas)
        st.success(f"Added {len(documents)} document chunks to the FAISS index!")

    def search_documents(self, query: str, n_results: int = 3):
        """Search for relevant documents and return both chunks and metadata"""
        if self.index is None or len(self.documents) == 0:
            return [], []
        query_embedding = self.embedding_model.encode([query])
        D, I = self.index.search(query_embedding, n_results)
        docs = [self.documents[i] for i in I[0] if i < len(self.documents)]
        metadatas = [self.metadatas[i] for i in I[0] if i < len(self.metadatas)]
        return docs, metadatas

    def generate_response(self, query: str, context_docs: list, context_metadatas: list) -> str:
        """Generate response using OpenAI and include references to sources"""
        if not self.openai_client:
            return "Please configure OpenAI API key first."
        # Build context with references
        context_with_refs = []
        for i, doc in enumerate(context_docs):
            meta = context_metadatas[i] if i < len(context_metadatas) else {}
            ref = f"[Source: {meta.get('source', 'unknown')}, Page: {meta.get('page', '?')}, Chunk: {meta.get('chunk_id', '?')}]"
            context_with_refs.append(f"{ref}\n{doc}")
        context = "\n\n".join(context_with_refs)
        prompt = f"""Based on the following context documents, please answer the user's question. For each answer, mention the source and chunk as shown in the context.\nIf the answer is not found in the context, please say so.\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer (include references):"""
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on provided documents. Always cite the relevant information with [Source: ...] as shown in the context."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating response: {str(e)}"

    def chat(self, query: str) -> str:
        """Main chat function"""
        # Search for relevant documents and their metadata
        relevant_docs, relevant_metas = self.search_documents(query, n_results=3)
        if not relevant_docs:
            return "I couldn't find any relevant information in the documents to answer your question."
        # Generate response with references
        response = self.generate_response(query, relevant_docs, relevant_metas)
        return response

def main():
    st.set_page_config(
        page_title="PDF RAG Chatbot",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("📚 PDF RAG Chatbot")
    st.markdown("### Upload your PDFs and chat with their content!")
    
    # Initialize session state
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = RAGChatbot()
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
     # Sidebar for documentation and configuration
    with st.sidebar:
        # --- Documentation Section ---
        with st.expander("ℹ️ How to use this app", expanded=False):
            st.markdown("""
**PDF RAG Chatbot Documentation**

1. **Configure OpenAI API Key**  
   - Enter your OpenAI API key in the sidebar. You can get one from [OpenAI API Keys](https://platform.openai.com/api-keys).

2. **Upload PDF Files**  
   - Click 'Choose PDF files' to upload up to 2 PDF documents.  
   - Click 'Process PDFs' to process and index the documents for chat.

3. **Chat with your PDFs**  
   - Type your question in the chat box and press 'Send'.  
   - The bot will answer using the content of your PDFs and cite the source file, page, and chunk for each answer.

4. **Clear Chat History**  
   - Use the 'Clear Chat History' button to reset the conversation.

**Tips:**
- Ask specific questions for best results.
- The bot only answers based on the uploaded PDFs.
            """)
        # --- End Documentation Section ---
        st.header("Configuration")
        # OpenAI API Key
        api_key = st.text_input(
            "OpenAI API Key", 
            type="password",
            help="Get your API key from https://platform.openai.com/api-keys"
        )
        if api_key:
            st.session_state.chatbot.setup_openai(api_key)
            st.success("✅ OpenAI API configured!")
        st.divider()
        # PDF Upload
        st.header("Upload PDFs")
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type="pdf",
            accept_multiple_files=True,
            help="Upload up to 2 PDF files"
        )
        
        if uploaded_files and len(uploaded_files) <= 2:
            if st.button("Process PDFs"):
                with st.spinner("Processing PDFs..."):
                    processor = PDFProcessor()
                    all_chunks = []
                    all_metadata = []
                    
                    for i, uploaded_file in enumerate(uploaded_files):
                        # Extract text per page
                        page_texts = processor.extract_text_from_pdf(uploaded_file)
                        for page_num, page_text in page_texts:
                            # Chunk per page
                            page_chunks = processor.chunk_text_with_page(page_text, page_num)
                            for j, (chunk, pg) in enumerate(page_chunks):
                                all_chunks.append(chunk)
                                all_metadata.append({
                                    "source": uploaded_file.name,
                                    "chunk_id": j,
                                    "page": pg
                                })
                    
                    # Add to vector database
                    st.session_state.chatbot.create_collection()
                    st.session_state.chatbot.add_documents(all_chunks, all_metadata)
                    
                    st.success(f"Processed {len(uploaded_files)} PDFs with {len(all_chunks)} chunks!")
        
        elif uploaded_files and len(uploaded_files) > 2:
            st.error("Please upload maximum 2 PDF files.")
        
        st.divider()
        
        # Clear chat history
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
    
    # Main chat interface
    st.header("Chat with your PDFs")
    
    # Display chat history
    for i, (question, answer) in enumerate(st.session_state.chat_history):
        with st.container():
            st.write(f"**You:** {question}")
            st.write(f"**Bot:** {answer}")
            st.divider()
    
    # Chat input
    if api_key and st.session_state.chatbot.index is not None:
        user_question = st.text_input(
            "Ask a question about your PDFs:",
            key="user_input",
            placeholder="What is the main topic discussed in the documents?"
        )
        
        if st.button("Send") and user_question:
            with st.spinner("Thinking..."):
                response = st.session_state.chatbot.chat(user_question)
                
                # Add to chat history
                st.session_state.chat_history.append((user_question, response))
                
                # Clear input and rerun to show new message
                st.rerun()
    
    else:
        if not api_key:
            st.info("👈 Please configure your OpenAI API key in the sidebar.")
        elif st.session_state.chatbot.index is None:
            st.info("👈 Please upload and process your PDF files first.")

if __name__ == "__main__":
    main()

# Requirements for requirements.txt:
"""
streamlit>=1.28.0
PyPDF2>=3.0.1
sentence-transformers>=2.2.2
chromadb>=0.4.15
ollama>=0.1.7
"""