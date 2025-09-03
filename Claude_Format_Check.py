# RAG Document Q&A Application
import streamlit as st
import os
import tempfile
from pathlib import Path
import hashlib
import pickle
from datetime import datetime
from dotenv import load_dotenv

# Document processing
import pandas as pd
from docx import Document
import PyPDF2
import json

# Vector embeddings and search
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    st.error("❌ sentence-transformers not installed. Run: pip install sentence-transformers")

# OpenAI for LLM responses
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re

# =============================
# Page Configuration
# =============================
st.set_page_config(
    page_title="RAG Document Q&A", 
    layout="wide", 
    initial_sidebar_state="expanded",
    page_icon="🤖"
)

# =============================
# Styling
# =============================
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .upload-area {
        border: 2px dashed #667eea;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background: rgba(102, 126, 234, 0.05);
        margin: 1rem 0;
    }
    
    .document-card {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .answer-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .source-box {
        background: #f8f9ff;
        border: 1px solid #e1e5ff;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
    }
    
    .stats-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# =============================
# Header
# =============================
st.markdown("""
<div class="main-header">
    <h1>🤖 RAG Document Q&A Assistant</h1>
    <p>Upload your documents and ask questions about their content</p>
</div>
""", unsafe_allow_html=True)

# =============================
# Configuration
# =============================
load_dotenv()

# Initialize models
@st.cache_resource
def load_embedding_model():
    if not EMBEDDINGS_AVAILABLE:
        return None
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        return model
    except Exception as e:
        st.error(f"Error loading embedding model: {e}")
        return None

embedding_model = load_embedding_model()

# OpenAI setup
openai_key = os.getenv("OPENAI_API_KEY", "")
client = OpenAI(api_key=openai_key) if openai_key and OPENAI_AVAILABLE else None

# =============================
# Document Processing Functions
# =============================
def extract_text_from_pdf(file):
    """Extract text from PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""

def extract_text_from_docx(file):
    """Extract text from DOCX file"""
    try:
        doc = Document(file)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading DOCX: {e}")
        return ""

def extract_text_from_csv(file):
    """Extract text from CSV file"""
    try:
        df = pd.read_csv(file)
        # Convert DataFrame to readable text
        text = f"CSV Data Summary:\n"
        text += f"Shape: {df.shape[0]} rows, {df.shape[1]} columns\n"
        text += f"Columns: {', '.join(df.columns)}\n\n"
        
        # Add column descriptions
        for col in df.columns:
            text += f"\nColumn '{col}':\n"
            text += f"- Data type: {df[col].dtype}\n"
            text += f"- Non-null count: {df[col].count()}\n"
            if df[col].dtype in ['int64', 'float64']:
                text += f"- Mean: {df[col].mean():.2f}\n"
                text += f"- Range: {df[col].min()} to {df[col].max()}\n"
            else:
                unique_vals = df[col].unique()[:5]
                text += f"- Sample values: {', '.join(map(str, unique_vals))}\n"
        
        # Add first few rows as context
        text += f"\n\nFirst 5 rows:\n{df.head().to_string()}\n"
        
        return text
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        return ""

def extract_text_from_file(file):
    """Extract text from various file types"""
    file_extension = file.name.lower().split('.')[-1]
    
    if file_extension == 'pdf':
        return extract_text_from_pdf(file)
    elif file_extension == 'docx':
        return extract_text_from_docx(file)
    elif file_extension == 'csv':
        return extract_text_from_csv(file)
    elif file_extension == 'txt':
        return str(file.read(), "utf-8")
    elif file_extension == 'json':
        data = json.loads(file.read())
        return json.dumps(data, indent=2)
    else:
        try:
            return str(file.read(), "utf-8")
        except:
            return "Could not extract text from this file type."

def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into overlapping chunks"""
    if not text:
        return []
    
    # Clean text
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Split into sentences for better chunking
    sentences = re.split(r'[.!?]+', text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        # If adding this sentence would exceed chunk size, save current chunk
        if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            # Start new chunk with overlap
            words = current_chunk.split()
            overlap_text = ' '.join(words[-overlap//10:]) if len(words) > overlap//10 else ""
            current_chunk = overlap_text + " " + sentence
        else:
            current_chunk += " " + sentence
    
    # Add the last chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks

# =============================
# Vector Database Functions
# =============================
class SimpleVectorDB:
    def __init__(self):
        self.documents = []
        self.embeddings = []
        self.metadata = []
    
    def add_documents(self, texts, metadatas):
        """Add documents to the vector database"""
        if not embedding_model:
            st.error("Embedding model not available")
            return
        
        for text, metadata in zip(texts, metadatas):
            if text.strip():  # Only add non-empty chunks
                embedding = embedding_model.encode([text])[0]
                self.documents.append(text)
                self.embeddings.append(embedding)
                self.metadata.append(metadata)
    
    def search(self, query, top_k=5):
        """Search for similar documents"""
        if not self.embeddings or not embedding_model:
            return []
        
        query_embedding = embedding_model.encode([query])[0]
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        
        # Get top k results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                'text': self.documents[idx],
                'score': similarities[idx],
                'metadata': self.metadata[idx]
            })
        
        return results
    
    def clear(self):
        """Clear all documents"""
        self.documents = []
        self.embeddings = []
        self.metadata = []

# Initialize vector database
if 'vector_db' not in st.session_state:
    st.session_state.vector_db = SimpleVectorDB()

# =============================
# Sidebar - Document Management
# =============================
with st.sidebar:
    st.header("📁 Document Management")
    
    # File upload
    uploaded_files = st.file_uploader(
        "Upload Documents",
        type=['pdf', 'docx', 'txt', 'csv', 'json'],
        accept_multiple_files=True,
        help="Supported formats: PDF, DOCX, TXT, CSV, JSON"
    )
    
    # Process uploaded files
    if uploaded_files:
        if st.button("📚 Process Documents", type="primary"):
            with st.spinner("Processing documents..."):
                # Clear existing database
                st.session_state.vector_db.clear()
                
                total_chunks = 0
                for file in uploaded_files:
                    st.write(f"Processing: {file.name}")
                    
                    # Extract text
                    text = extract_text_from_file(file)
                    
                    if text:
                        # Chunk the text
                        chunks = chunk_text(text)
                        
                        # Create metadata for each chunk
                        metadatas = [
                            {
                                'filename': file.name,
                                'chunk_id': i,
                                'total_chunks': len(chunks)
                            }
                            for i in range(len(chunks))
                        ]
                        
                        # Add to vector database
                        st.session_state.vector_db.add_documents(chunks, metadatas)
                        total_chunks += len(chunks)
                
                st.success(f"✅ Processed {len(uploaded_files)} files into {total_chunks} searchable chunks")
                st.rerun()
    
    # Database stats
    st.header("📊 Database Stats")
    num_docs = len(st.session_state.vector_db.documents)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Documents", num_docs)
    with col2:
        files_processed = len(set(meta['filename'] for meta in st.session_state.vector_db.metadata)) if st.session_state.vector_db.metadata else 0
        st.metric("Files", files_processed)
    
    if st.button("🗑️ Clear Database", type="secondary"):
        st.session_state.vector_db.clear()
        st.success("Database cleared!")
        st.rerun()
    
    # Configuration
    st.header("⚙️ Settings")
    
    # Number of results to retrieve
    num_results = st.slider("Results to retrieve", 3, 10, 5)
    
    # OpenAI settings
    if client:
        st.success("✅ OpenAI connected")
        model_choice = st.selectbox(
            "OpenAI Model",
            ["gpt-4o-mini", "gpt-4", "gpt-3.5-turbo"],
            index=0
        )
    else:
        st.warning("⚠️ OpenAI not configured")
        st.info("Set OPENAI_API_KEY in .env file for AI answers")
        model_choice = "gpt-4o-mini"

# =============================
# Main Interface
# =============================

# Question input
st.header("💬 Ask Questions About Your Documents")

if st.session_state.vector_db.documents:
    question = st.text_input(
        "What would you like to know?",
        placeholder="e.g., What are the main topics discussed in these documents?",
        key="question_input"
    )
    
    col1, col2 = st.columns([3, 1])
    with col2:
        search_button = st.button("🔍 Search & Answer", type="primary", use_container_width=True)
    
    if question and (search_button or question):
        with st.spinner("Searching documents..."):
            # Search for relevant documents
            results = st.session_state.vector_db.search(question, top_k=num_results)
            
            if results:
                # Display search results
                st.subheader("📄 Relevant Document Sections")
                
                context_text = ""
                for i, result in enumerate(results):
                    with st.expander(f"Source {i+1}: {result['metadata']['filename']} (Score: {result['score']:.3f})"):
                        st.write(result['text'])
                    
                    context_text += f"Source {i+1} ({result['metadata']['filename']}):\n{result['text']}\n\n"
                
                # Generate AI answer if OpenAI is available
                if client and question:
                    with st.spinner("Generating AI answer..."):
                        try:
                            prompt = f"""Based on the following document excerpts, please answer this question: {question}

Document excerpts:
{context_text}

Please provide a comprehensive answer based on the information provided. If the information is insufficient to fully answer the question, please say so."""

                            response = client.chat.completions.create(
                                model=model_choice,
                                messages=[
                                    {"role": "system", "content": "You are a helpful assistant that answers questions based on provided document excerpts. Be accurate and cite the sources when possible."},
                                    {"role": "user", "content": prompt}
                                ],
                                max_tokens=1000,
                                temperature=0.2
                            )
                            
                            st.markdown(f"""
                            <div class="answer-box">
                                <h3>🤖 AI Answer</h3>
                                <p>{response.choices[0].message.content}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                        except Exception as e:
                            st.error(f"Error generating AI answer: {e}")
                
                # Raw search results for reference
                with st.expander("🔍 View Raw Search Results"):
                    for i, result in enumerate(results):
                        st.write(f"**Result {i+1}** (Score: {result['score']:.3f})")
                        st.write(f"**File:** {result['metadata']['filename']}")
                        st.write(f"**Chunk:** {result['metadata']['chunk_id'] + 1} of {result['metadata']['total_chunks']}")
                        st.write(f"**Text:** {result['text'][:200]}...")
                        st.divider()
            else:
                st.warning("No relevant documents found for your question.")
else:
    st.info("👆 Please upload documents using the sidebar to get started!")
    
    st.markdown("""
    <div class="upload-area">
        <h3>🚀 Get Started</h3>
        <p>1. Upload your documents (PDF, DOCX, TXT, CSV, JSON)</p>
        <p>2. Click "Process Documents" to create searchable embeddings</p>
        <p>3. Ask questions about your document content</p>
        <p>4. Get AI-powered answers with source citations</p>
    </div>
    """, unsafe_allow_html=True)

# =============================
# Example Questions (if documents are loaded)
# =============================
if st.session_state.vector_db.documents:
    st.header("💡 Example Questions")
    
    example_questions = [
        "What are the main topics discussed?",
        "Can you summarize the key findings?",
        "What are the most important points?",
        "Are there any recommendations mentioned?",
        "What data or statistics are provided?",
        "Who are the key people or organizations mentioned?"
    ]
    
    cols = st.columns(3)
    for i, question in enumerate(example_questions):
        with cols[i % 3]:
            if st.button(question, key=f"example_{i}"):
                st.session_state.question_input = question
                st.rerun()

# =============================
# Footer
# =============================
st.markdown("---")
st.markdown("""
<div style="text-align: center; opacity: 0.7;">
    <p>🤖 <strong>RAG Document Q&A Assistant</strong> | Upload documents, ask questions, get intelligent answers</p>
    <p>Powered by SentenceTransformers for embeddings and OpenAI for responses</p>
</div>
""", unsafe_allow_html=True)
