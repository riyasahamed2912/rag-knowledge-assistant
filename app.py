import streamlit as st
import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import requests
import os
import pickle

# ============================================================
# STEP 2: STREAMLIT PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="AI Knowledge Assistant",
    page_icon="🤖",
    layout="centered"
)

# ============================================================
# STEP 3: LOAD EMBEDDING MODEL (Cached)
# ============================================================
@st.cache_resource
def load_embedding_model():
    """Load sentence-transformers embedding model once and cache it"""
    return SentenceTransformer('all-MiniLM-L6-v2')

embed_model = load_embedding_model()

# ============================================================
# STEP 4: PDF TEXT EXTRACTION FUNCTION
# ============================================================
def extract_text_with_pages(pdf_file):
    """
    Extract text from PDF file with page tracking.
    Returns list of tuples: (page_number, text)
    Skips empty pages.
    """
    try:
        pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
        pages_text = []
        
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            text = page.get_text()
            
            # Skip empty pages
            if text.strip():
                pages_text.append((page_num + 1, text))
        
        pdf_document.close()
        return pages_text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return []

# ============================================================
# STEP 5: TEXT CHUNKING FUNCTION
# ============================================================
def chunk_text(pages_text):
    """
    Split text into chunks using LangChain RecursiveCharacterTextSplitter.
    chunk_size = 512
    chunk_overlap = 50
    Returns list of dicts: {"text": chunk, "page": page_number}
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=50
    )
    
    chunks = []
    for page_num, page_text in pages_text:
        split_chunks = splitter.split_text(page_text)
        for chunk in split_chunks:
            chunks.append({
                "text": chunk,
                "page": page_num
            })
    
    return chunks

# ============================================================
# STEP 6: FAISS INDEX BUILD FUNCTION
# ============================================================
def build_faiss_index(chunks, embed_model):
    """
    Build FAISS index from chunks.
    - Convert text to embeddings
    - Create IndexFlatL2 (dimension 384)
    - Add all embeddings
    Returns: (index, chunks, embeddings)
    """
    try:
        # Extract text from chunks
        texts = [chunk["text"] for chunk in chunks]
        
        # Generate embeddings
        embeddings = embed_model.encode(texts, convert_to_numpy=True)
        embeddings = np.array(embeddings, dtype=np.float32)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        
        return index, chunks, embeddings
    except Exception as e:
        st.error(f"Error building FAISS index: {e}")
        return None, None, None

# ============================================================
# STEP 7: SEARCH FUNCTION
# ============================================================
def search_chunks(query, index, chunks, embed_model, top_k=3):
    """
    Search FAISS index for top_k most similar chunks.
    Returns list of matched chunk dicts with text and page.
    """
    try:
        # Convert query to embedding
        query_embedding = embed_model.encode([query], convert_to_numpy=True)
        query_embedding = np.array(query_embedding, dtype=np.float32)
        
        # Search FAISS index
        distances, indices = index.search(query_embedding, top_k)
        
        # Return matched chunks
        matched_chunks = [chunks[i] for i in indices[0]]
        return matched_chunks
    except Exception as e:
        st.error(f"Error searching chunks: {e}")
        return []

# ============================================================
# STEP 8: OLLAMA LLM FUNCTION
# ============================================================
def ask_ollama(question, context_chunks):
    """
    Call Ollama API with question and context.
    Model: llama3
    Returns the response text.
    """
    try:
        # Build context string
        context = "\n\n".join([chunk["text"] for chunk in context_chunks])
        
        # Build prompt
        prompt = f"""You are a helpful assistant. Answer the question using ONLY the context provided below. If the answer is not in the context, say: 'I don't have enough information in the document to answer this.'

Context:
{context}

Question: {question}

Answer:"""
        
        # Call Ollama API
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3",
                "prompt": prompt,
                "stream": False
            },
            timeout=300
        )
        
        if response.status_code == 200:
            result = response.json()
            return result.get("response", "No response from Ollama")
        else:
            return f"Error: Ollama returned status code {response.status_code}"
    
    except requests.exceptions.ConnectionError:
        return "❌ Error: Ollama is not running. Please start Ollama with 'ollama serve' in your terminal."
    except Exception as e:
        return f"❌ Error calling Ollama: {e}"

# ============================================================
# STEP 9 & 10: STREAMLIT UI WITH SESSION STATE
# ============================================================

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None
if "chunks" not in st.session_state:
    st.session_state.chunks = None
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False
if "pdf_filename" not in st.session_state:
    st.session_state.pdf_filename = None

# Title
st.title("🤖 AI Knowledge Assistant")
st.markdown("Upload a PDF and ask questions in natural language")

# ============================================================
# SIDEBAR: PDF UPLOAD
# ============================================================
with st.sidebar:
    st.header("📄 Upload Document")
    pdf_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if pdf_file is not None:
        # Check if it's a new file
        if st.session_state.pdf_filename != pdf_file.name:
            st.session_state.pdf_filename = pdf_file.name
            st.session_state.pdf_processed = False
            st.session_state.messages = []  # Clear chat history
        
        # Process PDF
        if not st.session_state.pdf_processed:
            with st.spinner("Processing PDF..."):
                # Extract text
                pages_text = extract_text_with_pages(pdf_file)
                
                if not pages_text:
                    st.error("No text found in PDF")
                else:
                    # Chunk text
                    chunks = chunk_text(pages_text)
                    
                    if chunks:
                        # Build FAISS index
                        index, chunks, embeddings = build_faiss_index(chunks, embed_model)
                        
                        if index is not None:
                            st.session_state.faiss_index = index
                            st.session_state.chunks = chunks
                            st.session_state.pdf_processed = True
                            st.success(f"✅ Document loaded: {pdf_file.name}")
                            st.info(f"📊 Chunks created: {len(chunks)}")
        else:
            st.success(f"✅ Document loaded: {pdf_file.name}")
            st.info(f"📊 Chunks created: {len(st.session_state.chunks)}")

# ============================================================
# MAIN AREA: CHAT AND QA
# ============================================================

if not st.session_state.pdf_processed:
    st.info("👈 Upload a PDF from the sidebar to get started")
else:
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "source" in message:
                st.caption(message["source"])
    
    # Chat input
    question = st.chat_input("Ask a question about the document...")
    
    if question:
        # Add user message to chat
        st.session_state.messages.append({
            "role": "user",
            "content": question
        })
        
        with st.chat_message("user"):
            st.markdown(question)
        
        # Search FAISS
        matched_chunks = search_chunks(question, st.session_state.faiss_index, st.session_state.chunks, embed_model, top_k=3)
        
        # Get answer from Ollama
        with st.spinner("Thinking..."):
            answer = ask_ollama(question, matched_chunks)
        
        # Get source pages
        source_pages = sorted(set([chunk["page"] for chunk in matched_chunks]))
        source_text = f"📄 Source: Page {', '.join(map(str, source_pages))}"
        
        # Add assistant message to chat
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "source": source_text
        })
        
        # Display assistant response
        with st.chat_message("assistant"):
            st.markdown(answer)
            st.caption(source_text)
