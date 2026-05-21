# Agentic AI Knowledge Assistant — RAG Pipeline

A RAG-based document Q&A system built with Python and Streamlit.
Upload any PDF and ask questions in natural language.

## Features

✅ **PDF Upload** — Upload any PDF file
✅ **Text Extraction** — Automatic text extraction with page tracking
✅ **Smart Chunking** — Split text into optimized chunks
✅ **Vector Search** — FAISS similarity search for relevant content
✅ **Local LLM** — Ollama + Llama 3 (no API keys needed)
✅ **Chat History** — Keep track of all Q&A
✅ **Source Citations** — Shows which page answered your question

## Tech Stack

- **Streamlit** — Interactive web UI
- **PyMuPDF (fitz)** — PDF text extraction
- **LangChain** — Text chunking
- **HuggingFace sentence-transformers** — Embeddings (free, local)
- **FAISS** — Vector similarity search
- **Ollama + Llama 3** — Local LLM (free, no API key)

## Setup Instructions

### 1. Install Ollama

Download from https://ollama.com and install on your system.

### 2. Pull Llama 3

Open terminal and run:
```bash
ollama pull llama3
```

### 3. Clone the Repository

```bash
git clone https://github.com/riyasahamed2912/rag-knowledge-assistant.git
cd rag-knowledge-assistant
```

### 4. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 5. Start Ollama Server

Open a **new terminal** and run:
```bash
ollama serve
```

Leave this terminal open while using the app.

### 6. Run the Streamlit App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## How It Works

1. **Upload PDF** → System extracts text and tracks page numbers
2. **Chunking** → Text split into 512-character chunks with 50-character overlap
3. **Embeddings** → Convert chunks to vector embeddings (all-MiniLM-L6-v2 model)
4. **Storage** → Store embeddings in FAISS vector database
5. **Query** → Convert user question to embedding
6. **Search** → Find top 3 most similar chunks using FAISS
7. **LLM** → Send chunks + question to Llama 3
8. **Answer** → Display response with source page numbers

## Usage Example

```
Q: "What is the main topic of this document?"
→ System searches for relevant chunks
→ Sends to Llama 3 with context
→ Returns answer with page citation

Q: "Summarize the key points"
→ Semantic search finds summary sections
→ Llama 3 generates concise summary
→ Shows source pages

Q: "What does page 2 say?"
→ Searches specifically for page 2 content
→ Returns answer with page citation
```

## Troubleshooting

### ❌ "Connection refused" error

**Problem:** Ollama is not running
**Solution:** Open a new terminal and run `ollama serve`

### ❌ "Model not found" error

**Problem:** Llama 3 is not downloaded
**Solution:** Run `ollama pull llama3`

### ❌ "No text found in PDF" warning

**Problem:** PDF might be scanned image without OCR
**Solution:** Try a PDF with selectable text

### ❌ Slow response time

**Problem:** First query takes time to load model
**Solution:** Subsequent queries will be faster. Use GPU if available.

## Performance Notes

- **First query:** ~5-10 seconds (model loading)
- **Subsequent queries:** ~2-5 seconds
- **Embedding generation:** ~1-2 seconds for PDF processing
- **FAISS search:** <100ms for similarity search

## System Requirements

- **RAM:** 8GB minimum (4GB for Ollama, 2GB for embeddings)
- **Disk:** 5GB for Llama 3 model + dependencies
- **Python:** 3.10+
- **OS:** Windows, macOS, Linux

## Configuration

Edit `app.py` to customize:

```python
# Chunk size and overlap (lines 75-76)
chunk_size=512
chunk_overlap=50

# Number of retrieved chunks (line 111)
top_k=3

# Ollama model (line 140)
"model": "llama3"
```

## Future Improvements

- [ ] Support for multiple PDF formats (DOCX, TXT, MD)
- [ ] Batch processing for multiple files
- [ ] Query history and bookmarking
- [ ] Export answers to PDF
- [ ] Fine-tuned embeddings for domain-specific knowledge
- [ ] Streaming responses for faster feedback
- [ ] Multi-language support
- [ ] Web deployment with Docker

## License

MIT License — Feel free to use and modify!

## Support

Having issues? Check:
1. Is Ollama running? (`ollama serve`)
2. Is Llama 3 installed? (`ollama list`)
3. Python version 3.10+? (`python --version`)
4. All dependencies installed? (`pip install -r requirements.txt`)

---

**Built with ❤️ using Streamlit + FAISS + Ollama**
