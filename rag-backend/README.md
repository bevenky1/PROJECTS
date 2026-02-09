# Agentic RAG System (Local)

A powerful Retrieval Augmented Generation (RAG) system built with **LangGraph**, **Ollama** (Llama 3.2), and **HuggingFace Embeddings**. This application allows users to query documents using an autonomous agent workflow that can search through indexed documents and external sources (Wikipedia).

## 🚀 Features

- **Local LLM Inference**: Uses [Ollama](https://ollama.com/) with `llama3.2` for private, cost-effective inference.
- **Local Embeddings**: Uses `all-MiniLM-L6-v2` via HuggingFace for fast, local vector embeddings.
- **Agentic Workflow**: Built on LangGraph to enable ReAct (Reasoning + Acting) capabilities.
- **Multi-Source Retrieval**:
  - **Vector Store**: Retrieves information from ingested PDFs and URLs.
  - **Wikipedia**: Fallback tool for general knowledge queries.
- **Interactive Mode**: Chat interface for continuous Q&A.

## 📂 Project Structure

```
Agentic-RAG-Local/
└── rag-backend/          # Application Source Code
    ├── data/             # Data directory for PDFs and URLs
    ├── src/              # Source code modules
    │   ├── config/       # Configuration (Ollama, Logging)
    │   ├── document_ingestion/ # PDF & URL Loaders
    │   ├── graph_builder/      # LangGraph Workflow Construction
    │   ├── node/              # Graph Nodes (Retrieval, Generation)
    │   ├── state/             # State Definitions
    │   └── vectorstore/       # FAISS Vector Store Logic
    ├── main.py           # Entry point
    ├── requirements.txt  # Python Dependencies
    └── .env              # Configuration Variables
```

## 🛠️ Prerequisites

1. **Python 3.10+**
2. **Ollama**: Download and install from [ollama.com](https://ollama.com).
3. **Pull Llama 3.2 Model**:
   ```bash
   ollama pull llama3.2
   ```

## 📦 Installation

1. Navigate to the backend directory:
   ```bash
   cd rag-backend
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) Configure Environment:
   The `.env` file is pre-configured for local usage.
   ```env
   # .env
   OLLAMA_BASE_URL="http://localhost:11434"
   ```

## ▶️ Usage

1. **Start Ollama Server** (if not running):
   ```bash
   ollama serve
   ```

2. **Run the Application**:
   ```bash
   python main.py
   ```

3. **Interactive Mode**:
   After the example questions, type `y` to enter interactive mode and ask your own questions.

## 🏗️ Architecture

- **Ingestion**: Documents are loaded from `data/urls.txt` and `data/` folder (PDFs), split into chunks, and embedded into a FAISS vector store.
- **Retrieval**: Semantic search using HuggingFace embeddings (`all-MiniLM-L6-v2`).
- **Generation**: The `llama3.2` model generates answers based on retrieved context.
- **Orchestration**: LangGraph manages the flow between retrieval and generation.

## 📝 License

[MIT License](LICENSE)
