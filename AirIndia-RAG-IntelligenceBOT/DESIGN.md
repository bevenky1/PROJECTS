# 🏥 Design Document: AirIndia-RAG-IntelligenceBOT

## 1. Introduction
This document outlines the architectural design and implementation details of the AirIndia-RAG-IntelligenceBOT. The system is designed to provide high-quality, document-grounded responses to user queries regarding airline operations, policies, and services.

## 2. System Goals
*   **Accuracy**: Ensure responses are strictly based on provided source documents.
*   **Modularity**: Separation of concerns between ingestion, retrieval, and generation.
*   **Scalability**: Ability to handle increasing volumes of documents and user traffic.
*   **Maintainability**: Production-grade logging, configuration, and structural sanity.
*   **User Experience**: Fast, responsive, and intuitive chat interface.

## 3. High-Level Architecture
The system follows a classic RAG architecture, split into two primary pipelines:

### 3.1 Data Ingestion Pipeline (Offline/Admin)
1.  **Document Loading**: PDF files are extracted from a localized storage directory.
2.  **Text Splitting**: Documents are partitioned into overlapping chunks using `RecursiveCharacterTextSplitter`.
3.  **Embedding Generation**: Chunks are converted into 1024-dimension vectors using **Amazon Titan Text Embeddings v2**.
4.  **Vector Storage**: Vectors and metadata are persisted in **ChromaDB**, an open-source vector database.

### 3.2 Retrieval & Generation Pipeline (Online/User)
1.  **Query Processing**: User input is received via the Streamlit interface.
2.  **Semantic Retrieval**: The input is embedded, and a similarity search is performed against ChromaDB to find the top-$k$ relevant context chunks.
3.  **System Prompting**: A system prompt is constructed combining the retrieved context and the user's question.
4.  **LLM Inference**: The prompt is sent to **Amazon Nova Pro** via AWS Bedrock.
5.  **Response Delivery**: The generated text is streamed back to the user interface.

## 4. Component Details

### 4.1 Configuration Layer (`config/settings.py`)
*   Uses a centralized approach to environment management via `python-dotenv`.
*   Encapsulates all AWS, Vector DB, and Logging parameters.
*   Provides a single source of truth for application constants.

### 4.2 Application Logic (`src/`)
*   **`embeddings.py`**: Custom wrapper for Bedrock embeddings with built-in token truncation (max 8000 tokens) to ensure API compatibility.
*   **`vector_store.py`**: Manages the lifecycle of the Chroma database. Handles both persistence (writing) and similarity searches (reading).
*   **`llm_engine.py`**: The orchestrator. It manages the prompt templates and coordinates the "Retrieval" and "Generation" steps.
*   **`logger.py`**: Implements a `RotatingFileHandler` system to ensure application activity is tracked without exhausting disk space.

### 4.3 User Interface (`app.py`)
*   Built using **Streamlit**.
*   Implements session-state based chat history.
*   Uses asynchronous-style UI updates (streaming simulation) for better perceived latency.

## 5. Data Flow Diagram (Conceptual)
```text
[PDF Documents] -> [Splitter] -> [Embedder (Titan v2)] -> [ChromaDB]
                                                              |
[User Query] -> [Embedder (Titan v2)] -> [Semantic Search] ---/
                                              |
                                      [Context Chunks]
                                              |
[User Query] + [Context Chunks] -> [LLM (Nova Pro)] -> [User Response]
```

## 6. Design Decisions & Trade-offs
*   **Choice of LLM (Nova Pro)**: Selected for its high reasoning capabilities and competitive latency within the AWS ecosystem.
*   **Choice of Vector DB (ChromaDB)**: Chosen for its ease of local persistence and native integration with LangChain, making it ideal for self-contained production environments.
*   **Token Management**: Implemented manual truncation in the embedding layer to handle edge cases where document fragments might exceed LLM context windows.
*   **Logging Strategy**: Chose a dual-handler (Console + File) approach to support both container logs (standard out) and persistent auditing (file logs).

## 7. Security & Compliance
*   **IAM Roles**: The system is designed to run using AWS IAM roles or environment-based credentials, following the principle of least privilege (only `bedrock:InvokeModel` is required).
*   **Local Data**: Source documents and vector indices remain within the host environment, ensuring data privacy for proprietary airline documents.

## 8. Future Roadmap
*   **Reranking**: Integrate a cross-encoder reranker to improve the relevance of retrieved context.
*   **Multi-Modal Support**: Expand to handle images in PDFs (e.g., flight maps or infographics) using Nova's multi-modal capabilities.
*   **Evaluation Framework**: Implement RAGAS or similar frameworks to quantify retrieval accuracy and generation quality.
