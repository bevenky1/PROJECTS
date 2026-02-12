# ✈️ AirIndia-RAG-IntelligenceBOT

![Air India Logo](https://upload.wikimedia.org/wikipedia/en/thumb/9/9b/Air_India_Logo.svg/1200px-Air_India_Logo.svg.png)

A production-grade **Retrieval-Augmented Generation (RAG)** system designed to provide instant, accurate answers about Air India's policies, flight operations, baggage rules, and more. This project leverages the power of **AWS Bedrock** and **ChromaDB** to deliver a state-of-the-art AI assistant experience.

---

## 🌟 Key Features

- **🧠 Intelligent Q&A**: Context-aware answers powered by Amazon Nova Pro.
- **🔍 Semantic Search**: High-performance document retrieval using Titan Text Embeddings v2.
- **💬 Chat History**: Multi-turn conversation support with intelligent query condensation.
- **📊 RAGAS Evaluation**: Quantifiable metrics (Faithfulness, Relevancy, etc.) for pipeline performance.
- **⚖️ LLM as a Judge**: Automated quality scoring and hallucination detection using LLM reasoning.
- **📂 Multi-Document Support**: Process and query complex PDF documents seamlessly.
- **🛡️ Production Ready**: Modular architecture, robust error handling, and structured logging.
- **💻 Modern UI**: Responsive chat interface with history and real-time streaming effects.
- **🗂️ Persistent Memory**: Efficient local vector storage with ChromaDB.
- **🐳 Containerized**: Fully Dockerized for consistent deployment across environments.

---

## 🏗️ Architecture Overview

The system follows a standard RAG pipeline:
1.  **Ingestion**: PDFs are loaded, split into chunks, and embedded via AWS Bedrock Titan.
2.  **Storage**: Embeddings are stored in a local ChromaDB collection.
3.  **Retrieval**: When a query is made, the most relevant chunks are retrieved using similarity search.
4.  **Generation**: The query and context are passed to Amazon Nova Pro to generate a natural language response.

---

## 🛠️ Tech Stack

- **Large Language Model (LLM)**: [AWS Bedrock](https://aws.amazon.com/bedrock/) (Amazon Nova Pro) / Local (Ollama)
- **Embeddings**: Amazon Titan Text Embeddings v2 / HuggingFace Local
- **Vector Database**: [ChromaDB](https://www.trychroma.com/)
- **Evaluation**: [RAGAS](https://docs.ragas.io/), LLM as a Judge
- **Frontend Framework**: [Streamlit](https://streamlit.io/)
- **Orchestration**: [LangChain](https://www.langchain.com/)
- **Containerization**: [Docker](https://www.docker.com/)

---

## 📂 Project Structure

```text
.
├── config/               # Application configuration logic
│   └── settings.py       # Centalized environment variable management
├── src/                  # Core source code
│   ├── logger.py         # Advanced logging (Console + Rotating File)
│   ├── embeddings.py     # Custom AWS Bedrock embedding wrappers
│   ├── vector_store.py   # ChromaDB management & document processing
│   ├── llm_engine.py     # RAG pipeline & Bedrock logic
│   └── ingest_data.py    # Command-line tool for document ingestion
├── tests/                # Automated test suite (Quality, RAGAS, Combined)
├── reports/              # Generated performance and quality reports
├── logs/                 # Persistent log storage
├── AirIndia/             # Default directory for source PDF files
├── app.py                # Main Streamlit application
├── Makefile              # Automation for common tasks
├── Dockerfile            # Container definition
├── .env.example          # Environment configuration template
└── README.md             # Project documentation
```

---

## ⚡ Quick Start

### 1. Prerequisites
- Python 3.10+
- AWS Account with Bedrock access (`nova-pro-v1` & `titan-embed-v2`)
- Properly configured AWS Credentials (via `~/.aws/credentials` or ENV vars)

### 2. Installation
```bash
# Clone the repository
git clone https://github.com/your-repo/air-india-rag.git
cd air-india-rag

# Setup virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
make install
```

### 3. Configuration
Copy the template and fill in your details:
```bash
cp .env.example .env
```

### 4. Data Ingestion
Place your PDFs in the `AirIndia/` folder, then run:
```bash
make ingest
```

### 5. Launch the Application
```bash
make run
```

---

## 🐳 Docker Deployment

To run the application inside a container:

```bash
# Build the image
docker build -t air-india-rag .

# Run the container
docker run -p 8501:8501 --env-file .env air-india-rag
```

---

## 📊 Monitoring & Logs

The application implements a robust logging system. You can monitor activities in real-time:

- **Console Output**: Real-time summary of operations.
- **File Logs**: Detailed trace stored in `logs/app.log` with automatic rotation.

```bash
# View last 50 log entries
tail -n 50 logs/app.log
```

---

## 🧪 Development

### Running Standard Quality Tests (LLM Judge)
```bash
make report
```

### Running RAGAS Evaluation
```bash
make ragas-report
```

### Running Combined Evaluation (Judge + RAGAS)
```bash
make combined-report
```

### Code Formatting
```bash
make format
make lint
```

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
**Disclaimer**: *This is an independent project and is not officially affiliated with Air India.*
