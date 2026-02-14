# Text to SQL Chatbot (Production Grade)

A modular, production-ready Python application that allows users to query a SQL database using natural language. Built with LangChain, LangGraph, and Google Gemini.

## Features

- **Natural Language to SQL**: Converts English queries into SQL and executes them.
- **Agentic Architecture**: Uses a ReAct agent to introspect the schema and recover from errors.
- **Modular Design**: Separated concerns (Database, LLM, Agent, Config).
- **Configuration Management**: Environment-based configuration.

## Installation

1.  **Clone the Repository**
    ```bash
    git clone <repo_url>
    cd Text-to-SQL-Chatbot-main
    ```

2.  **Install Dependencies**
    It is recommended to use a virtual environment.
    ```bash
    pip install .
    ```
    Or manually:
    ```bash
    pip install langchain langchain-community langchain-google-genai pymysql pydantic-settings langgraph cryptography
    ```

3.  **Configuration**
    Create a `.env` file in the root directory. You can choose your LLM provider by setting `LLM_PROVIDER` (default is `google`).

    **Google Gemini (Default)**:
    ```env
    LLM_PROVIDER=google
    GOOGLE_API_KEY=your_google_api_key
    ```

    **OpenAI (ChatGPT)**:
    ```env
    LLM_PROVIDER=openai
    OPENAI_API_KEY=your_openai_api_key
    LLM_MODEL=gpt-4-turbo
    ```

    **Anthropic (Claude)**:
    ```env
    LLM_PROVIDER=anthropic
    ANTHROPIC_API_KEY=your_anthropic_api_key
    LLM_MODEL=claude-3-opus-20240229
    ```

    **Ollama (Local)**:
    ```env
    LLM_PROVIDER=ollama
    LLM_MODEL=llama3
    # Ensure Ollama is running locally
    ```

    **Database Configuration**:
    ```env
    DB_HOST=localhost
    DB_PORT=3306
    DB_USER=root
    DB_PASSWORD=your_password
    DB_NAME=text_to_sql
    ```

    **Note**: If using providers other than Google, you must install their respective packages:
    ```bash
    pip install langchain-openai langchain-anthropic langchain-groq
    ```

    **Local Setup with Ollama**:
    If you choose `LLM_PROVIDER=ollama`:
    1.  Install Ollama from [ollama.com](https://ollama.com).
    2.  Pull the model you configured (e.g., `llama3`):
        ```bash
        ollama pull llama3
        ```
    3.  Ensure Ollama is running (usually `ollama serve`).

## Usage

Run the application:

```bash
python main.py
```

## Project Structure

- `config/`: Configuration settings.
- `src/`: Core application logic.
    - `database.py`: Database connection handling.
    - `llm.py`: LLM initialization.
    - `agent.py`: LangGraph agent definition.
- `main.py`: Application entry point.

## Architecture

The system uses a **ReAct agent** that has access to a SQL Toolkit. The toolkit provides tools to:
1.  List tables.
2.  Get schema info for specific tables.
3.  Execute SQL queries.
4.  Check query syntax.

The agent iteratively plans, executes tools, and observes the output to answer the user's question.
