# Text-to-SQL Chatbot Design Document

## 1. Executive Summary
The Text-to-SQL Chatbot is an intelligent interface that allows users to query a structured relational database using natural language. It leverages Large Language Models (LLMs) to translate user questions into SQL, executes them against the database, and returns conversational answers.

## 2. Architecture Overview

### High-Level Components
1.  **Frontend (Streamlit)**: Provides a chat interface for user interaction.
2.  **Orchestrator (LangGraph)**: Manages the flow between the agent, tools, and LLM.
3.  **LLM (Ollama/Llama 3)**: The reasoning engine that understands natural language and generates SQL.
4.  **Database (SQLite)**: Stores the business data.
5.  **Tools (LangChain)**: specialized functions that allow the LLM to inspect schema and execute queries.

### Data Flow
1.  **User Input**: User asks a question (e.g., "Total sales in North region?").
2.  **Agent Reasoning**: The LLM analyzes the question and the database schema.
3.  **SQL Generation**: The LLM generates a SQL query targeting the `denormalized_sales` view.
4.  **Critique & Execution**: The agent (optionally) checks the query syntax and executes it.
5.  **Response Synthesis**: The LLM interprets the SQL results and generates a natural language response.
6.  **Output**: The response is displayed to the user.

## 3. Technology Stack

-   **Language**: Python 3.10+
-   **Frontend**: Streamlit
-   **LLM Orchestration**: LangChain, LangGraph
-   **LLM Provider**: Ollama (Local Llama 3.2), extensible to Google Gemini/OpenAI
-   **Database**: SQLite
-   **Logging**: Python `logging` module

## 4. Database Schema
To simplify the interaction for smaller LLMs, the system relies on a **SQL View** that denormalizes the complex schema into a single, analytics-ready virtual table.

### `denormalized_sales` (View)
-   `OrderNumber`: Unique identifier
-   `OrderDate`: Date of order
-   `Product`: Name of the product
-   `Customer`: Name of the customer
-   `City`: Delivery city
-   `Region`: Delivery region (North, South, East, West)
-   `Quantity`: Number of units
-   `Unit Price`: Price per unit
-   `SalesAmount`: Total value of the line item

## 5. Security & Performance
-   **Read-Only Access**: The agent is instructed not to perform DML operations.
-   **Local Execution**: Data stays local when using Ollama, ensuring privacy.
-   **Logging**: Comprehensive logs in `app.log` for debugging and auditing.

## 6. Future Improvements
-   **Multi-turn Memory**: Enhanced context retention for follow-up questions.
-   **Fine-tuning**: Training a small model specifically on the schema for higher accuracy.
-   **Visualizations**: Auto-generating charts (bar/line) based on the data returned.
