from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langgraph.prebuilt import create_react_agent
from langchain_core.prompts import PromptTemplate
from src.logger import logger

def create_agent(db, llm):
    """
    Creates a ReAct agent for SQL database interaction.
    
    Args:
        db: The SQLDatabase instance.
        llm: The LLM instance.
        
    Returns:
        A compiled LangGraph agent executor.
    """
    logger.info("Creating SQL Agent...")
    try:
        # Create the toolkit
        toolkit = SQLDatabaseToolkit(db=db, llm=llm)
        tools = toolkit.get_tools()

        # Hardcoded system prompt to avoid langchainhub dependency issues
        # Based on langchain-ai/sql-agent-system-prompt
        template = """You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most {top_k} results.
You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns from a specific table, only ask for the relevant columns given the question.
You have access to tools for interacting with the database.
Only use the below tools. Only use the information returned by the below tools to construct your final answer.
You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

To start you should ALWAYS look at the tables in the database to see what you can query.
Do NOT skip this step.
Then you should query the schema of the most relevant tables.
"""
        # Get table names to help the LLM avoid guessing
        try:
            table_names = db.get_usable_table_names()
            table_info = ", ".join(table_names)
        except Exception:
            table_info = "Check the database for table names."

        prompt_template = PromptTemplate.from_template(template)
        base_system_message = prompt_template.format(dialect=db.dialect, top_k=5)
        
        # Enhance system message for natural language response and Schema Info
        enhanced_system_message = (
            f"{base_system_message}\n\n"
            f"The database contains a view named 'denormalized_sales' which consolidates key information (Product, Region, SalesAmount, etc.).\n"
            "ALWAYS check this view first. It is designed to answer most questions without complex joins.\n"
            "Use 'sql_db_schema' tool to inspect 'denormalized_sales' if needed.\n"
            "CRITICAL INSTRUCTION: After executing the SQL query and retrieving the result, "
            "you MUST transform the raw data into a natural, conversational response. "
            "Do not just output the SQL result or the raw list. "
            "Example: Instead of '[(30,)]', say 'There are 30 products in the database.'"
        )

        # Create the agent
        agent_executor = create_react_agent(llm, tools, prompt=enhanced_system_message)
        logger.info("SQL Agent created successfully.")
        
        return agent_executor
    except Exception as e:
        logger.error(f"Error creating agent: {e}")
        raise
