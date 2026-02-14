import sys
import os

# Ensure the current directory is in the python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.database import get_db_connection
from src.llm import get_llm
from src.agent import create_agent
from config.settings import settings
from src.logger import logger

def main():
    logger.info("Initializing Text-to-SQL Chatbot...")
    
    try:
        # Initialize components
        db = get_db_connection()
        logger.info(f"Connected to database: {settings.DB_NAME}")
        
        llm = get_llm()
        agent = create_agent(db, llm)
        
        print("\nChatbot ready! Type 'exit' to quit.\n")
        
        while True:
            user_input = input("User: ")
            if user_input.lower() in ["exit", "quit", "q"]:
                logger.info("User requested exit. Shutting down.")
                break
            
            try:
                logger.info(f"Processing query: {user_input}")
                # The input to the agent is a list of messages
                response = agent.invoke({"messages": [("user", user_input)]})
                
                # The response 'messages' list's last element is the AI's final answer
                ai_message = response["messages"][-1]
                print(f"AI: {ai_message.content}\n")
                logger.debug(f"Response content: {ai_message.content}")
                
            except Exception as e:
                logger.error(f"Error processing query: {e}")
                print("Sorry, I encountered an error while processing your request.")

    except Exception as e:
        logger.critical(f"Critical Error: {e}")
        print("Please check your .env file and database connection.")

if __name__ == "__main__":
    main()
