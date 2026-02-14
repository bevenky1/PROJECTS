from langchain_community.utilities import SQLDatabase
from config.settings import settings
from src.logger import logger
import os

def get_database_uri() -> str:
    """Constructs the database URI for SQLite."""
    # Build absolute path to data/sales.db
    # Assuming this file is in src/ and data/ is at project root
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_path = os.path.join(base_dir, "data", "sales.db")
    
    # If using a custom name from settings, respect it, but default to the structured path
    if settings.DB_NAME and settings.DB_NAME != "sales.db" and settings.DB_NAME != "simplified_sales.db":
         return f"sqlite:///{settings.DB_NAME}"
         
    return f"sqlite:///{db_path}"

def get_db_connection() -> SQLDatabase:
    """Establishes and returns a database connection."""
    uri = get_database_uri()
    logger.info(f"Connecting to database at {uri}...")
    try:
        db = SQLDatabase.from_uri(uri, sample_rows_in_table_info=3)
        logger.info("Database connection established successfully.")
        return db
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        raise
