import logging
import sys
import os

def setup_logger(name: str = "text_to_sql_chatbot"):
    """
    Sets up and returns a logger with standard formatting.
    """
    logger = logging.getLogger(name)
    
    # Only configure if no handlers exist
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        # Create a stream handler for console output
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(logging.INFO)

        # Use absolute path for log file to ensure it's created where expected
        # This assumes the script is run from a directory where __file__ is meaningful.
        # If this function is part of a package, adjust os.path.dirname calls as needed.
        log_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.log")
        
        # Create a file handler for logging to a file
        # Use 'a' mode (append) and encoding utf-8
        file_handler = logging.FileHandler(log_path, mode='a', encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        
        # Create a formatter and add it to the handlers
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        stream_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        # Add the handlers to the logger
        logger.addHandler(stream_handler)
        logger.addHandler(file_handler)
        
    return logger

logger = setup_logger()

