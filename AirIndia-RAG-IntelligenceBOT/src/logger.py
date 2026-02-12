
import logging
import os
from logging.handlers import RotatingFileHandler
from config.settings import LOG_LEVEL, LOG_FILE

def setup_logger(name):
    """
    Sets up a logger with both console and file handlers.
    Ensures the log directory exists.
    """
    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVEL)

    # Prevent adding handlers multiple times
    if logger.hasHandlers():
        return logger

    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - [%(levelname)s] - %(name)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Ensure log directory exists
    log_dir = os.path.dirname(LOG_FILE)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # File Handler (with rotation)
    file_handler = RotatingFileHandler(LOG_FILE, maxBytes=5*1024*1024, backupCount=3)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
