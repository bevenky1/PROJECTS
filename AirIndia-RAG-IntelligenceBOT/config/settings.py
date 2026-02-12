
import os
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# General Configuration
MODEL_TYPE = os.getenv("MODEL_TYPE", "local") # "bedrock" or "local"

# AWS Configuration
AWS_REGION = os.getenv("AWS_REGION", "eu-west-3")
BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "eu.amazon.nova-pro-v1:0")
BEDROCK_EMBEDDING_MODEL_ID = os.getenv("BEDROCK_EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0")

# Local Configuration (Ollama / HuggingFace)
LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "llama3")
LOCAL_EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Vector Store Configuration
VECTOR_DB_DIR = os.getenv("VECTOR_DB_DIR", "./chroma_vectorestore")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "air_india_collection")

# Document Processing
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))


# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("LOG_FILE", "logs/app.log")

