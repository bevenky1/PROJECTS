"""Configuration module for Agentic RAG system"""

import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for RAG system"""
    
    # Model Configuration - Using Ollama (open source)
    LLM_MODEL = "llama3.2"  # You can also use: mistral, llama3, gemma2, phi3
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    # Document Processing
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
    
    # Default URLs
    DEFAULT_URLS = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/"
    ]
    
    @classmethod
    def get_llm(cls):
        """Initialize and return the Ollama LLM model"""
        return ChatOllama(
            model=cls.LLM_MODEL,
            base_url=cls.OLLAMA_BASE_URL,
            temperature=0.7
        )