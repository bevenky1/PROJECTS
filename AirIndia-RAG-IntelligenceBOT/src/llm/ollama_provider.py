import time
from typing import Optional
from langchain_ollama import OllamaLLM
from src.llm.base import LLMProvider
from config.settings import LOCAL_LLM_MODEL
from src.logger import setup_logger

logger = setup_logger(__name__)

class OllamaProvider(LLMProvider):
    """Wrapper for Local Ollama models."""
    
    def __init__(self, model_name: str = LOCAL_LLM_MODEL):
        self.model_name = model_name
        try:
            self.llm = OllamaLLM(model=self.model_name)
            logger.info(f"Initialized OllamaProvider with model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize OllamaLLM: {e}")
            raise

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generates response using Ollama."""
        start_time = time.time()
        try:
            # Note: Ollama via Langchain applies system prompt differently depending on model template
            # For simplicity, we just pass the prompt. 
            # If system_prompt is crucial, we might prepend it.
            full_prompt = prompt
            if system_prompt:
               full_prompt = f"System: {system_prompt}\n\nUser: {prompt}"
               
            logger.info(f"Invoking Local Ollama model {self.model_name}...")
            response = self.llm.invoke(full_prompt)
            elapsed = time.time() - start_time
            logger.info(f"Received response from Ollama in {elapsed:.2f}s")
            return response
        except Exception as e:
            logger.error(f"Error generating response from Ollama: {e}")
            raise # Let the caller handle the fallback or user message

    def evaluate(self, prompt: str) -> str:
        """Evaluates response using Ollama (same as generate for now)."""
        return self.generate(prompt)
