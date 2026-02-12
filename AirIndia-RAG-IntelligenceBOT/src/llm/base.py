from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generates a response for the given prompt."""
        pass

    @abstractmethod
    def evaluate(self, prompt: str) -> str:
        """Evaluates a response, typically expecting JSON output."""
        pass
