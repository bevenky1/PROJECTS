from langchain_google_genai import ChatGoogleGenerativeAI
# Import other providers conditionally or assume installed if configured
try:
    from langchain_openai import ChatOpenAI
except ImportError:
    pass # OpenAI support not installed

try:
    from langchain_anthropic import ChatAnthropic
except ImportError:
    pass # Anthropic support not installed
    
try:
    from langchain_community.chat_models import ChatOllama
except ImportError:
    pass # Ollama support not installed

try:
    from langchain_groq import ChatGroq
except ImportError:
    pass # Groq support not installed

from config.settings import settings
from src.logger import logger

def get_llm():
    """Initializes and returns the configured LLM based on settings.LLM_PROVIDER."""
    provider = settings.LLM_PROVIDER.lower()
    model = settings.LLM_MODEL
    
    logger.info("Starting LLM initialization...")
    try:
        if provider == "google":
            logger.info("Configuring Google Gemini...")
            if not settings.GOOGLE_API_KEY:
                logger.error("GOOGLE_API_KEY missing.")
                raise ValueError("GOOGLE_API_KEY is not set")
            return ChatGoogleGenerativeAI(
                model=model,
                api_key=settings.GOOGLE_API_KEY,
                temperature=0
            )
            
        elif provider == "openai":
            if not settings.OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY is not set")
            return ChatOpenAI(
                model=model,
                api_key=settings.OPENAI_API_KEY,
                temperature=0
            )
            
        elif provider == "anthropic":
            if not settings.ANTHROPIC_API_KEY:
                raise ValueError("ANTHROPIC_API_KEY is not set")
            return ChatAnthropic(
                model=model,
                api_key=settings.ANTHROPIC_API_KEY,
                temperature=0
            )
            
        elif provider == "ollama":
            logger.info(f"Configuring Local Ollama: {model}")
            try:
                from langchain_ollama import ChatOllama
                logger.info("Using langchain_ollama package.")
            except ImportError:
                from langchain_community.chat_models import ChatOllama
                logger.warning("langchain_ollama not found, falling back to langchain_community.")
                
            return ChatOllama(
                model=model,
                temperature=0
            )
            
        elif provider == "groq":
            if not settings.GROQ_API_KEY:
                raise ValueError("GROQ_API_KEY is not set")
            return ChatGroq(
                model=model,
                api_key=settings.GROQ_API_KEY,
                temperature=0
            )
            
        else:
            raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")
            
    except ImportError as e:
        logger.error(f"Missing dependency for provider '{provider}': {e}")
        logger.info(f"Please install the package for {provider} (e.g., pip install langchain-{provider})")
        raise
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {e}")
        raise
