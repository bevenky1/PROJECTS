import json
import boto3
import tiktoken
from langchain.embeddings.base import Embeddings
from config.settings import AWS_REGION, BEDROCK_EMBEDDING_MODEL_ID, MODEL_TYPE, LOCAL_EMBEDDING_MODEL
from src.logger import setup_logger

logger = setup_logger(__name__)

class AmazonTitanEmbedding(Embeddings):
    """
    Custom LangChain Embedding class for Amazon Titan Bedrock Model.
    Provides robust handling for token limits and embedding generation.
    """
    def __init__(self, region_name: str = AWS_REGION, model_id: str = BEDROCK_EMBEDDING_MODEL_ID):
        try:
            self.client = boto3.client("bedrock-runtime", region_name=region_name)
            self.model_id = model_id
            self.max_tokens = 8000
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
            logger.info(f"Initialized AmazonTitanEmbedding with model {model_id} in {region_name}")
        except Exception as e:
            logger.error(f"Failed to initialize AmazonTitanEmbedding: {e}")
            raise

    def _safe_truncate(self, text: str) -> str:
        """Truncates text to ensure it fits within the model's token limit."""
        try:
            tokens = self.tokenizer.encode(text)
            if len(tokens) > self.max_tokens:
                logger.debug(f"Truncating text from {len(tokens)} to {self.max_tokens} tokens")
                tokens = tokens[:self.max_tokens]
            return self.tokenizer.decode(tokens)
        except Exception as e:
            logger.error(f"Error during tokenization/truncation: {e}")
            return text[:self.max_tokens * 4] # fallback character limit

    def embed_query(self, text: str) -> list:
        """Generates embedding for a single query string."""
        try:
            safe_text = self._safe_truncate(text)
            request = json.dumps({"inputText": safe_text})
            # Consider adding retry logic here if needed
            response = self.client.invoke_model(modelId=self.model_id, body=request)
            return json.loads(response["body"].read())["embedding"]
        except Exception as e:
            logger.error(f"Error embedding query: {e}")
            return []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generates embeddings for a list of documents."""
        embeddings = []
        for i, text in enumerate(texts):
            try:
                safe_text = self._safe_truncate(text)
                embedding = self.embed_query(safe_text)
                if embedding:
                    embeddings.append(embedding)
                else:
                    logger.warning(f"Empty embedding returned for text #{i}")
            except Exception as e:
                logger.warning(f"Skipping text #{i} due to error: {e}")
        return embeddings

def get_embedding_function():
    """Factory function to return the appropriate embedding model based on settings."""
    if MODEL_TYPE == "bedrock":
        logger.info("Using AWS Bedrock (Titan) for embeddings")
        return AmazonTitanEmbedding()
    else:
        logger.info(f"Using Local HuggingFace embeddings ({LOCAL_EMBEDDING_MODEL})")
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(
                model_name=LOCAL_EMBEDDING_MODEL,
                model_kwargs={'device': 'cpu'}
            )
        except ImportError:
            logger.error("langchain-huggingface not installed. Please install it for local embeddings.")
            raise
