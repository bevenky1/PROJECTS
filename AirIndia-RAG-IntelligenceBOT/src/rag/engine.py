import re
import json
import os
import time
from typing import Dict, Any, List, Optional
from config.settings import MODEL_TYPE
from config.prompts import RAG_PROMPT_TEMPLATE, SEARCH_QUERY_GENERATOR_PROMPT, EVALUATION_PROMPT
from src.logger import setup_logger
from src.rag.vector_store import VectorStoreManager
from src.llm.base import LLMProvider
from src.llm.bedrock_provider import BedrockProvider
from src.llm.ollama_provider import OllamaProvider

logger = setup_logger(__name__)

class RAGEngine:
    """Core RAG logic using a modular LLM provider and a Vector Store."""

    def __init__(self, vector_store_manager: VectorStoreManager, llm_provider: Optional[LLMProvider] = None):
        self.vector_store = vector_store_manager
        
        # Initialize LLM Provider if not passed externally
        if llm_provider:
            self.llm = llm_provider
        else:
            self._initialize_llm()
            
    def _initialize_llm(self):
        """Initializes the LLM provider based on configuration."""
        self.model_type = MODEL_TYPE
        try:
            if self.model_type == "bedrock":
                self.llm = BedrockProvider()
            else:
                self.llm = OllamaProvider()
            logger.info(f"Initialized RAGEngine with {self.model_type.upper()} provider.")
        except Exception as e:
            logger.error(f"Failed to initialize LLM provider: {e}")
            raise

    def _format_chat_history(self, chat_history: List[Dict[str, str]]) -> str:
        """Formats the chat history list into a string."""
        formatted_history = ""
        for message in chat_history:
            role = "User" if message["role"] == "user" else "Assistant"
            formatted_history += f"{role}: {message['content']}\n"
        return formatted_history

    def generate_response(self, question: str, chat_history: List[Dict[str, str]] = None) -> tuple[str, list[str]]:
        """Generates a response using the chosen LLM with retrieved context and chat history."""
        if chat_history is None:
            chat_history = []

        formatted_history = self._format_chat_history(chat_history)
        
        # 1. Query Condensation
        search_query = question
        if chat_history:
            logger.info("Generating optimized search query with chat history...")
            condense_prompt = SEARCH_QUERY_GENERATOR_PROMPT.format(
                chat_history=formatted_history, 
                question=question
            )
            try:
                search_query = self.llm.generate(condense_prompt).strip()
                logger.info(f"Condensed query: {search_query}")
            except Exception as e:
                logger.warning(f"Query condensation failed, using original query: {e}")
                search_query = question

        # 2. Retrieve relevant documents (Meta-question check)
        is_meta_question = any(word in search_query.lower() for word in ["asked you", "previous questions", "our conversation", "my last question"])
        
        if is_meta_question:
            logger.info("Meta-history question detected. Skipping vector retrieval.")
            context = "The user is asking about the previous conversation history, not requesting information from external documents."
            sources = ["System Memory"]
        else:
            docs = self.vector_store.similarity_search(search_query)
            context = "\n\n".join([doc.page_content for doc in docs])
            sources = sorted(list(set([os.path.basename(doc.metadata.get('source', 'Unknown')) for doc in docs])))

        if not context and not is_meta_question:
            logger.warning(f"No relevant context found for: {search_query}")
            context = "No specific documents found."
            sources = []

        # 3. Generate final answer
        prompt = RAG_PROMPT_TEMPLATE.format(
            chat_history=formatted_history if formatted_history else "No previous conversation.",
            context=context, 
            question=question
        )

        try:
            response = self.llm.generate(prompt)
            return response, sources
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I'm sorry, I encountered an error while processing your request.", []

    def evaluate_response(self, question: str, response: str, context: str) -> Dict[str, Any]:
        """Uses the LLM to evaluate a generated response against the context."""
        eval_prompt = EVALUATION_PROMPT.format(
            context=context,
            question=question,
            response=response
        )
        
        try:
            raw_eval = self.llm.evaluate(eval_prompt)
            
            # Extract JSON more robustly using regex
            match = re.search(r'\{.*\}', raw_eval, re.DOTALL)
            if match:
                clean_eval = match.group(0)
            else:
                clean_eval = raw_eval.strip()
            
            return json.loads(clean_eval)
        except Exception as e:
            logger.error(f"Error evaluating response: {e}")
            return {"score": 0, "reasoning": f"Evaluation failed: {e}"}
