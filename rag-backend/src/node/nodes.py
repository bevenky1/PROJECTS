"""LangGraph nodes for RAG workflow"""

from src.state.rag_state import RAGState
from src.config.logger import get_logger

logger = get_logger(__name__)

class RAGNodes:
    """Contains node functions for RAG workflow"""
    
    def __init__(self, retriever, llm):
        """
        Initialize RAG nodes
        
        Args:
            retriever: Document retriever instance
            llm: Language model instance
        """
        self.retriever = retriever
        self.llm = llm
    
    def retrieve_docs(self, state: RAGState) -> RAGState:
        """
        Retrieve relevant documents node
        
        Args:
            state: Current RAG state
            
        Returns:
            Updated RAG state with retrieved documents
        """
        logger.info("Retrieving documents for question: %s", state.question)
        docs = self.retriever.invoke(state.question)
        logger.debug("Retrieved %d documents", len(docs) if docs else 0)
        return RAGState(
            question=state.question,
            retrieved_docs=docs
        )
    
    def generate_answer(self, state: RAGState) -> RAGState:
        """
        Generate answer from retrieved documents node
        
        Args:
            state: Current RAG state with retrieved documents
            
        Returns:
            Updated RAG state with generated answer
        """
        logger.info("Generating answer for question: %s", state.question)
        # Combine retrieved documents into context
        context = "\n\n".join([doc.page_content for doc in state.retrieved_docs])

        # Create prompt
        prompt = f"""Answer the question based on the context.

    Context:
    {context}

    Question: {state.question}"""

        # Generate response
        response = self.llm.invoke(prompt)
        logger.debug("LLM response length: %s", len(getattr(response, 'content', '') or ""))

        return RAGState(
            question=state.question,
            retrieved_docs=state.retrieved_docs,
            answer=response.content
        )