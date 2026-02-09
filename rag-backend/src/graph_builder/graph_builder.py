"""Graph builder for LangGraph workflow"""

from langgraph.graph import StateGraph, END
from src.state.rag_state import RAGState
from src.node.reactnode import RAGNodes
from src.config.logger import get_logger

logger = get_logger(__name__)

class GraphBuilder:
    """Builds and manages the LangGraph workflow"""
    
    def __init__(self, retriever, llm):
        """
        Initialize graph builder
        
        Args:
            retriever: Document retriever instance
            llm: Language model instance
        """
        self.nodes = RAGNodes(retriever, llm)
        self.graph = None
    
    def build(self):
        """
        Build the RAG workflow graph
        
        Returns:
            Compiled graph instance
        """
        logger.info("Building RAG workflow graph")
        # Create state graph
        builder = StateGraph(RAGState)
        
        # Add nodes
        builder.add_node("retriever", self.nodes.retrieve_docs)
        builder.add_node("responder", self.nodes.generate_answer)
        
        # Set entry point
        builder.set_entry_point("retriever")
        
        # Add edges
        builder.add_edge("retriever", "responder")
        builder.add_edge("responder", END)
        
        # Compile graph
        self.graph = builder.compile()
        logger.info("Graph built successfully")
        return self.graph
    
    def run(self, question: str) -> dict:
        """
        Run the RAG workflow
        
        Args:
            question: User question
            
        Returns:
            Final state with answer
        """
        if self.graph is None:
            logger.debug("Graph is not built yet; building now")
            self.build()

        logger.info("Invoking graph for question: %s", question)
        initial_state = RAGState(question=question)
        result = self.graph.invoke(initial_state)
        logger.info("Graph invocation complete")
        return result