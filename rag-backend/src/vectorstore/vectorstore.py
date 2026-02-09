"""Vector store module for document embedding and retrieval"""

from typing import List
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from src.config.logger import get_logger

logger = get_logger(__name__)

class VectorStore:
    """Manages vector store operations"""
    
    def __init__(self):
        """Initialize vector store with HuggingFace embeddings (free, local)"""
        logger.info("Loading HuggingFace embeddings model (first run may download the model)...")
        self.embedding = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",  # Lightweight, fast, and effective
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.vectorstore = None
        self.retriever = None
        logger.debug("VectorStore initialized with HuggingFace embeddings")
    
    def create_vectorstore(self, documents: List[Document]):
        """
        Create vector store from documents
        
        Args:
            documents: List of documents to embed
        """
        logger.info("Creating vectorstore from %d documents", len(documents))
        self.vectorstore = FAISS.from_documents(documents, self.embedding)
        self.retriever = self.vectorstore.as_retriever()
        logger.info("Vectorstore created and retriever initialized")
    
    def get_retriever(self):
        """
        Get the retriever instance
        
        Returns:
            Retriever instance
        """
        if self.retriever is None:
            logger.error("Attempted to get retriever before initialization")
            raise ValueError("Vector store not initialized. Call create_vectorstore first.")
        logger.debug("Returning retriever instance")
        return self.retriever
    
    def retrieve(self, query: str, k: int = 4) -> List[Document]:
        """
        Retrieve relevant documents for a query
        
        Args:
            query: Search query
            k: Number of documents to retrieve
            
        Returns:
            List of relevant documents
        """
        if self.retriever is None:
            logger.error("Attempted retrieval when vectorstore not initialized")
            raise ValueError("Vector store not initialized. Call create_vectorstore first.")
        logger.info("Retrieving top %d documents for query: %s", k, query)
        docs = self.retriever.invoke(query)
        logger.debug("Retriever returned %d documents", len(docs) if docs else 0)
        return docs