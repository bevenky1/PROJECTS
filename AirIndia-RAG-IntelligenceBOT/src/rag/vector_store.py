
import os
from typing import List
from uuid import uuid4
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from config.settings import VECTOR_DB_DIR, COLLECTION_NAME, CHUNK_SIZE, CHUNK_OVERLAP
from src.logger import setup_logger
from src.rag.embeddings import get_embedding_function

logger = setup_logger(__name__)

class VectorStoreManager:
    """Manages the creation, persistence, and querying of the Chroma Vector Store."""
    
    def __init__(self, embedding_function):
        self.embedding_function = embedding_function
        self.persist_directory = VECTOR_DB_DIR
        self.collection_name = COLLECTION_NAME
        self.vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedding_function,
            persist_directory=self.persist_directory,
        )

    def load_and_split_documents(self, directory: str) -> List[Document]:
        """Loads PDFs from a directory and splits them into chunks."""
        logger.info(f"Loading documents from {directory}...")
        try:
            loader = PyPDFDirectoryLoader(directory, glob="**/*.pdf")
            documents = loader.load()
            unique_sources = {doc.metadata.get('source', 'unknown') for doc in documents}
            logger.info(f"files found: {len(unique_sources)}")
            for source in unique_sources:
                logger.debug(f"Processing file: {source}")
            
            logger.info(f"Loaded {len(documents)} document pages total.")

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE, 
                chunk_overlap=CHUNK_OVERLAP
            )
            texts = text_splitter.split_documents(documents)
            logger.info(f"Split into {len(texts)} chunks.")
            return texts
        except Exception as e:
            logger.error(f"Failed to load/split documents: {e}")
            return []

    def populate_vector_store(self, documents: List[Document]):
        """Adds documents to the vector store."""
        if not documents:
            logger.warning("No documents to add to vector store.")
            return

        logger.info("Adding documents to vector store...")
        try:
            ids = [str(uuid4()) for _ in range(len(documents))]
            # Batch adding logic could be implemented here for large sets
            self.vector_store.add_documents(documents=documents, ids=ids)
            logger.info("Successfully populated vector store.")
        except Exception as e:
            logger.error(f"Failed to populate vector store: {e}")

    def similarity_search(self, query: str, k: int = 3) -> List[Document]:
        """Searches the vector store for relevant documents."""
        try:
            results = self.vector_store.similarity_search(query, k=k)
            return results
        except Exception as e:
            logger.error(f"Error during similarity search: {e}")
            return []

def initialize_vector_store():
    embeddings = get_embedding_function()
    return VectorStoreManager(embeddings)
