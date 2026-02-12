
import os
import argparse
from src.rag.vector_store import VectorStoreManager
from src.rag.embeddings import get_embedding_function
from config.settings import VECTOR_DB_DIR
from src.logger import setup_logger

logger = setup_logger(__name__)

def ingest_data(docs_dir: str):
    """
    Ingests PDF documents from the specified directory into the vector store.
    """
    if not os.path.exists(docs_dir):
        logger.error(f"Documents directory '{docs_dir}' not found.")
        return

    logger.info("Initializing embedding model...")
    embeddings = get_embedding_function()
    
    logger.info("Initializing vector store manager...")
    vs_manager = VectorStoreManager(embeddings)

    logger.info(f"Loading and splitting documents from {docs_dir}...")
    documents = vs_manager.load_and_split_documents(docs_dir)
    
    if documents:
        logger.info(f"Adding {len(documents)} chunks to the vector store...")
        vs_manager.populate_vector_store(documents)
        logger.info("Data ingestion completed successfully.")
    else:
        logger.warning("No documents were processed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest PDF documents into Chroma Vector Store.")
    parser.add_argument("--docs-dir", type=str, default="AirIndia", help="Path to the directory containing PDF documents.")
    args = parser.parse_args()
    
    ingest_data(args.docs_dir)
