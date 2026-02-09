"""Document processing module for loading and splitting documents"""

from typing import List
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from typing import List, Union
from pathlib import Path
from langchain_community.document_loaders import (
    WebBaseLoader,
    PyPDFLoader,
    TextLoader,
    PyPDFDirectoryLoader
)
from src.config.logger import get_logger

logger = get_logger(__name__)

class DocumentProcessor:
    """Handles document loading and processing"""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initialize document processor
        
        Args:
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        logger.debug("DocumentProcessor initialized with chunk_size=%s, chunk_overlap=%s", chunk_size, chunk_overlap)
    def load_from_url(self, url: str) -> List[Document]:
        """Load document(s) from a URL"""
        logger.info("Loading document from URL: %s", url)
        loader = WebBaseLoader(url)
        docs = loader.load()
        logger.debug("Loaded %d documents from URL", len(docs))
        return docs

    def load_from_pdf_dir(self, directory: Union[str, Path]) -> List[Document]:
        """Load documents from all PDFs inside a directory"""
        logger.info("Loading PDFs from directory: %s", directory)
        loader = PyPDFDirectoryLoader(str(directory))
        docs = loader.load()
        logger.debug("Loaded %d documents from PDF directory", len(docs))
        return docs

    def load_from_txt(self, file_path: Union[str, Path]) -> List[Document]:
        """Load document(s) from a TXT file"""
        logger.info("Loading text file: %s", file_path)
        loader = TextLoader(str(file_path), encoding="utf-8")
        docs = loader.load()
        logger.debug("Loaded %d documents from txt", len(docs))
        return docs

    def load_from_pdf(self, file_path: Union[str, Path]) -> List[Document]:
        """Load document(s) from a PDF file"""
        logger.info("Loading single PDF: %s", file_path)
        loader = PyPDFDirectoryLoader(str("data"))
        docs = loader.load()
        logger.debug("Loaded %d documents from pdf", len(docs))
        return docs
    
    def load_documents(self, sources: List[str]) -> List[Document]:
        """
        Load documents from URLs, PDF directories, or TXT files

        Args:
            sources: List of URLs, PDF folder paths, or TXT file paths

        Returns:
            List of loaded documents
        """
        docs: List[Document] = []
        for src in sources:
            if src.startswith("http://") or src.startswith("https://"):
                docs.extend(self.load_from_url(src))
           
            path = Path("data")
            if path.is_dir():  # PDF directory
                docs.extend(self.load_from_pdf_dir(path))
            elif path.suffix.lower() == ".txt":
                docs.extend(self.load_from_txt(path))
            else:
                logger.warning("Unsupported source type encountered: %s", src)
                raise ValueError(
                    f"Unsupported source type: {src}. "
                    "Use URL, .txt file, or PDF directory."
                )
        logger.info("Total loaded documents: %d", len(docs))
        return docs
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks
        
        Args:
            documents: List of documents to split
            
        Returns:
            List of split documents
        """
        logger.debug("Splitting %d documents into chunks", len(documents))
        chunks = self.splitter.split_documents(documents)
        logger.info("Split into %d chunks", len(chunks))
        return chunks
    
    def process_urls(self, urls: List[str]) -> List[Document]:
        """
        Complete pipeline to load and split documents
        
        Args:
            urls: List of URLs to process
            
        Returns:
            List of processed document chunks
        """
        logger.info("Processing %d URLs...", len(urls))
        docs = self.load_documents(urls)
        return self.split_documents(docs)