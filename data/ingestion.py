"""
Document Ingestion Module
Handles loading various document formats (PDF, HTML, TXT, etc.)
"""

import os
from typing import List, Dict, Any
from pathlib import Path
import logging

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredHTMLLoader,
    DirectoryLoader,
)
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class DocumentIngestion:
    """Handles document loading from various sources and formats"""
    
    def __init__(self, data_dir: str = "./data/documents"):
        """
        Initialize document ingestion
        
        Args:
            data_dir: Directory containing documents to ingest
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Supported file extensions
        self.loaders = {
            '.pdf': PyPDFLoader,
            '.txt': TextLoader,
            '.html': UnstructuredHTMLLoader,
            '.htm': UnstructuredHTMLLoader,
        }
    
    def load_document(self, file_path: str) -> List[Document]:
        """
        Load a single document
        
        Args:
            file_path: Path to the document
            
        Returns:
            List of Document objects
        """
        file_path = Path(file_path)
        extension = file_path.suffix.lower()
        
        if extension not in self.loaders:
            logger.warning(f"Unsupported file type: {extension}")
            return []
        
        try:
            loader_class = self.loaders[extension]
            loader = loader_class(str(file_path))
            documents = loader.load()
            
            # Add metadata
            for doc in documents:
                doc.metadata['source'] = str(file_path)
                doc.metadata['file_type'] = extension
            
            logger.info(f"Loaded {len(documents)} documents from {file_path}")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading {file_path}: {str(e)}")
            return []
    
    def load_directory(self, directory: str = None) -> List[Document]:
        """
        Load all supported documents from a directory
        
        Args:
            directory: Directory path (defaults to self.data_dir)
            
        Returns:
            List of all loaded documents
        """
        if directory is None:
            directory = self.data_dir
        
        directory = Path(directory)
        
        if not directory.exists():
            logger.warning(f"Directory does not exist: {directory}")
            return []
        
        all_documents = []
        
        # Load files for each supported extension
        for extension, loader_class in self.loaders.items():
            try:
                glob_pattern = f"**/*{extension}"
                loader = DirectoryLoader(
                    str(directory),
                    glob=glob_pattern,
                    loader_cls=loader_class,
                    show_progress=True,
                    use_multithreading=True,
                )
                documents = loader.load()
                all_documents.extend(documents)
                logger.info(f"Loaded {len(documents)} {extension} files")
            except Exception as e:
                logger.error(f"Error loading {extension} files: {str(e)}")
        
        logger.info(f"Total documents loaded: {len(all_documents)}")
        return all_documents
    
    def load_from_text(self, text: str, metadata: Dict[str, Any] = None) -> Document:
        """
        Create a document from raw text
        
        Args:
            text: Raw text content
            metadata: Optional metadata dictionary
            
        Returns:
            Document object
        """
        if metadata is None:
            metadata = {}
        
        metadata['source'] = 'text_input'
        return Document(page_content=text, metadata=metadata)
    
    def preprocess_documents(self, documents: List[Document]) -> List[Document]:
        """
        Preprocess documents (cleaning, normalization)
        
        Args:
            documents: List of documents to preprocess
            
        Returns:
            Preprocessed documents
        """
        processed_docs = []
        
        for doc in documents:
            # Clean text
            text = doc.page_content
            text = text.strip()
            text = ' '.join(text.split())  # Normalize whitespace
            
            if len(text) > 0:  # Only keep non-empty documents
                doc.page_content = text
                processed_docs.append(doc)
        
        logger.info(f"Preprocessed {len(processed_docs)} documents")
        return processed_docs


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    ingestion = DocumentIngestion()
    docs = ingestion.load_directory()
    docs = ingestion.preprocess_documents(docs)
    
    print(f"Loaded {len(docs)} documents")
    if docs:
        print(f"Sample document: {docs[0].page_content[:200]}...")