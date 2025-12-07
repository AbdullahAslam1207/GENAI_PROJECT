"""
Text Chunking Module
Implements various chunking strategies for document splitting
"""

from typing import List, Dict, Any
import logging

from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter,
)
from langchain.schema import Document

logger = logging.getLogger(__name__)


class DocumentChunker:
    """Handles document chunking with various strategies"""
    
    def __init__(
        self,
        strategy: str = "recursive",
        chunk_size: int = 512,
        chunk_overlap: int = 128,
    ):
        """
        Initialize document chunker
        
        Args:
            strategy: Chunking strategy (recursive, character, token, sliding_window)
            chunk_size: Size of each chunk
            chunk_overlap: Overlap between chunks
        """
        self.strategy = strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        self.splitter = self._create_splitter()
    
    def _create_splitter(self):
        """Create the appropriate text splitter based on strategy"""
        
        if self.strategy == "recursive":
            return RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""],
            )
        
        elif self.strategy == "character":
            return CharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separator="\n",
            )
        
        elif self.strategy == "token":
            return TokenTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
        
        elif self.strategy == "sliding_window":
            return RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
            )
        
        else:
            logger.warning(f"Unknown strategy {self.strategy}, using recursive")
            return RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Chunk a list of documents
        
        Args:
            documents: List of documents to chunk
            
        Returns:
            List of chunked documents
        """
        chunked_docs = []
        
        for doc in documents:
            try:
                chunks = self.splitter.split_documents([doc])
                
                # Add chunk metadata
                for i, chunk in enumerate(chunks):
                    chunk.metadata['chunk_id'] = i
                    chunk.metadata['total_chunks'] = len(chunks)
                    chunk.metadata['chunk_strategy'] = self.strategy
                
                chunked_docs.extend(chunks)
                
            except Exception as e:
                logger.error(f"Error chunking document: {str(e)}")
                continue
        
        logger.info(f"Created {len(chunked_docs)} chunks from {len(documents)} documents")
        return chunked_docs
    
    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Document]:
        """
        Chunk raw text
        
        Args:
            text: Text to chunk
            metadata: Optional metadata to attach
            
        Returns:
            List of document chunks
        """
        if metadata is None:
            metadata = {}
        
        texts = self.splitter.split_text(text)
        
        documents = []
        for i, chunk_text in enumerate(texts):
            doc_metadata = metadata.copy()
            doc_metadata['chunk_id'] = i
            doc_metadata['total_chunks'] = len(texts)
            
            documents.append(Document(page_content=chunk_text, metadata=doc_metadata))
        
        return documents
    
    def adaptive_chunk(
        self,
        documents: List[Document],
        min_chunk_size: int = 256,
        max_chunk_size: int = 1024,
    ) -> List[Document]:
        """
        Adaptive chunking based on document structure
        
        Args:
            documents: Documents to chunk
            min_chunk_size: Minimum chunk size
            max_chunk_size: Maximum chunk size
            
        Returns:
            Adaptively chunked documents
        """
        chunked_docs = []
        
        for doc in documents:
            text = doc.page_content
            
            # Split on paragraph boundaries first
            paragraphs = text.split('\n\n')
            
            current_chunk = ""
            chunk_id = 0
            
            for para in paragraphs:
                para = para.strip()
                
                if len(current_chunk) + len(para) < max_chunk_size:
                    current_chunk += para + "\n\n"
                else:
                    if len(current_chunk) >= min_chunk_size:
                        # Save current chunk
                        metadata = doc.metadata.copy()
                        metadata['chunk_id'] = chunk_id
                        metadata['chunk_strategy'] = 'adaptive'
                        
                        chunked_docs.append(
                            Document(page_content=current_chunk.strip(), metadata=metadata)
                        )
                        chunk_id += 1
                    
                    current_chunk = para + "\n\n"
            
            # Add remaining chunk
            if len(current_chunk) >= min_chunk_size:
                metadata = doc.metadata.copy()
                metadata['chunk_id'] = chunk_id
                metadata['chunk_strategy'] = 'adaptive'
                
                chunked_docs.append(
                    Document(page_content=current_chunk.strip(), metadata=metadata)
                )
        
        logger.info(f"Adaptive chunking created {len(chunked_docs)} chunks")
        return chunked_docs


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    sample_text = """
    This is a sample document that needs to be chunked.
    It has multiple paragraphs and sections.
    
    This is the second paragraph. It contains more information.
    The chunker will split this text intelligently.
    
    Here's a third paragraph with additional content.
    """ * 10
    
    chunker = DocumentChunker(strategy="recursive", chunk_size=200, chunk_overlap=50)
    chunks = chunker.chunk_text(sample_text)
    
    print(f"Created {len(chunks)} chunks")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\nChunk {i}: {chunk.page_content[:100]}...")