"""
Dense Retrieval Module
Implements dense vector-based retrieval using embeddings
"""

from typing import List, Tuple, Dict, Any
import logging

import numpy as np
from langchain.schema import Document

from embeddings.encoder import EmbeddingEncoder
from indexing.faiss_index import FAISSIndex

logger = logging.getLogger(__name__)


class DenseRetriever:
    """Dense retrieval using vector embeddings and FAISS"""
    
    def __init__(
        self,
        encoder: EmbeddingEncoder,
        index: FAISSIndex = None,
        top_k: int = 20,
    ):
        """
        Initialize dense retriever
        
        Args:
            encoder: Embedding encoder
            index: FAISS index (optional, can be built later)
            top_k: Number of results to retrieve
        """
        self.encoder = encoder
        self.index = index
        self.top_k = top_k
    
    def build_index(
        self,
        documents: List[Document],
        index_type: str = "Flat",
        n_clusters: int = 100,
        show_progress: bool = True,
    ):
        """
        Build the retrieval index from documents
        
        Args:
            documents: List of documents to index
            index_type: Type of FAISS index
            n_clusters: Number of clusters for IVF
            show_progress: Show progress during encoding
        """
        logger.info(f"Building index for {len(documents)} documents...")
        
        # Generate embeddings
        embeddings = self.encoder.encode_documents(documents, show_progress=show_progress)
        
        # Create index
        self.index = FAISSIndex(
            embedding_dim=self.encoder.get_embedding_dim(),
            index_type=index_type,
            n_clusters=n_clusters,
        )
        
        # Build index
        self.index.build(embeddings, documents)
        
        logger.info("Index built successfully")
    
    def retrieve(
        self,
        query: str,
        k: int = None,
        return_scores: bool = True,
    ) -> List[Tuple[Document, float]]:
        """
        Retrieve relevant documents for a query
        
        Args:
            query: Query string
            k: Number of results (uses default if None)
            return_scores: Whether to return similarity scores
            
        Returns:
            List of (Document, score) tuples if return_scores=True,
            otherwise list of Documents
        """
        if self.index is None:
            logger.error("Index not built. Call build_index() first.")
            return []
        
        if k is None:
            k = self.top_k
        
        # Encode query
        query_embedding = self.encoder.encode_query(query)
        
        # Search
        documents, distances, indices = self.index.search(
            query_embedding,
            k=k,
        )
        
        if not documents:
            return []
        
        # Convert L2 distances to similarity scores (inverse)
        # Lower distance = higher similarity
        scores = [1.0 / (1.0 + d) for d in distances]
        
        if return_scores:
            results = list(zip(documents, scores))
        else:
            results = documents
        
        logger.info(f"Retrieved {len(documents)} documents")
        return results
    
    def retrieve_with_metadata(
        self,
        query: str,
        k: int = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents with detailed metadata
        
        Args:
            query: Query string
            k: Number of results
            
        Returns:
            List of dictionaries with document, score, and metadata
        """
        results = self.retrieve(query, k=k, return_scores=True)
        
        formatted_results = []
        for doc, score in results:
            formatted_results.append({
                'content': doc.page_content,
                'score': score,
                'metadata': doc.metadata,
            })
        
        return formatted_results
    
    def add_documents(self, documents: List[Document]):
        """
        Add new documents to the index
        
        Args:
            documents: Documents to add
        """
        if self.index is None:
            logger.error("Index not built. Call build_index() first.")
            return
        
        # Generate embeddings
        embeddings = self.encoder.encode_documents(documents, show_progress=False)
        
        # Add to index
        self.index.add_documents(embeddings, documents)
        
        logger.info(f"Added {len(documents)} documents to index")
    
    def save_index(self, save_dir: str):
        """Save the index to disk"""
        if self.index:
            self.index.save(save_dir)
        else:
            logger.warning("No index to save")
    
    def load_index(self, save_dir: str):
        """Load the index from disk"""
        self.index = FAISSIndex(embedding_dim=self.encoder.get_embedding_dim())
        self.index.load(save_dir)
        logger.info("Index loaded successfully")
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the index"""
        if self.index:
            return self.index.get_stats()
        return {}


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create encoder
    encoder = EmbeddingEncoder()
    
    # Create sample documents
    documents = [
        Document(page_content="Machine learning is a subset of artificial intelligence.", metadata={'id': 0}),
        Document(page_content="Natural language processing deals with text and language.", metadata={'id': 1}),
        Document(page_content="Deep learning uses neural networks with multiple layers.", metadata={'id': 2}),
        Document(page_content="Computer vision enables machines to interpret images.", metadata={'id': 3}),
    ]
    
    # Build retriever
    retriever = DenseRetriever(encoder)
    retriever.build_index(documents, index_type="Flat")
    
    # Search
    query = "What is machine learning?"
    results = retriever.retrieve(query, k=2)
    
    print(f"\nQuery: {query}")
    print(f"Results:")
    for doc, score in results:
        print(f"  Score: {score:.4f} - {doc.page_content}")