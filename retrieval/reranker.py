"""
Reranker Module
Implements cross-encoder based reranking for improved precision
"""

from typing import List, Tuple, Dict, Any
import logging

from sentence_transformers import CrossEncoder
from langchain.schema import Document
import numpy as np

logger = logging.getLogger(__name__)


class Reranker:
    """Cross-encoder based reranker for improving retrieval precision"""
    
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        batch_size: int = 16,
    ):
        """
        Initialize reranker
        
        Args:
            model_name: Name of the cross-encoder model
            batch_size: Batch size for reranking
        """
        self.model_name = model_name
        self.batch_size = batch_size
        
        logger.info(f"Loading cross-encoder model: {model_name}")
        self.model = CrossEncoder(model_name)
        logger.info("Reranker model loaded")
    
    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_n: int = 5,
        return_scores: bool = True,
    ) -> List[Tuple[Document, float]]:
        """
        Rerank documents using cross-encoder
        
        Args:
            query: Query string
            documents: List of documents to rerank
            top_n: Number of top results to return
            return_scores: Whether to return scores
            
        Returns:
            Reranked list of (Document, score) tuples or Documents
        """
        if not documents:
            return []
        
        # Prepare query-document pairs
        pairs = [[query, doc.page_content] for doc in documents]
        
        # Get cross-encoder scores
        scores = self.model.predict(pairs, batch_size=self.batch_size)
        
        # Create (document, score) tuples
        doc_scores = list(zip(documents, scores))
        
        # Sort by score (descending)
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-n
        results = doc_scores[:top_n]
        
        if not return_scores:
            results = [doc for doc, _ in results]
        
        logger.info(f"Reranked to top {len(results)} documents")
        return results
    
    def rerank_with_metadata(
        self,
        query: str,
        documents: List[Document],
        top_n: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents and return with detailed metadata
        
        Args:
            query: Query string
            documents: List of documents to rerank
            top_n: Number of top results
            
        Returns:
            List of dictionaries with document, score, and metadata
        """
        results = self.rerank(query, documents, top_n=top_n, return_scores=True)
        
        formatted_results = []
        for doc, score in results:
            formatted_results.append({
                'content': doc.page_content,
                'rerank_score': float(score),
                'metadata': doc.metadata,
            })
        
        return formatted_results
    
    def score_pairs(
        self,
        query: str,
        documents: List[Document],
    ) -> np.ndarray:
        """
        Get relevance scores for all documents without sorting
        
        Args:
            query: Query string
            documents: List of documents
            
        Returns:
            Array of relevance scores
        """
        if not documents:
            return np.array([])
        
        pairs = [[query, doc.page_content] for doc in documents]
        scores = self.model.predict(pairs, batch_size=self.batch_size)
        
        return scores


class RetrievalWithReranking:
    """Wrapper that combines retrieval with reranking"""
    
    def __init__(
        self,
        retriever,  # Any retriever (Dense, BM25, or Hybrid)
        reranker: Reranker,
        retrieve_k: int = 20,
        rerank_top_n: int = 5,
    ):
        """
        Initialize retrieval with reranking
        
        Args:
            retriever: Base retriever instance
            reranker: Reranker instance
            retrieve_k: Number of documents to retrieve initially
            rerank_top_n: Number of documents after reranking
        """
        self.retriever = retriever
        self.reranker = reranker
        self.retrieve_k = retrieve_k
        self.rerank_top_n = rerank_top_n
    
    def retrieve(
        self,
        query: str,
        return_scores: bool = True,
    ) -> List[Tuple[Document, float]]:
        """
        Retrieve and rerank documents
        
        Args:
            query: Query string
            return_scores: Whether to return scores
            
        Returns:
            List of (Document, score) tuples or Documents
        """
        # Initial retrieval
        initial_results = self.retriever.retrieve(
            query,
            k=self.retrieve_k,
            return_scores=False
        )
        
        if isinstance(initial_results[0], tuple):
            initial_docs = [doc for doc, _ in initial_results]
        else:
            initial_docs = initial_results
        
        # Rerank
        reranked_results = self.reranker.rerank(
            query,
            initial_docs,
            top_n=self.rerank_top_n,
            return_scores=return_scores
        )
        
        logger.info(f"Retrieved {self.retrieve_k} docs, reranked to top {len(reranked_results)}")
        return reranked_results
    
    def retrieve_with_metadata(
        self,
        query: str,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve and rerank with detailed metadata
        
        Args:
            query: Query string
            
        Returns:
            List of dictionaries with document, scores, and metadata
        """
        # Initial retrieval
        initial_results = self.retriever.retrieve(
            query,
            k=self.retrieve_k,
            return_scores=True
        )
        
        if not initial_results:
            return []
        
        initial_docs = [doc for doc, _ in initial_results]
        initial_scores = [score for _, score in initial_results]
        
        # Rerank
        reranked_results = self.reranker.rerank(
            query,
            initial_docs,
            top_n=self.rerank_top_n,
            return_scores=True
        )
        
        # Format with both retrieval and rerank scores
        formatted_results = []
        for doc, rerank_score in reranked_results:
            # Find original retrieval score
            original_idx = initial_docs.index(doc)
            retrieval_score = initial_scores[original_idx]
            
            formatted_results.append({
                'content': doc.page_content,
                'rerank_score': float(rerank_score),
                'retrieval_score': float(retrieval_score),
                'metadata': doc.metadata,
            })
        
        return formatted_results


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create sample documents
    documents = [
        Document(page_content="Machine learning is a subset of artificial intelligence.", metadata={'id': 0}),
        Document(page_content="Natural language processing deals with text and language.", metadata={'id': 1}),
        Document(page_content="Deep learning uses neural networks with multiple layers.", metadata={'id': 2}),
        Document(page_content="Computer vision enables machines to interpret images.", metadata={'id': 3}),
        Document(page_content="AI is transforming various industries.", metadata={'id': 4}),
    ]
    
    # Create reranker
    reranker = Reranker()
    
    # Rerank
    query = "What is machine learning?"
    results = reranker.rerank(query, documents, top_n=3)
    
    print(f"\nQuery: {query}")
    print(f"Reranked Results:")
    for doc, score in results:
        print(f"  Score: {score:.4f} - {doc.page_content}")