"""
Hybrid Retrieval Module
Combines dense and sparse retrieval for improved results
"""

from typing import List, Tuple, Dict, Any
import logging

import numpy as np
from langchain.schema import Document

from retrieval.dense import DenseRetriever
from retrieval.bm25 import BM25Retriever

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Hybrid retrieval combining dense embeddings and BM25"""
    
    def __init__(
        self,
        dense_retriever: DenseRetriever,
        bm25_retriever: BM25Retriever,
        alpha: float = 0.7,
        top_k: int = 20,
    ):
        """
        Initialize hybrid retriever
        
        Args:
            dense_retriever: Dense retriever instance
            bm25_retriever: BM25 retriever instance
            alpha: Weight for dense scores (1-alpha for BM25)
            top_k: Number of results to retrieve
        """
        self.dense_retriever = dense_retriever
        self.bm25_retriever = bm25_retriever
        self.alpha = alpha
        self.top_k = top_k
    
    def _normalize_scores(self, scores: List[float]) -> np.ndarray:
        """
        Normalize scores to [0, 1] range using min-max scaling
        
        Args:
            scores: List of scores
            
        Returns:
            Normalized scores
        """
        scores = np.array(scores)
        
        if len(scores) == 0:
            return scores
        
        min_score = scores.min()
        max_score = scores.max()
        
        if max_score == min_score:
            return np.ones_like(scores)
        
        return (scores - min_score) / (max_score - min_score)
    
    def retrieve(
        self,
        query: str,
        k: int = None,
        return_scores: bool = True,
    ) -> List[Tuple[Document, float]]:
        """
        Retrieve documents using hybrid approach
        
        Args:
            query: Query string
            k: Number of results (uses default if None)
            return_scores: Whether to return scores
            
        Returns:
            List of (Document, score) tuples or Documents
        """
        if k is None:
            k = self.top_k
        
        # Retrieve from both systems (get more than k for fusion)
        retrieve_k = min(k * 3, 100)  # Get 3x for better fusion
        
        # Dense retrieval
        dense_results = self.dense_retriever.retrieve(
            query,
            k=retrieve_k,
            return_scores=True
        )
        
        # BM25 retrieval
        bm25_results = self.bm25_retriever.retrieve(
            query,
            k=retrieve_k,
            return_scores=True
        )
        
        # Combine results
        combined_scores = {}
        
        # Add dense results
        if dense_results:
            dense_docs, dense_scores = zip(*dense_results)
            normalized_dense = self._normalize_scores(dense_scores)
            
            for doc, score in zip(dense_docs, normalized_dense):
                doc_key = doc.page_content  # Use content as key
                combined_scores[doc_key] = {
                    'document': doc,
                    'dense_score': score,
                    'bm25_score': 0.0
                }
        
        # Add BM25 results
        if bm25_results:
            bm25_docs, bm25_scores = zip(*bm25_results)
            normalized_bm25 = self._normalize_scores(bm25_scores)
            
            for doc, score in zip(bm25_docs, normalized_bm25):
                doc_key = doc.page_content
                
                if doc_key in combined_scores:
                    combined_scores[doc_key]['bm25_score'] = score
                else:
                    combined_scores[doc_key] = {
                        'document': doc,
                        'dense_score': 0.0,
                        'bm25_score': score
                    }
        
        # Calculate hybrid scores
        hybrid_results = []
        for doc_key, scores in combined_scores.items():
            hybrid_score = (
                self.alpha * scores['dense_score'] +
                (1 - self.alpha) * scores['bm25_score']
            )
            hybrid_results.append((scores['document'], hybrid_score))
        
        # Sort by hybrid score
        hybrid_results.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k
        results = hybrid_results[:k]
        
        if not return_scores:
            results = [doc for doc, _ in results]
        
        logger.info(f"Retrieved {len(results)} documents using hybrid retrieval")
        return results
    
    def retrieve_with_metadata(
        self,
        query: str,
        k: int = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents with detailed metadata including separate scores
        
        Args:
            query: Query string
            k: Number of results
            
        Returns:
            List of dictionaries with document, scores, and metadata
        """
        if k is None:
            k = self.top_k
        
        retrieve_k = min(k * 3, 100)
        
        # Get results from both retrievers
        dense_results = self.dense_retriever.retrieve(query, k=retrieve_k, return_scores=True)
        bm25_results = self.bm25_retriever.retrieve(query, k=retrieve_k, return_scores=True)
        
        # Create mapping
        score_map = {}
        
        # Dense results
        if dense_results:
            dense_docs, dense_scores = zip(*dense_results)
            normalized_dense = self._normalize_scores(dense_scores)
            
            for doc, score in zip(dense_docs, normalized_dense):
                doc_key = doc.page_content
                score_map[doc_key] = {
                    'document': doc,
                    'dense_score': float(score),
                    'bm25_score': 0.0
                }
        
        # BM25 results
        if bm25_results:
            bm25_docs, bm25_scores = zip(*bm25_results)
            normalized_bm25 = self._normalize_scores(bm25_scores)
            
            for doc, score in zip(bm25_docs, normalized_bm25):
                doc_key = doc.page_content
                
                if doc_key in score_map:
                    score_map[doc_key]['bm25_score'] = float(score)
                else:
                    score_map[doc_key] = {
                        'document': doc,
                        'dense_score': 0.0,
                        'bm25_score': float(score)
                    }
        
        # Calculate hybrid scores and format results
        formatted_results = []
        for doc_key, scores in score_map.items():
            hybrid_score = (
                self.alpha * scores['dense_score'] +
                (1 - self.alpha) * scores['bm25_score']
            )
            
            formatted_results.append({
                'content': scores['document'].page_content,
                'hybrid_score': hybrid_score,
                'dense_score': scores['dense_score'],
                'bm25_score': scores['bm25_score'],
                'metadata': scores['document'].metadata,
            })
        
        # Sort and return top-k
        formatted_results.sort(key=lambda x: x['hybrid_score'], reverse=True)
        return formatted_results[:k]
    
    def set_alpha(self, alpha: float):
        """
        Update the alpha parameter
        
        Args:
            alpha: New alpha value (0 to 1)
        """
        if not 0 <= alpha <= 1:
            raise ValueError("Alpha must be between 0 and 1")
        
        self.alpha = alpha
        logger.info(f"Updated alpha to {alpha}")


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    from embeddings.encoder import EmbeddingEncoder
    
    # Create sample documents
    documents = [
        Document(page_content="Machine learning is a subset of artificial intelligence.", metadata={'id': 0}),
        Document(page_content="Natural language processing deals with text and language.", metadata={'id': 1}),
        Document(page_content="Deep learning uses neural networks with multiple layers.", metadata={'id': 2}),
        Document(page_content="Computer vision enables machines to interpret images.", metadata={'id': 3}),
    ]
    
    # Create retrievers
    encoder = EmbeddingEncoder()
    dense_retriever = DenseRetriever(encoder)
    dense_retriever.build_index(documents)
    
    bm25_retriever = BM25Retriever(documents)
    
    # Create hybrid retriever
    hybrid = HybridRetriever(dense_retriever, bm25_retriever, alpha=0.7)
    
    # Search
    query = "What is machine learning?"
    results = hybrid.retrieve(query, k=2)
    
    print(f"\nQuery: {query}")
    print(f"Results:")
    for doc, score in results:
        print(f"  Score: {score:.4f} - {doc.page_content}")