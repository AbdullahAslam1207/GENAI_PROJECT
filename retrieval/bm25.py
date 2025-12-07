"""
BM25 Retrieval Module
Implements sparse lexical retrieval using BM25 algorithm
"""

from typing import List, Tuple, Dict, Any
import logging
import pickle
from pathlib import Path

from rank_bm25 import BM25Okapi
from langchain.schema import Document
import numpy as np

logger = logging.getLogger(__name__)


class BM25Retriever:
    """BM25-based lexical retrieval"""
    
    def __init__(self, documents: List[Document] = None, top_k: int = 20):
        """
        Initialize BM25 retriever
        
        Args:
            documents: Optional list of documents to index
            top_k: Number of results to retrieve
        """
        self.top_k = top_k
        self.documents = []
        self.bm25 = None
        self.tokenized_corpus = []
        
        if documents:
            self.build_index(documents)
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Simple tokenization (can be improved with spaCy/nltk)
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of tokens
        """
        # Simple whitespace tokenization + lowercasing
        return text.lower().split()
    
    def build_index(self, documents: List[Document]):
        """
        Build BM25 index from documents
        
        Args:
            documents: List of documents to index
        """
        logger.info(f"Building BM25 index for {len(documents)} documents...")
        
        self.documents = documents
        
        # Tokenize all documents
        self.tokenized_corpus = [
            self._tokenize(doc.page_content)
            for doc in documents
        ]
        
        # Build BM25 index
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        
        logger.info("BM25 index built successfully")
    
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
            return_scores: Whether to return scores
            
        Returns:
            List of (Document, score) tuples if return_scores=True,
            otherwise list of Documents
        """
        if self.bm25 is None:
            logger.error("Index not built. Call build_index() first.")
            return []
        
        if k is None:
            k = self.top_k
        
        # Tokenize query
        tokenized_query = self._tokenize(query)
        
        # Get BM25 scores
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:k]
        
        # Get documents and scores
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include non-zero scores
                doc = self.documents[idx]
                score = float(scores[idx])
                results.append((doc, score) if return_scores else doc)
        
        logger.info(f"Retrieved {len(results)} documents with BM25")
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
        Add new documents to the index (requires rebuilding)
        
        Args:
            documents: Documents to add
        """
        if self.documents:
            self.documents.extend(documents)
            self.build_index(self.documents)
        else:
            self.build_index(documents)
        
        logger.info(f"Added {len(documents)} documents to BM25 index")
    
    def save(self, save_dir: str):
        """
        Save BM25 index to disk
        
        Args:
            save_dir: Directory to save files
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save documents
        docs_path = save_dir / "bm25_documents.pkl"
        with open(docs_path, 'wb') as f:
            pickle.dump(self.documents, f)
        
        # Save tokenized corpus
        corpus_path = save_dir / "bm25_corpus.pkl"
        with open(corpus_path, 'wb') as f:
            pickle.dump(self.tokenized_corpus, f)
        
        # Save BM25 object
        bm25_path = save_dir / "bm25_index.pkl"
        with open(bm25_path, 'wb') as f:
            pickle.dump(self.bm25, f)
        
        logger.info(f"Saved BM25 index to {save_dir}")
    
    def load(self, save_dir: str):
        """
        Load BM25 index from disk
        
        Args:
            save_dir: Directory containing saved files
        """
        save_dir = Path(save_dir)
        
        # Load documents
        docs_path = save_dir / "bm25_documents.pkl"
        with open(docs_path, 'rb') as f:
            self.documents = pickle.load(f)
        
        # Load tokenized corpus
        corpus_path = save_dir / "bm25_corpus.pkl"
        with open(corpus_path, 'rb') as f:
            self.tokenized_corpus = pickle.load(f)
        
        # Load BM25 object
        bm25_path = save_dir / "bm25_index.pkl"
        with open(bm25_path, 'rb') as f:
            self.bm25 = pickle.load(f)
        
        logger.info(f"Loaded BM25 index from {save_dir}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the index"""
        return {
            'n_documents': len(self.documents),
            'avg_doc_length': np.mean([len(tokens) for tokens in self.tokenized_corpus]) if self.tokenized_corpus else 0,
        }


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create sample documents
    documents = [
        Document(page_content="Machine learning is a subset of artificial intelligence.", metadata={'id': 0}),
        Document(page_content="Natural language processing deals with text and language.", metadata={'id': 1}),
        Document(page_content="Deep learning uses neural networks with multiple layers.", metadata={'id': 2}),
        Document(page_content="Computer vision enables machines to interpret images.", metadata={'id': 3}),
    ]
    
    # Build retriever
    retriever = BM25Retriever(documents)
    
    # Search
    query = "What is machine learning?"
    results = retriever.retrieve(query, k=2)
    
    print(f"\nQuery: {query}")
    print(f"Results:")
    for doc, score in results:
        print(f"  Score: {score:.4f} - {doc.page_content}")