"""
Embedding Encoder Module
Handles text embedding generation using sentence-transformers
"""

from typing import List, Union
import logging
import numpy as np

from sentence_transformers import SentenceTransformer
from langchain.schema import Document
import torch

logger = logging.getLogger(__name__)


class EmbeddingEncoder:
    """Generates embeddings for text using sentence transformers"""
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
        normalize_embeddings: bool = True,
        batch_size: int = 32,
    ):
        """
        Initialize embedding encoder
        
        Args:
            model_name: Name of the sentence-transformer model
            device: Device to use (cpu or cuda)
            normalize_embeddings: Whether to normalize embeddings
            batch_size: Batch size for encoding
        """
        self.model_name = model_name
        self.device = device
        self.normalize_embeddings = normalize_embeddings
        self.batch_size = batch_size
        
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name, device=device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        logger.info(f"Model loaded. Embedding dimension: {self.embedding_dim}")
    
    def encode_texts(
        self,
        texts: List[str],
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Encode a list of texts into embeddings
        
        Args:
            texts: List of text strings
            show_progress: Whether to show progress bar
            
        Returns:
            Numpy array of embeddings (n_texts, embedding_dim)
        """
        if not texts:
            return np.array([])
        
        logger.info(f"Encoding {len(texts)} texts...")
        
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
        )
        
        logger.info(f"Encoded {len(texts)} texts to shape {embeddings.shape}")
        return embeddings
    
    def encode_documents(
        self,
        documents: List[Document],
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Encode a list of Document objects
        
        Args:
            documents: List of Document objects
            show_progress: Whether to show progress bar
            
        Returns:
            Numpy array of embeddings
        """
        texts = [doc.page_content for doc in documents]
        return self.encode_texts(texts, show_progress=show_progress)
    
    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a single query
        
        Args:
            query: Query string
            
        Returns:
            1D numpy array of the query embedding
        """
        embedding = self.model.encode(
            [query],
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
        )
        return embedding[0]
    
    def encode_batch(
        self,
        texts: List[str],
        batch_size: int = None,
    ) -> List[np.ndarray]:
        """
        Encode texts in batches (for memory efficiency)
        
        Args:
            texts: List of texts
            batch_size: Batch size (uses default if None)
            
        Returns:
            List of embedding arrays
        """
        if batch_size is None:
            batch_size = self.batch_size
        
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings = self.encode_texts(batch, show_progress=False)
            all_embeddings.append(embeddings)
            
            if i % (batch_size * 10) == 0:
                logger.info(f"Processed {i}/{len(texts)} texts")
        
        return np.vstack(all_embeddings) if all_embeddings else np.array([])
    
    def get_embedding_dim(self) -> int:
        """Get the dimensionality of embeddings"""
        return self.embedding_dim
    
    def compute_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
    ) -> float:
        """
        Compute cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Cosine similarity score
        """
        return np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        )
    
    def compute_similarities(
        self,
        query_embedding: np.ndarray,
        corpus_embeddings: np.ndarray,
    ) -> np.ndarray:
        """
        Compute similarities between a query and corpus
        
        Args:
            query_embedding: Query embedding (1D array)
            corpus_embeddings: Corpus embeddings (2D array)
            
        Returns:
            Array of similarity scores
        """
        if self.normalize_embeddings:
            # If normalized, dot product = cosine similarity
            return np.dot(corpus_embeddings, query_embedding)
        else:
            # Compute cosine similarity
            norms = np.linalg.norm(corpus_embeddings, axis=1) * np.linalg.norm(query_embedding)
            return np.dot(corpus_embeddings, query_embedding) / norms


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    encoder = EmbeddingEncoder()
    
    texts = [
        "This is a sample sentence.",
        "Here is another example text.",
        "Machine learning is fascinating.",
    ]
    
    embeddings = encoder.encode_texts(texts)
    print(f"Embeddings shape: {embeddings.shape}")
    
    query = "Tell me about machine learning"
    query_emb = encoder.encode_query(query)
    print(f"Query embedding shape: {query_emb.shape}")
    
    similarities = encoder.compute_similarities(query_emb, embeddings)
    print(f"Similarities: {similarities}")