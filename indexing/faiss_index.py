"""
FAISS Index Module
Handles FAISS index creation and similarity search
"""

import os
from typing import List, Tuple, Optional
import logging
import pickle
from pathlib import Path

import numpy as np
import faiss
from langchain.schema import Document

logger = logging.getLogger(__name__)


class FAISSIndex:
    """FAISS-based vector index for similarity search"""
    
    def __init__(
        self,
        embedding_dim: int,
        index_type: str = "Flat",
        n_clusters: int = 100,
        use_gpu: bool = False,
    ):
        """
        Initialize FAISS index
        
        Args:
            embedding_dim: Dimension of embeddings
            index_type: Type of index (Flat, IVFFlat, IVFPQ)
            n_clusters: Number of clusters for IVF indices
            use_gpu: Whether to use GPU (not supported in this config)
        """
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.n_clusters = n_clusters
        self.use_gpu = use_gpu
        
        self.index = None
        self.documents = []
        self.embeddings = None
        
        self._create_index()
    
    def _create_index(self):
        """Create the FAISS index based on configuration"""
        
        if self.index_type == "Flat":
            # Exact search using L2 distance
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            logger.info("Created Flat index (exact search)")
        
        elif self.index_type == "IVFFlat":
            # Approximate search with inverted file
            quantizer = faiss.IndexFlatL2(self.embedding_dim)
            self.index = faiss.IndexIVFFlat(
                quantizer,
                self.embedding_dim,
                self.n_clusters,
                faiss.METRIC_L2
            )
            logger.info(f"Created IVFFlat index with {self.n_clusters} clusters")
        
        elif self.index_type == "IVFPQ":
            # Approximate search with product quantization
            quantizer = faiss.IndexFlatL2(self.embedding_dim)
            m = 8  # Number of sub-quantizers
            bits = 8  # Bits per sub-quantizer
            
            self.index = faiss.IndexIVFPQ(
                quantizer,
                self.embedding_dim,
                self.n_clusters,
                m,
                bits
            )
            logger.info(f"Created IVFPQ index with {self.n_clusters} clusters")
        
        else:
            logger.warning(f"Unknown index type {self.index_type}, using Flat")
            self.index = faiss.IndexFlatL2(self.embedding_dim)
    
    def build(
        self,
        embeddings: np.ndarray,
        documents: List[Document],
    ):
        """
        Build the index with embeddings and documents
        
        Args:
            embeddings: Array of embeddings (n_docs, embedding_dim)
            documents: List of corresponding documents
        """
        if len(embeddings) != len(documents):
            raise ValueError("Number of embeddings must match number of documents")
        
        # Convert to float32 (required by FAISS)
        embeddings = embeddings.astype('float32')
        
        # Store data
        self.embeddings = embeddings
        self.documents = documents
        
        # Train index if necessary
        if self.index_type in ["IVFFlat", "IVFPQ"]:
            logger.info(f"Training index on {len(embeddings)} vectors...")
            self.index.train(embeddings)
        
        # Add vectors to index
        logger.info(f"Adding {len(embeddings)} vectors to index...")
        self.index.add(embeddings)
        
        logger.info(f"Index built successfully. Total vectors: {self.index.ntotal}")
    
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        nprobe: int = 10,
    ) -> Tuple[List[Document], List[float], List[int]]:
        """
        Search for similar documents
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            nprobe: Number of clusters to search (for IVF indices)
            
        Returns:
            Tuple of (documents, distances, indices)
        """
        if self.index is None or self.index.ntotal == 0:
            logger.warning("Index is empty")
            return [], [], []
        
        # Set nprobe for IVF indices
        if self.index_type in ["IVFFlat", "IVFPQ"]:
            self.index.nprobe = nprobe
        
        # Prepare query
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        
        # Search
        distances, indices = self.index.search(query_embedding, k)
        
        # Get documents
        results = []
        result_distances = []
        result_indices = []
        
        for i, idx in enumerate(indices[0]):
            if idx >= 0 and idx < len(self.documents):
                results.append(self.documents[idx])
                result_distances.append(float(distances[0][i]))
                result_indices.append(int(idx))
        
        logger.info(f"Found {len(results)} results")
        return results, result_distances, result_indices
    
    def add_documents(
        self,
        embeddings: np.ndarray,
        documents: List[Document],
    ):
        """
        Add new documents to existing index
        
        Args:
            embeddings: New embeddings to add
            documents: Corresponding documents
        """
        embeddings = embeddings.astype('float32')
        
        self.index.add(embeddings)
        self.documents.extend(documents)
        
        if self.embeddings is not None:
            self.embeddings = np.vstack([self.embeddings, embeddings])
        else:
            self.embeddings = embeddings
        
        logger.info(f"Added {len(documents)} documents. Total: {self.index.ntotal}")
    
    def save(self, save_dir: str):
        """
        Save index and documents to disk
        
        Args:
            save_dir: Directory to save files
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        index_path = save_dir / "faiss_index.bin"
        faiss.write_index(self.index, str(index_path))
        logger.info(f"Saved FAISS index to {index_path}")
        
        # Save documents
        docs_path = save_dir / "documents.pkl"
        with open(docs_path, 'wb') as f:
            pickle.dump(self.documents, f)
        logger.info(f"Saved documents to {docs_path}")
        
        # Save embeddings
        if self.embeddings is not None:
            emb_path = save_dir / "embeddings.npy"
            np.save(emb_path, self.embeddings)
            logger.info(f"Saved embeddings to {emb_path}")
        
        # Save config
        config = {
            'embedding_dim': self.embedding_dim,
            'index_type': self.index_type,
            'n_clusters': self.n_clusters,
        }
        config_path = save_dir / "config.pkl"
        with open(config_path, 'wb') as f:
            pickle.dump(config, f)
    
    def load(self, save_dir: str):
        """
        Load index and documents from disk
        
        Args:
            save_dir: Directory containing saved files
        """
        save_dir = Path(save_dir)
        
        # Load config
        config_path = save_dir / "config.pkl"
        with open(config_path, 'rb') as f:
            config = pickle.load(f)
        
        self.embedding_dim = config['embedding_dim']
        self.index_type = config['index_type']
        self.n_clusters = config['n_clusters']
        
        # Load FAISS index
        index_path = save_dir / "faiss_index.bin"
        self.index = faiss.read_index(str(index_path))
        logger.info(f"Loaded FAISS index from {index_path}")
        
        # Load documents
        docs_path = save_dir / "documents.pkl"
        with open(docs_path, 'rb') as f:
            self.documents = pickle.load(f)
        logger.info(f"Loaded {len(self.documents)} documents")
        
        # Load embeddings
        emb_path = save_dir / "embeddings.npy"
        if emb_path.exists():
            self.embeddings = np.load(emb_path)
            logger.info(f"Loaded embeddings of shape {self.embeddings.shape}")
    
    def get_stats(self) -> dict:
        """Get statistics about the index"""
        return {
            'total_vectors': self.index.ntotal if self.index else 0,
            'embedding_dim': self.embedding_dim,
            'index_type': self.index_type,
            'n_documents': len(self.documents),
        }


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create sample data
    n_docs = 1000
    dim = 384
    embeddings = np.random.randn(n_docs, dim).astype('float32')
    documents = [Document(page_content=f"Document {i}", metadata={'id': i}) for i in range(n_docs)]
    
    # Build index
    index = FAISSIndex(embedding_dim=dim, index_type="IVFFlat", n_clusters=10)
    index.build(embeddings, documents)
    
    # Search
    query = np.random.randn(dim).astype('float32')
    results, distances, indices = index.search(query, k=5)
    
    print(f"Found {len(results)} results")
    print(f"Stats: {index.get_stats()}")