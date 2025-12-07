"""
RAG Pipeline
Main orchestration module for the RAG system
"""

import os
import time
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import yaml

from data.ingestion import DocumentIngestion
from data.chunking import DocumentChunker
from embeddings.encoder import EmbeddingEncoder
from retrieval.dense import DenseRetriever
from retrieval.bm25 import BM25Retriever
from retrieval.hybrid import HybridRetriever
from retrieval.reranker import Reranker, RetrievalWithReranking
from generation.llm_wrapper import LLMWrapper
from generation.prompt_templates import PromptTemplates
from evaluation.metrics import LatencyMetrics

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RAGPipeline:
    """Complete RAG pipeline integrating all components"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize RAG pipeline
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize components
        self.ingestion = None
        self.chunker = None
        self.encoder = None
        self.retriever = None
        self.reranker = None
        self.llm = None
        
        # Metrics
        self.retrieval_latency = LatencyMetrics()
        self.generation_latency = LatencyMetrics()
        
        # Initialize pipeline
        self._initialize_components()
        
        logger.info("RAG Pipeline initialized successfully")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def _initialize_components(self):
        """Initialize all pipeline components"""
        
        # Document ingestion
        self.ingestion = DocumentIngestion(
            data_dir=self.config['datasets']['data_dir']
        )
        
        # Document chunking
        chunking_config = self.config['chunking']
        self.chunker = DocumentChunker(
            strategy=chunking_config['strategy'],
            chunk_size=chunking_config['chunk_size'],
            chunk_overlap=chunking_config['chunk_overlap'],
        )
        
        # Embedding encoder
        embedding_config = self.config['embedding']
        self.encoder = EmbeddingEncoder(
            model_name=embedding_config['model_name'],
            device=embedding_config['device'],
            normalize_embeddings=embedding_config['normalize_embeddings'],
            batch_size=embedding_config['batch_size'],
        )
        
        # LLM
        llm_config = self.config['llm']
        try:
            self.llm = LLMWrapper(
                model_name=llm_config['model_name'],
                temperature=llm_config['temperature'],
                max_tokens=llm_config['max_tokens'],
            )
        except ValueError as e:
            logger.error(f"Failed to initialize LLM: {e}")
            logger.error("Please set GROQ_API_KEY in your .env file")
            raise
    
    def build_index(
        self,
        documents_dir: str = None,
        save_index: bool = True,
    ):
        """
        Build the retrieval index from documents
        
        Args:
            documents_dir: Directory containing documents
            save_index: Whether to save the index
        """
        logger.info("Starting index building process...")
        
        # Load documents
        if documents_dir is None:
            documents_dir = self.config['datasets']['data_dir']
        
        logger.info(f"Loading documents from {documents_dir}...")
        documents = self.ingestion.load_directory(documents_dir)
        documents = self.ingestion.preprocess_documents(documents)
        
        if not documents:
            logger.warning("No documents found!")
            return
        
        # Chunk documents
        logger.info("Chunking documents...")
        chunked_docs = self.chunker.chunk_documents(documents)
        
        logger.info(f"Created {len(chunked_docs)} chunks from {len(documents)} documents")
        
        # Build retrievers based on strategy
        retrieval_config = self.config['retrieval']
        strategy = retrieval_config['strategy']
        
        if strategy == "dense":
            self._build_dense_retriever(chunked_docs)
        
        elif strategy == "hybrid":
            self._build_hybrid_retriever(chunked_docs)
        
        elif strategy == "rerank":
            self._build_rerank_retriever(chunked_docs)
        
        else:
            logger.warning(f"Unknown strategy {strategy}, using dense")
            self._build_dense_retriever(chunked_docs)
        
        # Save index
        if save_index:
            index_dir = self.config['datasets']['index_dir']
            self.save_index(index_dir)
        
        logger.info("Index building complete!")
    
    def _build_dense_retriever(self, documents):
        """Build dense retriever"""
        logger.info("Building dense retriever...")
        
        retriever = DenseRetriever(
            encoder=self.encoder,
            top_k=self.config['retrieval']['top_k'],
        )
        
        indexing_config = self.config['indexing']
        retriever.build_index(
            documents,
            index_type=indexing_config['index_type'],
            n_clusters=indexing_config['n_clusters'],
        )
        
        self.retriever = retriever
    
    def _build_hybrid_retriever(self, documents):
        """Build hybrid retriever (dense + BM25)"""
        logger.info("Building hybrid retriever...")
        
        # Build dense retriever
        dense_retriever = DenseRetriever(
            encoder=self.encoder,
            top_k=self.config['retrieval']['top_k'],
        )
        
        indexing_config = self.config['indexing']
        dense_retriever.build_index(
            documents,
            index_type=indexing_config['index_type'],
            n_clusters=indexing_config['n_clusters'],
        )
        
        # Build BM25 retriever
        bm25_retriever = BM25Retriever(
            documents=documents,
            top_k=self.config['retrieval']['top_k'],
        )
        
        # Combine into hybrid
        self.retriever = HybridRetriever(
            dense_retriever=dense_retriever,
            bm25_retriever=bm25_retriever,
            alpha=self.config['retrieval']['hybrid_alpha'],
            top_k=self.config['retrieval']['top_k'],
        )
    
    def _build_rerank_retriever(self, documents):
        """Build retriever with reranking"""
        logger.info("Building retriever with reranking...")
        
        # Build base hybrid retriever
        self._build_hybrid_retriever(documents)
        
        # Add reranker
        if self.config['reranking']['enabled']:
            self.reranker = Reranker(
                model_name=self.config['reranking']['model_name'],
                batch_size=self.config['reranking']['batch_size'],
            )
            
            # Wrap retriever with reranking
            self.retriever = RetrievalWithReranking(
                retriever=self.retriever,
                reranker=self.reranker,
                retrieve_k=self.config['retrieval']['top_k'],
                rerank_top_n=self.config['retrieval']['rerank_top_n'],
            )
    
    def query(
        self,
        question: str,
        return_context: bool = False,
        chat_history: List[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Query the RAG system
        
        Args:
            question: User question
            return_context: Whether to return retrieved context
            chat_history: Optional chat history
            
        Returns:
            Dictionary with answer and metadata
        """
        if self.retriever is None:
            return {
                'answer': "Index not built. Please run build_index() first.",
                'error': True
            }
        
        # Retrieve relevant documents
        start_time = time.time()
        retrieved_docs = self.retriever.retrieve(
            question,
            return_scores=False
        )
        
        if isinstance(retrieved_docs[0], tuple):
            retrieved_docs = [doc for doc, _ in retrieved_docs]
        
        retrieval_time = (time.time() - start_time) * 1000
        self.retrieval_latency.add(retrieval_time)
        
        logger.info(f"Retrieved {len(retrieved_docs)} documents in {retrieval_time:.2f}ms")
        
        # Generate answer
        start_time = time.time()
        
        if chat_history:
            answer = self.llm.generate_conversational(
                question,
                retrieved_docs,
                chat_history=chat_history,
            )
        else:
            answer = self.llm.generate_with_context(
                question,
                retrieved_docs,
            )
        
        generation_time = (time.time() - start_time) * 1000
        self.generation_latency.add(generation_time)
        
        logger.info(f"Generated answer in {generation_time:.2f}ms")
        
        # Prepare response
        response = {
            'answer': answer,
            'retrieval_time_ms': retrieval_time,
            'generation_time_ms': generation_time,
            'total_time_ms': retrieval_time + generation_time,
            'num_docs_retrieved': len(retrieved_docs),
        }
        
        if return_context:
            response['context'] = [
                {
                    'content': doc.page_content,
                    'metadata': doc.metadata
                }
                for doc in retrieved_docs
            ]
        
        return response
    
    def batch_query(
        self,
        questions: List[str],
    ) -> List[Dict[str, Any]]:
        """
        Process multiple queries
        
        Args:
            questions: List of questions
            
        Returns:
            List of responses
        """
        responses = []
        
        for question in questions:
            response = self.query(question)
            responses.append(response)
        
        return responses
    
    def save_index(self, save_dir: str):
        """
        Save the index to disk
        
        Args:
            save_dir: Directory to save index
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save based on retriever type
        if isinstance(self.retriever, DenseRetriever):
            self.retriever.save_index(str(save_dir / "dense"))
        
        elif isinstance(self.retriever, HybridRetriever):
            self.retriever.dense_retriever.save_index(str(save_dir / "dense"))
            self.retriever.bm25_retriever.save(str(save_dir / "bm25"))
        
        elif isinstance(self.retriever, RetrievalWithReranking):
            # Save the underlying retriever
            base_retriever = self.retriever.retriever
            if isinstance(base_retriever, HybridRetriever):
                base_retriever.dense_retriever.save_index(str(save_dir / "dense"))
                base_retriever.bm25_retriever.save(str(save_dir / "bm25"))
        
        logger.info(f"Index saved to {save_dir}")
    
    def load_index(self, load_dir: str):
        """
        Load the index from disk
        
        Args:
            load_dir: Directory containing saved index
        """
        load_dir = Path(load_dir)
        
        retrieval_config = self.config['retrieval']
        strategy = retrieval_config['strategy']
        
        if strategy == "dense":
            self.retriever = DenseRetriever(
                encoder=self.encoder,
                top_k=retrieval_config['top_k'],
            )
            self.retriever.load_index(str(load_dir / "dense"))
        
        elif strategy in ["hybrid", "rerank"]:
            # Load dense retriever
            dense_retriever = DenseRetriever(
                encoder=self.encoder,
                top_k=retrieval_config['top_k'],
            )
            dense_retriever.load_index(str(load_dir / "dense"))
            
            # Load BM25 retriever
            bm25_retriever = BM25Retriever(top_k=retrieval_config['top_k'])
            bm25_retriever.load(str(load_dir / "bm25"))
            
            # Create hybrid
            self.retriever = HybridRetriever(
                dense_retriever=dense_retriever,
                bm25_retriever=bm25_retriever,
                alpha=retrieval_config['hybrid_alpha'],
                top_k=retrieval_config['top_k'],
            )
            
            # Add reranking if needed
            if strategy == "rerank" and self.config['reranking']['enabled']:
                self.reranker = Reranker(
                    model_name=self.config['reranking']['model_name'],
                )
                self.retriever = RetrievalWithReranking(
                    retriever=self.retriever,
                    reranker=self.reranker,
                    retrieve_k=retrieval_config['top_k'],
                    rerank_top_n=retrieval_config['rerank_top_n'],
                )
        
        logger.info(f"Index loaded from {load_dir}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        stats = {
            'retrieval_latency': self.retrieval_latency.get_stats(),
            'generation_latency': self.generation_latency.get_stats(),
        }
        
        if self.retriever:
            if isinstance(self.retriever, DenseRetriever):
                stats['index'] = self.retriever.get_index_stats()
            elif isinstance(self.retriever, HybridRetriever):
                stats['dense_index'] = self.retriever.dense_retriever.get_index_stats()
                stats['bm25_index'] = self.retriever.bm25_retriever.get_stats()
        
        return stats


if __name__ == "__main__":
    # Example usage
    pipeline = RAGPipeline()
    
    # Build index
    pipeline.build_index()
    
    # Query
    response = pipeline.query("What is retrieval-augmented generation?")
    
    print(f"\nAnswer: {response['answer']}")
    print(f"Time: {response['total_time_ms']:.2f}ms")
    
    # Get stats
    stats = pipeline.get_stats()
    print(f"\nPipeline Stats: {stats}")