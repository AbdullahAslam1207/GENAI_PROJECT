"""
Multi-Query Retrieval Evaluation Script
Extended version for evaluating multiple queries across all retrieval strategies
"""

import os
import time
import logging
from pathlib import Path
from typing import List, Dict, Any
import yaml
import json

from rag_pipeline import RAGPipeline
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MultiQueryRetrieverEvaluator:
    """Evaluates different retrieval methods across multiple queries"""
    
    def __init__(self, config_path: str = "config.yaml", documents_dir: str = None):
        """
        Initialize the evaluator
        
        Args:
            config_path: Path to config file
            documents_dir: Directory containing documents
        """
        self.config_path = config_path
        
        # Load base config
        with open(config_path, 'r') as f:
            self.base_config = yaml.safe_load(f)
        
        self.documents_dir = documents_dir or self.base_config['datasets']['data_dir']
        
        # Retrieval strategies to evaluate
        self.retrieval_strategies = ['dense', 'hybrid', 'rerank']
        
        # Store results
        self.all_results = []
        
        # Pre-built pipelines (build once, reuse for all queries)
        self.pipelines = {}
    
    def build_all_indexes(self):
        """Build indexes for all retrieval strategies once"""
        logger.info("\n" + "="*80)
        logger.info("BUILDING INDEXES FOR ALL STRATEGIES")
        logger.info("="*80 + "\n")
        
        for strategy in self.retrieval_strategies:
            logger.info(f"Building index for: {strategy.upper()}")
            
            # Create temporary config with this strategy
            config = self.base_config.copy()
            config['retrieval']['strategy'] = strategy
            
            # Save temporary config
            temp_config_path = f"config_temp_{strategy}.yaml"
            with open(temp_config_path, 'w') as f:
                yaml.dump(config, f)
            
            # Initialize pipeline
            pipeline = RAGPipeline(config_path=temp_config_path)
            
            # Build index
            pipeline.build_index(
                documents_dir=self.documents_dir,
                save_index=False  # Keep in memory
            )
            
            # Store pipeline
            self.pipelines[strategy] = pipeline
            
            # Clean up temp config
            os.remove(temp_config_path)
            
            logger.info(f"✓ {strategy.upper()} index ready\n")
    
    def evaluate_single_query_strategy(
        self,
        query: str,
        strategy: str
    ) -> Dict[str, Any]:
        """
        Evaluate a single query with one strategy
        
        Args:
            query: Query string
            strategy: Retrieval strategy name
            
        Returns:
            Dictionary with evaluation results
        """
        pipeline = self.pipelines[strategy]
        
        # Measure retrieval time
        start_time = time.time()
        
        # Retrieve documents (handle rerank case differently)
        if strategy == 'rerank':
            # RetrievalWithReranking doesn't accept k parameter
            retrieved_docs = pipeline.retriever.retrieve(
                query=query,
                return_scores=True
            )
        else:
            # Dense and Hybrid retrievers accept k parameter
            retrieved_docs = pipeline.retriever.retrieve(
                query=query,
                k=pipeline.config['retrieval']['top_k'],
                return_scores=True
            )
        
        retrieval_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Extract results
        num_docs = len(retrieved_docs)
        top_scores = [score for _, score in retrieved_docs[:5]] if retrieved_docs else []
        
        result = {
            'strategy': strategy,
            'query': query,
            'num_docs_retrieved': num_docs,
            'retrieval_time_ms': round(retrieval_time, 2),
            'top_5_scores': [round(s, 4) for s in top_scores],
            'avg_score': round(sum(top_scores) / len(top_scores), 4) if top_scores else 0,
            'max_score': round(max(top_scores), 4) if top_scores else 0,
        }
        
        return result
    
    def evaluate_query_all_strategies(self, query: str) -> Dict[str, Any]:
        """
        Evaluate one query across all strategies
        
        Args:
            query: Query string
            
        Returns:
            Dictionary with results for all strategies
        """
        logger.info(f"\nEvaluating: {query[:60]}...")
        
        query_results = {
            'query': query,
            'strategies': {}
        }
        
        for strategy in self.retrieval_strategies:
            try:
                result = self.evaluate_single_query_strategy(query, strategy)
                query_results['strategies'][strategy] = result
                logger.info(f"  {strategy.upper():<10} - Time: {result['retrieval_time_ms']:>6.2f}ms, Top Score: {result['max_score']:.4f}")
            except Exception as e:
                logger.error(f"Error with {strategy}: {e}")
        
        return query_results
    
    def evaluate_multiple_queries(self, queries: List[str]):
        """
        Evaluate multiple queries across all strategies
        
        Args:
            queries: List of query strings
        """
        print("\n" + "#"*80)
        print("# MULTI-QUERY RETRIEVAL EVALUATION")
        print(f"# Queries: {len(queries)}")
        print(f"# Strategies: {', '.join([s.upper() for s in self.retrieval_strategies])}")
        print("#"*80 + "\n")
        
        # Build indexes once
        self.build_all_indexes()
        
        print("\n" + "="*80)
        print("EVALUATING QUERIES")
        print("="*80)
        
        # Evaluate each query
        for i, query in enumerate(queries, 1):
            print(f"\n[{i}/{len(queries)}] Query: {query}")
            query_result = self.evaluate_query_all_strategies(query)
            self.all_results.append(query_result)
        
        # Print aggregate summary
        self._print_aggregate_summary()
    
    def _print_aggregate_summary(self):
        """Print aggregate summary across all queries"""
        print("\n\n" + "="*80)
        print("AGGREGATE SUMMARY - ALL QUERIES")
        print("="*80 + "\n")
        
        # Calculate aggregate metrics for each strategy
        strategy_stats = {strategy: {
            'total_time': 0,
            'avg_max_score': 0,
            'avg_avg_score': 0,
            'num_queries': 0
        } for strategy in self.retrieval_strategies}
        
        for query_result in self.all_results:
            for strategy, result in query_result['strategies'].items():
                strategy_stats[strategy]['total_time'] += result['retrieval_time_ms']
                strategy_stats[strategy]['avg_max_score'] += result['max_score']
                strategy_stats[strategy]['avg_avg_score'] += result['avg_score']
                strategy_stats[strategy]['num_queries'] += 1
        
        # Calculate averages
        for strategy in strategy_stats:
            n = strategy_stats[strategy]['num_queries']
            if n > 0:
                strategy_stats[strategy]['avg_time'] = strategy_stats[strategy]['total_time'] / n
                strategy_stats[strategy]['avg_max_score'] = strategy_stats[strategy]['avg_max_score'] / n
                strategy_stats[strategy]['avg_avg_score'] = strategy_stats[strategy]['avg_avg_score'] / n
        
        # Print table
        print(f"{'Strategy':<15} {'Avg Time (ms)':<15} {'Avg Max Score':<15} {'Avg Consistency':<15}")
        print("-" * 80)
        
        for strategy in self.retrieval_strategies:
            stats = strategy_stats[strategy]
            print(f"{strategy.upper():<15} {stats['avg_time']:<15.2f} {stats['avg_max_score']:<15.4f} {stats['avg_avg_score']:<15.4f}")
        
        print("\n" + "="*80 + "\n")
        
        # Print query-by-query comparison
        print("\nQUERY-BY-QUERY COMPARISON\n")
        print("-" * 80)
        
        for i, query_result in enumerate(self.all_results, 1):
            query = query_result['query']
            print(f"\n{i}. {query}")
            print(f"   {'Strategy':<12} {'Time (ms)':<12} {'Max Score':<12}")
            
            for strategy in self.retrieval_strategies:
                result = query_result['strategies'][strategy]
                print(f"   {strategy.upper():<12} {result['retrieval_time_ms']:<12.2f} {result['max_score']:<12.4f}")
    
    def save_results(self, output_file: str = "multi_query_evaluation_results.json"):
        """
        Save results to JSON file
        
        Args:
            output_file: Output file path
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.all_results, f, indent=2)
        
        logger.info(f"\n✓ Results saved to {output_file}")


def main():
    """Main execution function"""
    
    # Initialize evaluator
    evaluator = MultiQueryRetrieverEvaluator(config_path="config.yaml")
    
    # Test queries about Vision Transformers paper
    test_queries = [
        "What is the key innovation of Vision Transformers compared to CNNs?",
        # Add more queries here as needed
        # Examples:
        # "How does the attention mechanism work in Vision Transformers?",
        # "What are the computational requirements of ViT?",
        # "How does ViT perform on image classification tasks?",
        # "What is the patch embedding process in Vision Transformers?",
    ]
    
    # Evaluate all queries across all strategies
    evaluator.evaluate_multiple_queries(test_queries)
    
    # Save results
    evaluator.save_results()
    
    print("\n" + "#"*80)
    print("# EVALUATION COMPLETE!")
    print("#"*80 + "\n")
    print("✓ Results saved to: multi_query_evaluation_results.json\n")


if __name__ == "__main__":
    main()
