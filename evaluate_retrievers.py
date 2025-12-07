"""
Retrieval Methods Evaluation Script
Evaluates different retrieval strategies (dense, hybrid, rerank) on test queries
"""

import os
import time
import logging
from pathlib import Path
from typing import List, Dict, Any
import yaml

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


class RetrieverEvaluator:
    """Evaluates different retrieval methods"""
    
    def __init__(self, config_path: str = "config.yaml", documents_dir: str = None, llm_models: List[str] = None):
        """
        Initialize the evaluator
        
        Args:
            config_path: Path to config file
            documents_dir: Directory containing documents
            llm_models: List of LLM model names to test
        """
        self.config_path = config_path
        
        # Load base config
        with open(config_path, 'r') as f:
            self.base_config = yaml.safe_load(f)
        
        self.documents_dir = documents_dir or self.base_config['datasets']['data_dir']
        
        # Retrieval strategies to evaluate
        self.retrieval_strategies = ['dense', 'hybrid', 'rerank']
        
        # LLM models to test
        if llm_models is None:
            llm_models = [
                "llama-3.1-8b-instant",
                "llama-3.3-70b-versatile", 
                "llama-guard-3-8b"
            ]
        self.llm_models = llm_models
        
        # Store results
        self.results = []
    
    def build_index_for_strategy(self, strategy: str) -> RAGPipeline:
        """
        Build index for a specific retrieval strategy
        
        Args:
            strategy: Retrieval strategy (dense, hybrid, rerank)
            
        Returns:
            Initialized RAG pipeline
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"Building index for strategy: {strategy.upper()}")
        logger.info(f"{'='*80}\n")
        
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
            save_index=False  # Don't save, keep in memory
        )
        
        # Clean up temp config
        os.remove(temp_config_path)
        
        logger.info(f"✓ Index built for {strategy} strategy\n")
        
        return pipeline
    
    def evaluate_query(
        self,
        query: str,
        strategy: str,
        pipeline: RAGPipeline,
        include_generation: bool = True,
        llm_models: List[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a single query with a retrieval strategy
        
        Args:
            query: Query string
            strategy: Retrieval strategy name
            pipeline: RAG pipeline to use
            include_generation: Whether to include LLM generation
            llm_models: List of LLM models to test (uses default if None)
            
        Returns:
            Dictionary with evaluation results
        """
        logger.info(f"Querying with {strategy} strategy...")
        
        # Measure retrieval time
        start_time = time.time()
        
        # Get retrieval results
        if pipeline.retriever is None:
            logger.error("Retriever not initialized!")
            return None
        
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
        top_scores = [score for _, score in retrieved_docs[:5]]
        
        result = {
            'strategy': strategy,
            'query': query,
            'num_docs_retrieved': num_docs,
            'retrieval_time_ms': round(retrieval_time, 2),
            'top_5_scores': [round(s, 4) for s in top_scores],
            'top_3_documents': [],
            'llm_results': {}  # Store results for each LLM model
        }
        
        # Store top 3 document previews
        for i, (doc, score) in enumerate(retrieved_docs[:3], 1):
            result['top_3_documents'].append({
                'rank': i,
                'score': round(score, 4),
                'content_preview': doc.page_content[:200] + "...",
                'metadata': doc.metadata
            })
        
        logger.info(f"  Retrieved: {num_docs} documents")
        logger.info(f"  Retrieval time: {retrieval_time:.2f}ms")
        logger.info(f"  Top score: {top_scores[0]:.4f}")
        
        # Generate answers with multiple LLMs if requested
        if include_generation and pipeline.llm is not None:
            if llm_models is None:
                llm_models = self.llm_models
            
            # Extract documents without scores
            docs_only = [doc for doc, _ in retrieved_docs]
            
            for model_name in llm_models:
                try:
                    logger.info(f"  Generating with {model_name}...")
                    gen_start = time.time()
                    
                    # Update LLM model
                    original_model = pipeline.llm.model_name
                    pipeline.llm.model_name = model_name
                    pipeline.llm.llm = pipeline.llm._initialize_llm(model_name)
                    
                    # Generate answer
                    answer = pipeline.llm.generate_with_context(
                        query=query,
                        context_documents=docs_only[:5]  # Use top 5 for generation
                    )
                    
                    generation_time = (time.time() - gen_start) * 1000
                    
                    result['llm_results'][model_name] = {
                        'answer': answer,
                        'generation_time_ms': round(generation_time, 2),
                        'total_time_ms': round(retrieval_time + generation_time, 2)
                    }
                    
                    logger.info(f"    Generation time: {generation_time:.2f}ms")
                    
                    # Restore original model
                    pipeline.llm.model_name = original_model
                    pipeline.llm.llm = pipeline.llm._initialize_llm(original_model)
                    
                except Exception as e:
                    logger.warning(f"    LLM generation failed for {model_name}: {e}")
                    result['llm_results'][model_name] = {
                        'answer': f"Error: {str(e)}",
                        'generation_time_ms': 0,
                        'total_time_ms': round(retrieval_time, 2)
                    }
        
        logger.info("")
        return result
    
    def evaluate_all_strategies(self, query: str):
        """
        Evaluate all retrieval strategies on a single query
        
        Args:
            query: Query string to evaluate
        """
        print("\n" + "="*80)
        print(f"EVALUATING QUERY: {query}")
        print("="*80 + "\n")
        
        query_results = {
            'query': query,
            'strategies': {}
        }
        
        for strategy in self.retrieval_strategies:
            try:
                # Build index for this strategy
                pipeline = self.build_index_for_strategy(strategy)
                
                # Evaluate query
                result = self.evaluate_query(query, strategy, pipeline)
                
                if result:
                    query_results['strategies'][strategy] = result
                    self.results.append(result)
                
            except Exception as e:
                logger.error(f"Error evaluating {strategy}: {e}")
                continue
        
        # Print comparison summary
        self._print_comparison_summary(query_results)
    
    def _print_comparison_summary(self, query_results: Dict[str, Any]):
        """
        Print a comparison summary of all strategies
        
        Args:
            query_results: Results for all strategies
        """
        print("\n" + "="*80)
        print("COMPARISON SUMMARY")
        print("="*80 + "\n")
        
        strategies = query_results['strategies']
        
        if not strategies:
            print("No results to compare.")
            return
        
        # Check if generation is included
        has_generation = any(result.get('llm_results') for result in strategies.values())
        
        # Print retrieval metrics table
        print(f"{'Strategy':<15} {'Docs':<8} {'Ret Time (ms)':<15} {'Top Score':<12} {'Avg Top-5':<12}")
        print("-" * 80)
        
        for strategy_name, result in strategies.items():
            docs = result['num_docs_retrieved']
            ret_time = result['retrieval_time_ms']
            top_score = result['top_5_scores'][0] if result['top_5_scores'] else 0
            avg_score = sum(result['top_5_scores']) / len(result['top_5_scores']) if result['top_5_scores'] else 0
            
            print(f"{strategy_name.upper():<15} {docs:<8} {ret_time:<15.2f} {top_score:<12.4f} {avg_score:<12.4f}")
        
        print("\n" + "="*80 + "\n")
        
        # Print LLM generation results if available
        if has_generation:
            # Get all LLM models tested
            llm_models = set()
            for result in strategies.values():
                llm_results = result.get('llm_results', {})
                llm_models.update(llm_results.keys())
            llm_models = sorted(llm_models)
            
            if llm_models:
                print("\n" + "="*80)
                print("LLM GENERATION TIME COMPARISON")
                print("="*80 + "\n")
                
                # Header
                header = f"{'Strategy':<15}"
                for model in llm_models:
                    model_short = model.split('/')[-1][:20]  # Shorten model name
                    header += f" {model_short:<22}"
                print(header)
                print("-" * (15 + 23 * len(llm_models)))
                
                # Print generation times for each strategy
                for strategy_name, result in strategies.items():
                    row = f"{strategy_name.upper():<15}"
                    llm_results = result.get('llm_results', {})
                    for model in llm_models:
                        if model in llm_results:
                            gen_time = llm_results[model]['generation_time_ms']
                            row += f" {gen_time:<22.2f}"
                        else:
                            row += f" {'N/A':<22}"
                    print(row)
                
                print("\n" + "="*80 + "\n")
                
                # Print generated answers for each strategy and LLM
                print("\n" + "="*80)
                print("GENERATED ANSWERS COMPARISON")
                print("="*80 + "\n")
                
                for strategy_name, result in strategies.items():
                    print(f"\n{'═'*80}")
                    print(f"RETRIEVAL STRATEGY: {strategy_name.upper()}")
                    print(f"{'═'*80}\n")
                    
                    llm_results = result.get('llm_results', {})
                    for model_name in llm_models:
                        if model_name in llm_results:
                            answer_data = llm_results[model_name]
                            answer = answer_data['answer']
                            gen_time = answer_data['generation_time_ms']
                            
                            print(f"{'─'*80}")
                            print(f"LLM: {model_name}")
                            print(f"Generation Time: {gen_time:.2f}ms")
                            print(f"{'─'*80}")
                            print(f"{answer}")
                            print()
        
        print("\n" + "="*80 + "\n")
        
        # Print top 3 documents for each strategy
        for strategy_name, result in strategies.items():
            print(f"\n{'─'*80}")
            print(f"TOP 3 DOCUMENTS - {strategy_name.upper()} STRATEGY")
            print(f"{'─'*80}\n")
            
            for doc_info in result['top_3_documents']:
                print(f"Rank {doc_info['rank']} | Score: {doc_info['score']:.4f}")
                print(f"Content: {doc_info['content_preview']}")
                print(f"Metadata: {doc_info['metadata']}\n")
    
    def save_results(self, output_file: str = "retrieval_evaluation_results.txt"):
        """
        Save results to a file
        
        Args:
            output_file: Output file path
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("RETRIEVAL METHODS & LLM EVALUATION RESULTS\n")
            f.write("="*80 + "\n\n")
            
            for result in self.results:
                f.write(f"\nQuery: {result['query']}\n")
                f.write(f"Strategy: {result['strategy'].upper()}\n")
                f.write(f"Documents Retrieved: {result['num_docs_retrieved']}\n")
                f.write(f"Retrieval Time: {result['retrieval_time_ms']}ms\n")
                f.write(f"Top 5 Scores: {result['top_5_scores']}\n")
                
                # Write LLM generation results if available
                llm_results = result.get('llm_results', {})
                if llm_results:
                    f.write("\n" + "="*80 + "\n")
                    f.write("LLM GENERATED ANSWERS\n")
                    f.write("="*80 + "\n\n")
                    
                    for model_name, model_result in llm_results.items():
                        f.write("-" * 80 + "\n")
                        f.write(f"LLM Model: {model_name}\n")
                        f.write(f"Generation Time: {model_result['generation_time_ms']}ms\n")
                        f.write(f"Total Time (Retrieval + Generation): {model_result['total_time_ms']}ms\n")
                        f.write("-" * 80 + "\n")
                        f.write(f"{model_result['answer']}\n\n")
                
                f.write("\n" + "-" * 80 + "\n")
                f.write("TOP 3 RETRIEVED DOCUMENTS\n")
                f.write("-" * 80 + "\n")
                
                for doc_info in result['top_3_documents']:
                    f.write(f"\n  Rank {doc_info['rank']} | Score: {doc_info['score']}\n")
                    f.write(f"  {doc_info['content_preview']}\n")
                
                f.write("\n" + "="*80 + "\n")
        
        logger.info(f"Results saved to {output_file}")


def main():
    """Main execution function"""
    
    # Initialize evaluator with multiple LLM models
    llm_models = [
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile",
        "qwen/qwen3-32b"
    ]
    
    evaluator = RetrieverEvaluator(
        config_path="config.yaml",
        llm_models=llm_models
    )
    
    # Test query about BERT (the actual paper content)
    test_query = "What is the key innovation of BERT compared to previous language models like GPT and ELMo?"
    
    print("\n" + "#"*80)
    print("# RETRIEVAL METHODS & MULTI-LLM EVALUATION")
    print(f"# Testing Retrieval: Dense, Hybrid, and Rerank")
    print(f"# Testing LLMs: {', '.join([m.split('/')[-1] for m in llm_models])}")
    print("#"*80)
    
    # Evaluate all strategies on the test query
    evaluator.evaluate_all_strategies(test_query)
    
    # Save results
    evaluator.save_results()
    
    print("\n" + "#"*80)
    print("# EVALUATION COMPLETE!")
    print("#"*80 + "\n")
    
    print("✓ Results saved to: retrieval_evaluation_results.txt")
    print(f"✓ Tested {len(evaluator.retrieval_strategies)} retrieval strategies")
    print(f"✓ Tested {len(llm_models)} LLM models")
    print("\nTo add more queries, modify the test_query in main()")


if __name__ == "__main__":
    main()
