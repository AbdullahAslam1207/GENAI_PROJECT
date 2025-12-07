"""
Multi-Query Multi-PDF Evaluation Script
Evaluates retrieval strategies and LLMs across multiple documents and queries
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


class MultiQueryEvaluator:
    """Evaluates different retrieval methods across multiple queries and documents"""
    
    def __init__(self, config_path: str = "config.yaml", llm_models: List[str] = None):
        """
        Initialize the evaluator
        
        Args:
            config_path: Path to config file
            llm_models: List of LLM model names to test
        """
        self.config_path = config_path
        
        # Load base config
        with open(config_path, 'r') as f:
            self.base_config = yaml.safe_load(f)
        
        # Retrieval strategies to evaluate
        self.retrieval_strategies = ['dense', 'hybrid', 'rerank']
        
        # LLM models to test
        self.llm_models = llm_models or ["llama-3.1-8b-instant"]
        
        # Store results
        self.all_results = []
        
        # Pipelines for each strategy
        self.pipelines = {}
    
    def _build_indexes(self):
        """Build indexes for all retrieval strategies"""
        logger.info("Building indexes for all retrieval strategies...")
        
        for strategy in self.retrieval_strategies:
            logger.info(f"Building index for {strategy} strategy...")
            
            # Initialize pipeline with config path
            pipeline = RAGPipeline(self.config_path)
            
            # Update the retrieval strategy
            pipeline.config['retrieval']['strategy'] = strategy
            
            # Build index (loads documents internally)
            pipeline.build_index()
            
            self.pipelines[strategy] = pipeline
            logger.info(f"✓ {strategy} index built")
        
        logger.info("All indexes built successfully!\n")
    
    def evaluate_query(self, query: str, pdf_name: str) -> Dict[str, Any]:
        """
        Evaluate a single query across all strategies and LLMs
        
        Args:
            query: Query string
            pdf_name: Name of the PDF this query is from
            
        Returns:
            Dictionary with results for all strategies and LLMs
        """
        result = {
            'query': query,
            'pdf': pdf_name,
            'strategies': {}
        }
        
        for strategy in self.retrieval_strategies:
            pipeline = self.pipelines[strategy]
            
            # Retrieve documents
            start_time = time.time()
            
            if strategy == 'rerank':
                retrieved = pipeline.retriever.retrieve(query=query, return_scores=True)
            else:
                retrieved = pipeline.retriever.retrieve(query=query, k=20, return_scores=True)
            
            retrieval_time = (time.time() - start_time) * 1000  # Convert to ms
            
            # Extract documents and scores
            docs_only = [doc for doc, _ in retrieved]
            scores = [score for _, score in retrieved]
            
            strategy_result = {
                'num_docs': len(docs_only),
                'retrieval_time_ms': retrieval_time,
                'scores': scores[:5],  # Top 5 scores
                'documents': docs_only[:3],  # Top 3 docs for inspection
                'llm_results': {}
            }
            
            # Test with each LLM model
            for model_name in self.llm_models:
                try:
                    # Reinitialize LLM with current model
                    pipeline.llm.model_name = model_name
                    pipeline.llm.llm = pipeline.llm._initialize_llm(model_name)
                    
                    # Generate answer
                    llm_start = time.time()
                    answer = pipeline.llm.generate_with_context(
                        query=query,
                        context_documents=docs_only[:5]
                    )
                    llm_time = (time.time() - llm_start) * 1000
                    
                    strategy_result['llm_results'][model_name] = {
                        'answer': answer,
                        'generation_time_ms': llm_time,
                        'total_time_ms': retrieval_time + llm_time
                    }
                    
                except Exception as e:
                    strategy_result['llm_results'][model_name] = {
                        'answer': f"Error: {str(e)}",
                        'generation_time_ms': 0,
                        'total_time_ms': retrieval_time
                    }
            
            result['strategies'][strategy] = strategy_result
        
        self.all_results.append(result)
        return result
    
    def evaluate_all_queries(self, queries: List[Dict[str, str]]):
        """
        Evaluate all queries
        
        Args:
            queries: List of dicts with 'query' and 'pdf' keys
        """
        print(f"\n{'='*80}")
        print(f"Evaluating {len(queries)} queries across {len(self.retrieval_strategies)} strategies and {len(self.llm_models)} LLMs")
        print(f"{'='*80}\n")
        
        for i, query_info in enumerate(queries, 1):
            query = query_info['query']
            pdf = query_info['pdf']
            
            print(f"[{i}/{len(queries)}] Processing query from {pdf}...")
            print(f"Query: {query[:80]}...")
            
            self.evaluate_query(query, pdf)
            print(f"✓ Completed\n")
    
    def _format_report(self) -> str:
        """Generate a clean formatted report"""
        report = []
        
        # Header
        report.append("=" * 100)
        report.append("RAG SYSTEM EVALUATION REPORT")
        report.append("=" * 100)
        report.append(f"\nTotal Queries: {len(self.all_results)}")
        report.append(f"Retrieval Strategies: {', '.join(self.retrieval_strategies)}")
        report.append(f"LLM Models: {', '.join(self.llm_models)}")
        report.append(f"\n{'=' * 100}\n")
        
        # Results for each query
        for idx, result in enumerate(self.all_results, 1):
            report.append(f"\n{'#' * 100}")
            report.append(f"QUERY {idx}: {result['pdf']}")
            report.append(f"{'#' * 100}")
            report.append(f"\nQuestion: {result['query']}\n")
            
            # Strategy comparison table
            report.append(f"{'─' * 100}")
            report.append("RETRIEVAL PERFORMANCE")
            report.append(f"{'─' * 100}")
            report.append(f"{'Strategy':<15} {'Docs':<8} {'Time (ms)':<12} {'Top Score':<12}")
            report.append(f"{'-' * 100}")
            
            for strategy in self.retrieval_strategies:
                strategy_data = result['strategies'][strategy]
                top_score = strategy_data['scores'][0] if strategy_data['scores'] else 0
                report.append(
                    f"{strategy.upper():<15} "
                    f"{strategy_data['num_docs']:<8} "
                    f"{strategy_data['retrieval_time_ms']:<12.2f} "
                    f"{top_score:<12.4f}"
                )
            
            # LLM Results
            report.append(f"\n{'─' * 100}")
            report.append("LLM GENERATION RESULTS")
            report.append(f"{'─' * 100}\n")
            
            for strategy in self.retrieval_strategies:
                strategy_data = result['strategies'][strategy]
                
                report.append(f"┌{'─' * 98}┐")
                report.append(f"│ {strategy.upper()} RETRIEVAL{' ' * (88 - len(strategy))}│")
                report.append(f"└{'─' * 98}┘\n")
                
                for model_name in self.llm_models:
                    llm_data = strategy_data['llm_results'][model_name]
                    
                    report.append(f"  Model: {model_name}")
                    report.append(f"  Generation Time: {llm_data['generation_time_ms']:.2f}ms")
                    report.append(f"  Total Time: {llm_data['total_time_ms']:.2f}ms")
                    report.append(f"\n  Answer:")
                    
                    # Format answer with proper line breaks
                    answer = llm_data['answer']
                    for line in answer.split('\n'):
                        if line.strip():
                            # Wrap long lines
                            words = line.split()
                            current_line = "  "
                            for word in words:
                                if len(current_line) + len(word) + 1 > 95:
                                    report.append(current_line)
                                    current_line = "  " + word
                                else:
                                    current_line += " " + word if current_line != "  " else word
                            if current_line.strip():
                                report.append(current_line)
                    
                    report.append("")  # Blank line between models
                
                report.append("")  # Blank line between strategies
        
        # Summary statistics
        report.append(f"\n{'=' * 100}")
        report.append("SUMMARY STATISTICS")
        report.append(f"{'=' * 100}\n")
        
        # Average times per strategy
        report.append("Average Retrieval Time by Strategy:")
        for strategy in self.retrieval_strategies:
            avg_time = sum(r['strategies'][strategy]['retrieval_time_ms'] 
                          for r in self.all_results) / len(self.all_results)
            report.append(f"  {strategy.upper():<15} {avg_time:.2f}ms")
        
        # Average generation time per LLM
        report.append("\nAverage Generation Time by LLM:")
        for model in self.llm_models:
            times = []
            for r in self.all_results:
                for strategy in self.retrieval_strategies:
                    llm_data = r['strategies'][strategy]['llm_results'][model]
                    if 'Error' not in llm_data['answer']:
                        times.append(llm_data['generation_time_ms'])
            if times:
                avg_time = sum(times) / len(times)
                report.append(f"  {model:<40} {avg_time:.2f}ms")
        
        report.append(f"\n{'=' * 100}")
        
        return '\n'.join(report)
    
    def save_results(self, filename: str = "evaluation_report.txt"):
        """Save evaluation results to file"""
        report = self._format_report()
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n✓ Report saved to: {filename}")


def main():
    """Main execution function"""
    
    # Define test queries from each PDF
    queries = [
        # BERT.pdf
        {
            'query': "What is the key innovation of BERT compared to previous language models like GPT and ELMo?",
            'pdf': "BERT.pdf"
        },
        {
            'query': "How does BERT's pre-training approach differ from traditional left-to-right language modeling?",
            'pdf': "BERT.pdf"
        },
        
        # GRAPHRAG.pdf
        {
            'query': "What is GraphRAG and how does it address the limitations of traditional RAG systems?",
            'pdf': "GRAPHRAG.pdf"
        },
        {
            'query': "How does GraphRAG handle global queries that require understanding of an entire dataset?",
            'pdf': "GRAPHRAG.pdf"
        },
        
        # PIXELRNN.pdf
        {
            'query': "What architectural novelties does PixelRNN introduce for image generation?",
            'pdf': "PIXELRNN.pdf"
        },
        {
            'query': "How does PixelRNN model the distribution of natural images sequentially?",
            'pdf': "PIXELRNN.pdf"
        },
        
        # REFRAG.pdf
        {
            'query': "What is the main efficiency problem that REFRAG aims to solve in RAG systems?",
            'pdf': "REFRAG.pdf"
        },
        {
            'query': "How does REFRAG reduce system latency while maintaining knowledge enrichment in RAG applications?",
            'pdf': "REFRAG.pdf"
        }
    ]
    
    # Initialize evaluator with multiple LLM models
    llm_models = [
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile",
        "qwen/qwen3-32b"
    ]
    
    print("\n" + "#"*80)
    print("# MULTI-PDF RAG EVALUATION")
    print(f"# PDFs: 4 (BERT, GRAPHRAG, PIXELRNN, REFRAG)")
    print(f"# Queries: {len(queries)} (2 per PDF)")
    print(f"# Retrieval Strategies: {len(['dense', 'hybrid', 'rerank'])}")
    print(f"# LLM Models: {len(llm_models)}")
    print("#"*80)
    
    evaluator = MultiQueryEvaluator(
        config_path="config.yaml",
        llm_models=llm_models
    )
    
    # Build indexes once for all strategies
    evaluator._build_indexes()
    
    # Evaluate all queries
    evaluator.evaluate_all_queries(queries)
    
    # Save results
    evaluator.save_results("evaluation_report.txt")
    
    print("\n" + "#"*80)
    print("# EVALUATION COMPLETE!")
    print("#"*80)
    print(f"\n✓ Evaluated {len(queries)} queries")
    print(f"✓ Tested 3 retrieval strategies × 3 LLMs = 9 combinations per query")
    print(f"✓ Total: {len(queries) * 3 * 3} retrieval + generation operations")
    print(f"✓ Report saved to: evaluation_report.txt\n")


if __name__ == "__main__":
    main()
