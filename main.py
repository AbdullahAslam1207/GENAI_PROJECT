"""
Main Entry Point for RAG Chatbot
Interactive command-line interface
"""

import os
import sys
import argparse
import logging
from pathlib import Path

from rag_pipeline import RAGPipeline
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/rag_chatbot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def build_index_command(args):
    """Build the retrieval index"""
    logger.info("Building index...")
    
    pipeline = RAGPipeline(config_path=args.config)
    pipeline.build_index(
        documents_dir=args.documents_dir,
        save_index=True,
    )
    
    print("\n✓ Index built successfully!")
    
    # Show stats
    stats = pipeline.get_stats()
    print(f"\nIndex Statistics:")
    if 'index' in stats:
        print(f"  Total vectors: {stats['index']['total_vectors']}")
        print(f"  Documents: {stats['index']['n_documents']}")


def query_command(args):
    """Query the RAG system"""
    # Initialize pipeline
    pipeline = RAGPipeline(config_path=args.config)
    
    # Load index
    index_dir = pipeline.config['datasets']['index_dir']
    if Path(index_dir).exists():
        logger.info("Loading existing index...")
        pipeline.load_index(index_dir)
    else:
        print("No index found. Please build the index first using --build")
        return
    
    # Single query mode
    if args.query:
        response = pipeline.query(args.query, return_context=args.show_context)
        
        print(f"\nQuestion: {args.query}")
        print(f"\nAnswer: {response['answer']}")
        print(f"\nMetrics:")
        print(f"  Retrieval time: {response['retrieval_time_ms']:.2f}ms")
        print(f"  Generation time: {response['generation_time_ms']:.2f}ms")
        print(f"  Total time: {response['total_time_ms']:.2f}ms")
        print(f"  Documents retrieved: {response['num_docs_retrieved']}")
        
        if args.show_context:
            print(f"\nContext:")
            for i, ctx in enumerate(response['context'][:3], 1):
                print(f"\n[{i}] {ctx['content'][:200]}...")
    
    # Interactive mode
    else:
        print("\n" + "="*60)
        print("RAG Chatbot - Interactive Mode")
        print("Type 'quit' or 'exit' to end the session")
        print("="*60 + "\n")
        
        chat_history = []
        
        while True:
            try:
                question = input("\nYou: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("\nGoodbye!")
                    break
                
                if not question:
                    continue
                
                # Query pipeline
                response = pipeline.query(
                    question,
                    chat_history=chat_history if args.conversational else None
                )
                
                # Print answer
                print(f"\nAssistant: {response['answer']}")
                print(f"\n[Time: {response['total_time_ms']:.0f}ms | "
                      f"Docs: {response['num_docs_retrieved']}]")
                
                # Update chat history
                if args.conversational:
                    chat_history.append({'role': 'user', 'content': question})
                    chat_history.append({'role': 'assistant', 'content': response['answer']})
                    
                    # Keep only last 5 turns
                    if len(chat_history) > 10:
                        chat_history = chat_history[-10:]
            
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                logger.error(f"Error processing query: {e}")
                print(f"\nError: {str(e)}")


def eval_command(args):
    """Evaluate the RAG system"""
    print("Evaluation mode - Coming soon!")
    # TODO: Implement evaluation on benchmark datasets


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="RAG Chatbot - Retrieval-Augmented Generation System"
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Build command
    build_parser = subparsers.add_parser('build', help='Build the retrieval index')
    build_parser.add_argument(
        '--documents-dir',
        type=str,
        help='Directory containing documents (overrides config)'
    )
    
    # Query command
    query_parser = subparsers.add_parser('query', help='Query the RAG system')
    query_parser.add_argument(
        '-q', '--query',
        type=str,
        help='Single query (if not provided, enters interactive mode)'
    )
    query_parser.add_argument(
        '--show-context',
        action='store_true',
        help='Show retrieved context'
    )
    query_parser.add_argument(
        '--conversational',
        action='store_true',
        help='Enable conversational mode with history'
    )
    
    # Eval command
    eval_parser = subparsers.add_parser('eval', help='Evaluate the system')
    eval_parser.add_argument(
        '--dataset',
        type=str,
        help='Evaluation dataset'
    )
    
    args = parser.parse_args()
    
    # Create necessary directories
    Path('logs').mkdir(exist_ok=True)
    Path('data/documents').mkdir(parents=True, exist_ok=True)
    Path('data/indices').mkdir(parents=True, exist_ok=True)
    
    # Check for API key
    if not os.getenv('GROQ_API_KEY'):
        print("\n⚠ WARNING: GROQ_API_KEY not found in environment variables!")
        print("Please set it in your .env file or export it:")
        print("  export GROQ_API_KEY='your_api_key_here'\n")
        if args.command != 'build':
            sys.exit(1)
    
    # Execute command
    if args.command == 'build':
        build_index_command(args)
    elif args.command == 'query':
        query_command(args)
    elif args.command == 'eval':
        eval_command(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()