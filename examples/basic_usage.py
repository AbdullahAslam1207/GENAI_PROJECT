"""
Basic Usage Examples
Demonstrates various RAG pipeline features
"""

import sys
sys.path.append('..')

from rag_pipeline import RAGPipeline
from langchain.schema import Document
import logging

logging.basicConfig(level=logging.INFO)

print("="*70)
print("RAG Pipeline - Usage Examples")
print("="*70)

# Example 1: Basic Query
print("\n" + "="*70)
print("Example 1: Basic Query")
print("="*70)

pipeline = RAGPipeline()

# Assuming index is already built
try:
    pipeline.load_index("../data/indices")
    
    query = "What is machine learning?"
    response = pipeline.query(query)
    
    print(f"\nQuery: {query}")
    print(f"Answer: {response['answer']}")
    print(f"Time: {response['total_time_ms']:.2f}ms")
except Exception as e:
    print(f"Error: {e}")
    print("Please build the index first using: python main.py build")


# Example 2: Query with Context
print("\n" + "="*70)
print("Example 2: Query with Retrieved Context")
print("="*70)

try:
    query = "What are the types of machine learning?"
    response = pipeline.query(query, return_context=True)
    
    print(f"\nQuery: {query}")
    print(f"Answer: {response['answer']}")
    
    print(f"\nRetrieved Context ({len(response['context'])} documents):")
    for i, ctx in enumerate(response['context'][:2], 1):
        print(f"\n[{i}] Source: {ctx['metadata'].get('source', 'Unknown')}")
        print(f"Content: {ctx['content'][:200]}...")

except Exception as e:
    print(f"Error: {e}")


# Example 3: Conversational Query
print("\n" + "="*70)
print("Example 3: Conversational Query with History")
print("="*70)

try:
    # Simulated conversation
    chat_history = [
        {'role': 'user', 'content': 'What is machine learning?'},
        {'role': 'assistant', 'content': 'Machine learning is a subset of AI...'},
    ]
    
    query = "Can you give me more examples?"
    response = pipeline.query(query, chat_history=chat_history)
    
    print(f"\nConversation Context:")
    for turn in chat_history:
        print(f"  {turn['role'].capitalize()}: {turn['content'][:50]}...")
    
    print(f"\nNew Query: {query}")
    print(f"Answer: {response['answer']}")

except Exception as e:
    print(f"Error: {e}")


# Example 4: Batch Queries
print("\n" + "="*70)
print("Example 4: Batch Query Processing")
print("="*70)

try:
    queries = [
        "What is supervised learning?",
        "What is unsupervised learning?",
        "What is reinforcement learning?",
    ]
    
    responses = pipeline.batch_query(queries)
    
    for query, response in zip(queries, responses):
        print(f"\nQ: {query}")
        print(f"A: {response['answer'][:100]}...")
        print(f"Time: {response['total_time_ms']:.0f}ms")

except Exception as e:
    print(f"Error: {e}")


# Example 5: Pipeline Statistics
print("\n" + "="*70)
print("Example 5: Pipeline Statistics")
print("="*70)

try:
    stats = pipeline.get_stats()
    
    print("\nRetrieval Latency:")
    if stats.get('retrieval_latency'):
        lat_stats = stats['retrieval_latency']
        print(f"  Mean: {lat_stats.get('mean', 0):.2f}ms")
        print(f"  Median: {lat_stats.get('median', 0):.2f}ms")
        print(f"  P95: {lat_stats.get('p95', 0):.2f}ms")
    
    print("\nGeneration Latency:")
    if stats.get('generation_latency'):
        gen_stats = stats['generation_latency']
        print(f"  Mean: {gen_stats.get('mean', 0):.2f}ms")
        print(f"  Median: {gen_stats.get('median', 0):.2f}ms")
    
    print("\nIndex Statistics:")
    if 'dense_index' in stats:
        print(f"  Total vectors: {stats['dense_index']['total_vectors']}")
        print(f"  Documents: {stats['dense_index']['n_documents']}")
    elif 'index' in stats:
        print(f"  Total vectors: {stats['index']['total_vectors']}")
        print(f"  Documents: {stats['index']['n_documents']}")

except Exception as e:
    print(f"Error: {e}")


# Example 6: Custom Components
print("\n" + "="*70)
print("Example 6: Using Individual Components")
print("="*70)

try:
    from embeddings.encoder import EmbeddingEncoder
    from retrieval.dense import DenseRetriever
    
    # Create sample documents
    docs = [
        Document(page_content="Python is a programming language.", metadata={'id': 0}),
        Document(page_content="Machine learning uses algorithms.", metadata={'id': 1}),
        Document(page_content="Neural networks are used in deep learning.", metadata={'id': 2}),
    ]
    
    # Create encoder and retriever
    encoder = EmbeddingEncoder(model_name="all-MiniLM-L6-v2")
    retriever = DenseRetriever(encoder)
    
    # Build index
    retriever.build_index(docs, index_type="Flat")
    
    # Query
    query = "What is machine learning?"
    results = retriever.retrieve(query, k=2)
    
    print(f"\nQuery: {query}")
    print(f"Results:")
    for doc, score in results:
        print(f"  Score: {score:.4f} - {doc.page_content}")

except Exception as e:
    print(f"Error: {e}")


print("\n" + "="*70)
print("Examples completed!")
print("="*70)
print("\nFor more examples, check the documentation or run:")
print("  python main.py query --help")