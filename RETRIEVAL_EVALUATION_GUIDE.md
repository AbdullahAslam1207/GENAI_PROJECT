# Retrieval Methods Evaluation Guide

## Project Overview

This is a **RAG (Retrieval-Augmented Generation) Chatbot** that processes PDF documents and answers questions using various retrieval strategies. The project supports multiple retrieval methods for finding relevant information from documents.

## Key Components

### 1. **Document Processing**
- **Location**: `data/documents/` (contains `vision transformer.pdf`)
- **Ingestion**: Loads and preprocesses PDFs
- **Chunking**: Splits documents into manageable chunks (512 tokens with 128 overlap)

### 2. **Retrieval Strategies** (defined in `config.yaml`)
The project implements three retrieval methods:

#### a) **Dense Retrieval**
- Uses semantic embeddings (all-MiniLM-L6-v2)
- FAISS vector search
- Best for: Semantic similarity

#### b) **Hybrid Retrieval**
- Combines dense embeddings + BM25 (keyword-based)
- Weighted fusion (70% dense, 30% BM25 by default)
- Best for: Balanced semantic + keyword matching

#### c) **Rerank Strategy**
- Starts with hybrid retrieval
- Applies cross-encoder reranking (ms-marco-MiniLM-L-6-v2)
- Retrieves top 20, reranks to top 5
- Best for: Maximum precision

### 3. **Pipeline Architecture**
```
PDF → Chunking → Embeddings → Indexing → Retrieval → (Optional: Reranking) → LLM Generation
```

## The Evaluation Script

### File: `evaluate_retrievers.py`

This script systematically evaluates all three retrieval strategies turn-by-turn on test queries.

### What It Does:

1. **Builds Index** for each strategy (dense, hybrid, rerank)
2. **Retrieves Documents** using each method
3. **Measures Performance**:
   - Number of documents retrieved
   - Retrieval time (milliseconds)
   - Relevance scores
   - Top document previews

4. **Compares Results** in a side-by-side comparison table

5. **Saves Results** to `retrieval_evaluation_results.txt`

### Current Test Query:
```
"What is the key innovation of Vision Transformers compared to CNNs?"
```

## How to Run

### Step 1: Ensure Virtual Environment is Active
```powershell
.venv\Scripts\Activate.ps1
```

### Step 2: Run the Evaluation Script
```powershell
python evaluate_retrievers.py
```

### Expected Output:
```
================================================================================
EVALUATING QUERY: What is the key innovation of Vision Transformers compared to CNNs?
================================================================================

Building index for strategy: DENSE
...

Building index for strategy: HYBRID
...

Building index for strategy: RERANK
...

================================================================================
COMPARISON SUMMARY
================================================================================

Strategy        Docs     Time (ms)    Top Score    Avg Top-5
--------------------------------------------------------------------------------
DENSE           20       45.23        0.8543       0.7821
HYBRID          20       67.89        0.8721       0.8012
RERANK          5        123.45       0.9234       0.8934

================================================================================
```

## Adding More Queries (Future Enhancement)

To evaluate multiple queries, modify the `main()` function in `evaluate_retrievers.py`:

```python
def main():
    evaluator = RetrieverEvaluator(config_path="config.yaml")
    
    # Add multiple test queries
    test_queries = [
        "What is the key innovation of Vision Transformers compared to CNNs?",
        "How does the attention mechanism work in Vision Transformers?",
        "What are the computational requirements of ViT?",
        "How does ViT perform on image classification tasks?",
        "What is the patch embedding process in Vision Transformers?"
    ]
    
    # Evaluate each query
    for query in test_queries:
        evaluator.evaluate_all_strategies(query)
    
    evaluator.save_results()
```

## Configuration (config.yaml)

Key settings you can adjust:

```yaml
retrieval:
  strategy: "hybrid"  # Options: dense, hybrid, rerank
  top_k: 20           # Number of documents to retrieve
  rerank_top_n: 5     # Final number after reranking
  hybrid_alpha: 0.7   # Weight: 70% dense, 30% BM25

chunking:
  chunk_size: 512     # Token size per chunk
  chunk_overlap: 128  # Overlap between chunks

embedding:
  model_name: "all-MiniLM-L6-v2"  # Embedding model
```

## Understanding the Results

### Metrics Explained:

1. **Docs Retrieved**: Number of documents returned by the retriever
2. **Time (ms)**: How long the retrieval took
3. **Top Score**: Highest relevance score (higher is better)
4. **Avg Top-5**: Average score of top 5 results (consistency indicator)

### Typical Performance Patterns:

- **Dense**: Fast, good semantic understanding
- **Hybrid**: Slightly slower, better keyword + semantic matching
- **Rerank**: Slowest but highest precision (best top results)

## Next Steps

1. ✅ **Current**: Single query evaluation across all retrieval methods
2. ⏭️ **Next**: Add 5-10 test queries for comprehensive evaluation
3. ⏭️ **Future**: Add evaluation metrics (Recall@K, MRR, NDCG)
4. ⏭️ **Future**: Compare with LLM generation quality

## Troubleshooting

### Issue: "No index found"
**Solution**: The script builds indices automatically, but ensure the PDF exists in `data/documents/`

### Issue: "GROQ_API_KEY not found"
**Solution**: Only needed for LLM generation. Retrieval evaluation works without it.

### Issue: Low relevance scores
**Solution**: Check:
- PDF was loaded correctly
- Chunk size isn't too small/large
- Query matches document content

## Files Created

- ✅ `evaluate_retrievers.py` - Main evaluation script
- ✅ `RETRIEVAL_EVALUATION_GUIDE.md` - This guide
- 📄 `retrieval_evaluation_results.txt` - Generated after running

---

**Ready to run!** Execute `python evaluate_retrievers.py` to see the comparison.
