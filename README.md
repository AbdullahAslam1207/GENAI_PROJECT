# RAG Chatbot - Retrieval-Augmented Generation System

A complete implementation of a Retrieval-Augmented Generation (RAG) chatbot using vector search, combining dense/sparse retrieval, and Llama 3.1 via Groq API.

## Features

- **Multiple Retrieval Strategies**: Dense, BM25, Hybrid, and Reranking
- **Flexible Document Processing**: Supports PDF, HTML, TXT files
- **Vector Search**: FAISS-based indexing with multiple index types
- **LLM Integration**: Groq API with Llama 3.1 8B
- **Evaluation Metrics**: Recall@k, MRR, nDCG, EM, F1
- **Modular Architecture**: Easy to extend and customize
- **CPU-Optimized**: No GPU required

## Project Structure

```
rag_chatbot/
├── config.yaml              # Configuration file
├── .env.example            # Environment variables template
├── requirements.txt        # Python dependencies
├── main.py                # Main entry point
├── rag_pipeline.py        # Pipeline orchestration
├── data/
│   ├── ingestion.py       # Document loaders
│   └── chunking.py        # Text chunking strategies
├── embeddings/
│   └── encoder.py         # Embedding generation
├── indexing/
│   └── faiss_index.py     # FAISS index management
├── retrieval/
│   ├── dense.py           # Dense retrieval
│   ├── bm25.py            # BM25 retrieval
│   ├── hybrid.py          # Hybrid retrieval
│   └── reranker.py        # Cross-encoder reranking
├── generation/
│   ├── llm_wrapper.py     # LLM interface
│   └── prompt_templates.py # Prompt templates
└── evaluation/
    └── metrics.py         # Evaluation metrics
```

## Installation

### 1. Clone and Setup

```bash
# Create project directory
mkdir rag_chatbot && cd rag_chatbot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Configuration

Create a `.env` file from the example:

```bash
cp .env.example .env
```

Edit `.env` and add your Groq API key:

```
GROQ_API_KEY=your_groq_api_key_here
```

Get your free Groq API key from: https://console.groq.com/

### 3. Prepare Documents

Place your documents in the `data/documents/` directory:

```bash
mkdir -p data/documents
# Copy your PDF, HTML, or TXT files here
```

## Usage

### Build the Index

First, build the retrieval index from your documents:

```bash
python main.py build
```

This will:
- Load all documents from `data/documents/`
- Chunk them into passages
- Generate embeddings
- Build the vector index
- Save the index to `data/indices/`

### Query the System

#### Interactive Mode

Start an interactive chat session:

```bash
python main.py query
```

With conversational history:

```bash
python main.py query --conversational
```

#### Single Query Mode

Ask a single question:

```bash
python main.py query -q "What is machine learning?"
```

Show retrieved context:

```bash
python main.py query -q "What is RAG?" --show-context
```

## Configuration

Edit `config.yaml` to customize the system:

### Key Configuration Options

```yaml
# Chunking
chunking:
  strategy: "recursive"  # recursive, sliding_window, semantic
  chunk_size: 512
  chunk_overlap: 128

# Embeddings
embedding:
  model_name: "all-MiniLM-L6-v2"  # Fast and efficient
  # Alternatives: "all-mpnet-base-v2", "sentence-t5-base"

# Retrieval
retrieval:
  strategy: "hybrid"  # dense, hybrid, rerank
  top_k: 20
  hybrid_alpha: 0.7  # 0.7 = 70% dense, 30% BM25

# Reranking
reranking:
  enabled: true
  model_name: "cross-encoder/ms-marco-MiniLM-L-6-v2"
  
# LLM
llm:
  model_name: "llama-3.1-8b-instant"
  temperature: 0.1
  max_tokens: 1024
```

### Retrieval Strategies

1. **Dense**: Pure vector similarity search (fastest)
2. **Hybrid**: Combines dense + BM25 lexical search (balanced)
3. **Rerank**: Hybrid + cross-encoder reranking (most accurate)

## Examples

### Example 1: Build and Query

```python
from rag_pipeline import RAGPipeline

# Initialize
pipeline = RAGPipeline()

# Build index
pipeline.build_index()

# Query
response = pipeline.query("What is retrieval-augmented generation?")
print(response['answer'])
```

### Example 2: Custom Configuration

```python
# Create custom config
config = {
    'chunking': {'chunk_size': 256, 'chunk_overlap': 64},
    'retrieval': {'strategy': 'rerank', 'top_k': 10},
    # ... other settings
}

# Initialize with custom config
pipeline = RAGPipeline(config_path='my_config.yaml')
```

### Example 3: Batch Processing

```python
questions = [
    "What is machine learning?",
    "How does RAG work?",
    "What are embeddings?"
]

responses = pipeline.batch_query(questions)
for q, r in zip(questions, responses):
    print(f"Q: {q}\nA: {r['answer']}\n")
```

## Performance Optimization

### Memory Usage

- Use `IVFFlat` index for large datasets (>100K documents)
- Reduce `chunk_size` if memory limited
- Use smaller embedding model (`all-MiniLM-L6-v2`)

### Speed

- Use `Flat` index for exact search on small datasets (<10K docs)
- Reduce `top_k` for faster retrieval
- Disable reranking if not needed

### Quality

- Enable reranking for best accuracy
- Use hybrid retrieval (dense + BM25)
- Increase `chunk_overlap` for better context

## Evaluation

Evaluate on standard benchmarks:

```bash
python main.py eval --dataset naturalquestions
```

Available metrics:
- **Retrieval**: Recall@k, MRR, nDCG
- **QA**: Exact Match, F1 Score
- **Latency**: Response time, throughput

## API Integration

### Groq Models Available

- `llama-3.1-8b-instant` (default, fastest)
- `llama-3.1-70b-versatile` (higher quality)
- `mixtral-8x7b-32768` (large context)

Change in `config.yaml`:

```yaml
llm:
  model_name: "llama-3.1-70b-versatile"
```

## Troubleshooting

### Common Issues

1. **No documents found**
   - Ensure documents are in `data/documents/`
   - Check file formats (PDF, HTML, TXT supported)

2. **API key error**
   - Verify GROQ_API_KEY in `.env`
   - Check key validity at console.groq.com

3. **Memory errors**
   - Reduce `chunk_size` in config
   - Use smaller embedding model
   - Process documents in batches

4. **Slow retrieval**
   - Use `IVFFlat` index instead of `Flat`
   - Reduce `top_k`
   - Disable reranking

## Advanced Usage

### Custom Prompt Templates

```python
from generation.prompt_templates import PromptTemplates

custom_template = """
You are an expert in {domain}.
Use the context below to answer.

Context: {context}
Question: {question}
Answer:
"""

response = pipeline.llm.generate_with_context(
    query="What is X?",
    context_documents=docs,
    template=custom_template
)
```

### Adding New Documents

```python
# Add to existing index
from data.ingestion import DocumentIngestion

new_docs = ingestion.load_document("new_file.pdf")
chunked = chunker.chunk_documents(new_docs)
pipeline.retriever.add_documents(chunked)
pipeline.save_index("data/indices")
```

## Contributing

Contributions welcome! Areas for improvement:

- Additional document formats (DOCX, Markdown)
- More embedding models
- Advanced reranking strategies
- Streaming responses
- Web UI interface

## References

Based on research from the proposal:
- RAG (Lewis et al. 2020)
- DPR (Karpukhin et al. 2020)
- ColBERT (Khattab & Zaharia 2020)
- FAISS (Johnson et al. 2017)
- BEIR Benchmark (Thakur et al. 2021)

## License

MIT License - see LICENSE file

## Support

For issues and questions:
- Check the troubleshooting section
- Review configuration options
- Open an issue on GitHub

---

**Authors**: Abdullah Aslam, Fakhir Ali, Ayna Sulaiman  
**Course**: GenAI Project  
**Instructor**: Dr. Akhtar Jamil