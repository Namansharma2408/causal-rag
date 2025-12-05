# FinalAgent - Production-Ready Multi-Agent RAG System

A modular, production-ready RAG (Retrieval-Augmented Generation) system with multi-agent architecture.

## Features

- **Hybrid Search**: Combines vector similarity (60%) with BM25 text matching (40%)
- **Multi-Hop Reasoning**: Automatically decomposes complex queries into sub-queries
- **Multi-Agent Pipeline**: Decomposer → Router → Retriever → Reranker → Extractor → Reasoner → Quality
- **Evidence Verification**: ProofAgent verifies answers against full transcripts
- **Session Memory**: Persistent conversation history across sessions
- **Ollama Integration**: Uses local LLMs (codellama:7b, qwen2.5-coder:7b)

## Installation

```bash
pip install pymongo httpx numpy python-dotenv
```

## Quick Start

### Python API

```python
from finalAgent import answer_question, get_evidence

# Simple question (processed directly)
answer = answer_question("What is your refund policy?")
print(answer)

# Complex question (automatically decomposed into sub-queries)
answer = answer_question("Compare churn triggers vs loyalty decline factors")
print(answer)

# Get evidence for last answer
evidence = get_evidence()
print(evidence)
```

### With Full Result

```python
from finalAgent import RAGSystem

rag = RAGSystem()
answer = rag.answer("How do I cancel my order?", include_proof=True)

result = rag.get_last_result()
print(f"Answer: {result.answer}")
print(f"Quality: {result.quality_score}/100")
print(f"Sources: {result.transcript_ids}")
print(f"Multi-hop: {result.metadata.get('multihop', False)}")
```

### Command Line

```bash
# Interactive mode
python -m finalAgent

# Single question
python -m finalAgent -q "What is your return policy?"

# Complex query (auto-decomposed)
python -m finalAgent -q "Compare churn vs loyalty decline and explain procedural friction"

# With evidence verification
python -m finalAgent -q "How do I get a refund?" --proof
```

## Multi-Hop Reasoning

The system intelligently handles query complexity:

### Simple Queries (Direct Processing)
```
"What causes customer churn?" → Single-hop, direct pipeline
"Why do customers leave?" → Single-hop, direct pipeline
```

### Complex Queries (Auto-Decomposition)
```
"Compare X vs Y and explain Z" → Decomposed into 3-4 sub-queries
"What are all factors for A and how do they relate to B?" → Multi-hop
```

**Complexity Detection:**
- Comparison requests ("compare", "versus", "difference between")
- Multiple questions combined ("and also", "as well as")
- Multi-topic synthesis requirements
- Queries with 2+ distinct question marks

The decomposer only activates when necessary (complexity score ≥ 7/10).

## Architecture

```
finalAgent/
├── __init__.py          # Package exports
├── config.py            # Configuration & logging
├── models.py            # Data models (Query, Document, RAGResult)
├── rag_system.py        # Main orchestrator & RAGSystem class
├── api.py               # Simple functional API
├── cli.py               # Command-line interface
├── services/
│   ├── embedding.py     # Ollama embeddings (nomic-embed-text)
│   ├── ollama.py        # LLM service (fast & quality models)
│   ├── mongodb.py       # MongoDB with hybrid search
│   ├── memory.py        # Conversation persistence
│   └── transcripts.py   # Full transcript access
└── agents/
    ├── base.py          # BaseAgent abstract class
    ├── decomposer.py    # Multi-hop query decomposition
    ├── router.py        # Query classification
    ├── retriever.py     # Document retrieval
    ├── reranker.py      # Relevance reranking
    ├── extractor.py     # Information extraction
    ├── reasoner.py      # Answer generation
    ├── proof.py         # Evidence verification
    └── quality.py       # Quality scoring
```

## Configuration

Set via environment variables or modify `config.py`:

| Variable | Default | Description |
|----------|---------|-------------|
| MONGODB_URI | localhost:27017 | MongoDB connection |
| OLLAMA_BASE_URL | localhost:11434 | Ollama server |
| LOG_LEVEL | INFO | Logging level |

## MongoDB Setup

Requires two databases:

1. `conversations_db.spans` - Document embeddings with clusters
2. `full_transcripts.transcripts` - Full conversation transcripts

## Models Used

- **Embeddings**: `nomic-embed-text:latest` (768-dim)
- **Fast Model**: `codellama:7b` (routing, reranking)
- **Quality Model**: `qwen2.5-coder:7b` (answers, extraction)

## API Reference

### answer_question(question, session_id=None)
Simple question answering.

### answer_with_proof(question, session_id=None)
Returns dict with answer, evidence, quality_score.

### get_evidence(session_id=None)
Get evidence for last answer.

### get_conversation(session_id=None)
Get conversation history.

## License

MIT
