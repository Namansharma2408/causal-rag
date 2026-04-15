# Causal AI - Multi-Agent RAG System

A production-ready Multi-Agent Retrieval-Augmented Generation (RAG) system for intelligent question answering over conversation transcripts.

---

## 🚀 Quick Start (Docker)

```bash
# 1. Add your Gemini API key
echo "GEMINI_API_KEY=your_key_here" > .env

# 2. Build and run
make build && make up

# 3. Open browser: http://localhost:5000
```

**Stop:** `make down`

---

## 📋 Prerequisites

| Requirement | Version | Purpose |
|-------------|---------|---------|
| Docker | 20.0+ | Container runtime |
| Gemini API Key | - | Main LLM provider |
| Ollama (optional) | Latest | ThinkingMode (multi-model) |

### Get Gemini API Key
1. Go to https://makersuite.google.com/app/apikey
2. Create a new API key
3. Add to `.env` file

### Ollama Setup (Optional - for ThinkingMode)
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull required models
ollama pull phi3:14b
ollama pull deepseek-r1:14b
ollama pull qwen2.5-coder:7b
ollama pull codellama:7b
ollama pull nomic-embed-text

# Start Ollama
ollama serve
```

---

## 🏗️ System Architecture

### Multi-Agent Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER QUERY                                │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│ 1. DECOMPOSER AGENT                                              │
│    • Analyzes query complexity (1-10 score)                      │
│    • Simple queries (score < 7) → Pass through                   │
│    • Complex queries → Break into sub-queries                    │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. ROUTER AGENT                                                  │
│    • Classifies query type (factual/analytical/procedural/etc)   │
│    • Determines optimal retrieval strategy                       │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. RETRIEVER AGENT                                               │
│    • Hybrid Search: Vector (60%) + BM25 (40%)                    │
│    • Uses nomic-embed-text embeddings (768-dim)                  │
│    • Retrieves top-k relevant documents                          │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. RERANKER AGENT                                                │
│    • Cross-encoder reranking for precision                       │
│    • Filters low-relevance documents                             │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│ 5. EXTRACTOR AGENT                                               │
│    • Extracts key information from documents                     │
│    • Identifies relevant quotes and facts                        │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│ 6. REASONER AGENT                                                │
│    • Generates comprehensive answer                              │
│    • Synthesizes information from all sources                    │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│ 7. PROOF AGENT                                                   │
│    • Verifies answer against original transcripts                │
│    • Prevents hallucination                                      │
│    • Provides evidence citations                                 │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│ 8. QUALITY AGENT                                                 │
│    • Scores answer quality (0-100)                               │
│    • Evaluates relevance, completeness, coherence                │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                     FINAL RESPONSE                               │
│    • Answer + Quality Score + Evidence                           │
└─────────────────────────────────────────────────────────────────┘
```

### ThinkingMode (Multi-Model Consensus)

When enabled, uses 4 different Ollama models for consensus-based reasoning:

```
┌─────────────────────────────────────────────────────────────────┐
│                    THINKING MODE PIPELINE                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │ phi3:14b │  │deepseek- │  │qwen2.5-  │  │codellama │        │
│  │ ANALYST  │  │r1:14b    │  │coder:7b  │  │:7b       │        │
│  │          │  │ CRITIC   │  │SYNTHESIZE│  │PRAGMATIST│        │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘        │
│       │             │             │             │                │
│       └─────────────┴──────┬──────┴─────────────┘                │
│                            │                                     │
│                     PEER REVIEW                                  │
│              (Each model reviews others)                         │
│                            │                                     │
│                            ▼                                     │
│                   SELECT WINNER                                  │
│              (Highest consensus score)                           │
│                            │                                     │
│                            ▼                                     │
│                 REFINED FINAL ANSWER                             │
│           (Winner incorporates feedback)                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
SumbissionMp5/
├── 🐳 Docker Files
│   ├── Dockerfile              # Multi-stage build (Python 3.11)
│   ├── docker-compose.yml      # Service orchestration
│   ├── Makefile               # Easy Docker commands
│   ├── .dockerignore          # Build exclusions
│   └── scripts/
│       ├── docker-entrypoint.sh  # Container startup
│       └── init_mongodb.py       # DB initialization
│
├── 📄 Configuration
│   ├── .env                   # API keys (create this)
│   ├── .env.docker            # Docker env template
│   ├── config.py              # App configuration
│   └── requirements.txt       # Python dependencies
│
├── 🌐 Web Application
│   ├── server.py              # FastAPI server (port 5000)
│   ├── frontend.html          # Web UI
│   └── app.py                 # App wrapper
│
├── 🧠 Core System
│   ├── rag_system.py          # RAG Orchestrator
│   ├── models.py              # Data models
│   ├── api.py                 # REST API
│   └── cli.py                 # Command-line interface
│
├── 🤖 Agents (agents/)
│   ├── base.py                # BaseAgent class
│   ├── decomposer.py          # Query decomposition
│   ├── router.py              # Query routing
│   ├── retriever.py           # Hybrid search
│   ├── reranker.py            # Document reranking
│   ├── extractor.py           # Information extraction
│   ├── reasoner.py            # Answer generation
│   ├── proof.py               # Evidence verification
│   ├── quality.py             # Quality scoring
│   └── thinking.py            # Multi-model consensus
│
├── ⚙️ Services (services/)
│   ├── llm_provider.py        # Unified LLM interface
│   ├── embedding.py           # Ollama embeddings
│   ├── mongodb.py             # Database operations
│   ├── memory.py              # Session memory
│   ├── transcripts.py         # Transcript access
│   └── ollama.py              # Ollama client
│
└── 📊 Data (finalAgent/mongo_data/)
    ├── conversations_db/
    │   ├── spans.json              # Document embeddings
    │   ├── cluster_centroids.json  # Cluster centers
    │   ├── cluster_summary.json    # Cluster metadata
    │   ├── keywords.json           # Keywords index
    │   ├── intent_summary.json     # Intent analysis
    │   ├── domain_summary.json     # Domain classification
    │   ├── event_type_summary.json # Event types
    │   └── causal_motif_summary.json # Causal patterns
    │
    └── full_transcripts/
        └── transcripts.json        # Complete conversations
```

---

## 🔧 Docker Commands

| Command | Description |
|---------|-------------|
| `make build` | Build Docker image |
| `make up` | Start MongoDB + App |
| `make down` | Stop all containers |
| `make restart` | Restart app container |
| `make logs` | View app logs |
| `make logs-mongo` | View MongoDB logs |
| `make status` | Container status |
| `make shell` | Shell into app container |
| `make clean` | Remove containers/images |
| `make clean-all` | Clean including data |

---

## 🔌 API Endpoints

### REST API

```bash
# Query endpoint
POST /api/query
{
  "message": "What causes customer churn?",
  "settings": {
    "thinking_mode": false,
    "include_proof": true
  }
}

# Response
{
  "answer": "...",
  "quality_score": 85,
  "evidence": [...],
  "metadata": {...}
}
```

### Python API

```python
from finalAgent import answer_question, RAGSystem

# Simple usage
answer = answer_question("What is your refund policy?")

# With full control
rag = RAGSystem()
result = rag.answer(
    "Compare churn triggers vs loyalty factors",
    thinking_mode=True,
    include_proof=True
)
print(f"Answer: {result.answer}")
print(f"Quality: {result.quality_score}/100")
```

---

## 🧪 Features

### 1. Hybrid Search
- **Vector Similarity (60%)**: Semantic matching using `nomic-embed-text` embeddings
- **BM25 Text Matching (40%)**: Keyword-based retrieval

### 2. Multi-Hop Reasoning
Complex queries automatically decomposed:
```
"Compare X vs Y and explain Z" 
    → Sub-query 1: "What is X?"
    → Sub-query 2: "What is Y?"  
    → Sub-query 3: "How does Z relate?"
    → Synthesized answer
```

### 3. Evidence Verification
ProofAgent validates every answer against original transcripts to prevent hallucination.

### 4. Quality Scoring

| Score | Level | Meaning |
|-------|-------|---------|
| 80-100 | Excellent | High confidence |
| 60-79 | Good | Reliable answer |
| 40-59 | Moderate | May need verification |
| 0-39 | Low | Consider rephrasing |

---

## 📊 Models Used

| Model | Provider | Purpose |
|-------|----------|---------|
| gemini-2.0-flash | Google | Main reasoning & generation |
| nomic-embed-text | Ollama | Embedding generation (768-dim) |
| phi3:14b | Ollama | ThinkingMode - Analyst |
| deepseek-r1:14b | Ollama | ThinkingMode - Critic |
| qwen2.5-coder:7b | Ollama | ThinkingMode - Synthesizer |
| codellama:7b | Ollama | ThinkingMode - Pragmatist |

---

## 🗄️ Database Schema

### MongoDB Collections

| Database | Collection | Description |
|----------|------------|-------------|
| conversations_db | spans | Document embeddings with cluster info |
| conversations_db | cluster_centroids | Cluster center vectors |
| conversations_db | keywords | Keyword index for BM25 |
| full_transcripts | transcripts | Complete conversation texts |

---

## 🔍 Troubleshooting

### Docker Issues
```bash
# Check container status
make status

# View logs
make logs

# Reset everything
make clean-all && make build && make up
```

### MongoDB Connection
```bash
# Test MongoDB
sudo docker exec mongodb mongosh --eval 'db.adminCommand("ping")'
```

### Ollama Not Working
```bash
# Check Ollama is running
curl http://localhost:11434/api/tags

# List available models
ollama list
```

---

## 📦 Submission Contents

| Item | Size | Required |
|------|------|----------|
| Source Code | ~500KB | ✅ |
| Docker Config | ~50KB | ✅ |
| mongo_data/ | ~650MB | ✅ (embeddings) |
| Documentation | ~100KB | ✅ |
| **Total** | **~670MB** | |

---

## 📝 License

MIT License

---

## 👥 Team

Developed for InterIIT Tech Meet
