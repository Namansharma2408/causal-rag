# Causal AI RAG System

Multi-Agent RAG system for intelligent question answering over conversation transcripts.

## 🚀 Quick Start (One Command)

```bash
# 1. Add your Gemini API key
echo "GEMINI_API_KEY=your_key_here" > .env

# 2. Build and run
make build && make up
```

**Open:** http://localhost:5000

## 📋 Commands

| Command | Description |
|---------|-------------|
| `make build` | Build Docker image |
| `make up` | Start application |
| `make down` | Stop application |
| `make logs` | View logs |

## ⚙️ Requirements

- Docker
- Gemini API Key
- (Optional) Ollama for ThinkingMode

## 🏗️ Architecture

**Multi-Agent Pipeline:** Query → Decomposer → Router → Retriever → Reranker → Extractor → Reasoner → ProofAgent → QualityAgent → Answer

## 📊 Models

- **Gemini 2.0 Flash** - Main reasoning
- **Ollama** (phi3:14b, deepseek-r1:14b, qwen2.5-coder:7b, codellama:7b) - ThinkingMode

---
*InterIIT Tech Meet Submission*
