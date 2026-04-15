# Causal AI RAG System

Production-ready multi-agent RAG pipeline for conversational analytics over transcript data.

## Quick start

```bash
echo "GEMINI_API_KEY=your_key_here" > .env
make build && make up
```

Open: `http://localhost:5000`

Stop:

```bash
make down
```

## What this project does

- Hybrid retrieval: vector similarity + BM25-style keyword scoring.
- Multi-agent reasoning pipeline:
	- `Decomposer` → `Router` → `Retriever` → `Reranker` → `Extractor` → `Reasoner` → `Proof` → `QualityChecker`
- Optional Thinking Mode: multi-model consensus using Ollama.
- Session-aware chat with persisted `chatSessions/*.json`.

## Clear file structure

```text
SumbissionMp5/
├── server.py                # FastAPI server + routes
├── app.py                   # Frontend request/session orchestration
├── ragSystem.py             # Core orchestrator pipeline
├── config.py                # Runtime config (env + defaults)
├── models.py                # Shared dataclasses
├── frontend.html            # Web UI
│
├── agents/                  # Agent implementations
├── services/                # LLM, embedding, MongoDB, memory, transcript services
├── scripts/
│   ├── dockerEntrypoint.sh  # Container startup/wait logic
│   └── initMongoDb.py       # MongoDB bootstrap script
│
├── Dockerfile
├── docker-compose.yml
├── Makefile
└── requirements.txt
```

## Docker commands

| Command | Purpose |
|---|---|
| `make build` | Build app image |
| `make up` | Start MongoDB + app |
| `make down` | Stop containers |
| `make logs` | App logs |
| `make logs-mongo` | MongoDB logs |
| `make status` | Container/network status |
| `make clean` | Remove containers and image |
| `make clean-all` | Also remove MongoDB volume |

## Required environment

Create `.env`:

```bash
GEMINI_API_KEY=your_gemini_api_key_here
```

Optional overrides:

```bash
MONGODB_HOST=mongodb
MONGODB_PORT=27017
OLLAMA_BASE_URL=http://host.docker.internal:11434
LOG_LEVEL=INFO
```

## Thinking mode setup (optional)

If you use Thinking Mode, run Ollama on host and pull:

```bash
ollama pull phi3:14b
ollama pull deepseek-r1:14b
ollama pull qwen2.5-coder:7b
ollama pull codellama:7b
ollama pull nomic-embed-text
```

## API usage

```bash
curl -X POST http://localhost:5000/api/query \
	-H "Content-Type: application/json" \
	-d '{"message":"What causes customer churn?","settings":{"thinking_mode":false,"include_proof":true}}'
```

## Troubleshooting

- Container health/logs:

```bash
make status
make logs
```

- MongoDB check:

```bash
sudo docker exec mongodb mongosh --eval 'db.adminCommand("ping")'
```

- Full reset:

```bash
make clean-all
make build
make up
```

---
InterIIT Tech Meet submission
