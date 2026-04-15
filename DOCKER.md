# Docker Deployment Guide

This guide explains how to deploy the Causal AI RAG System using Docker.

## 📋 Prerequisites

1. **Docker** - [Install Docker](https://docs.docker.com/get-docker/)
2. **Gemini API Key** - [Get API Key](https://makersuite.google.com/app/apikey)
3. **Ollama** (optional, for ThinkingMode) - [Install Ollama](https://ollama.ai/)

## 🚀 Quick Start

### Step 1: Create Environment File

```bash
# Create .env file with your API key
echo "GEMINI_API_KEY=your_api_key_here" > .env
```

### Step 2: Build Docker Image

```bash
make build
```

This builds a multi-stage Docker image (~1.5GB) with:
- Python 3.11 runtime
- All Python dependencies
- Application code
- Entrypoint script

### Step 3: Start Containers

```bash
make up
```

This starts:
- **mongodb** - MongoDB 7.0 database on port 27017
- **rag-app** - RAG application on port 5000

### Step 4: Access Application

Open your browser: **http://localhost:5000**

### Step 5: Stop Containers

```bash
make down
```

## 📦 Docker Commands Reference

### Basic Operations

| Command | Description |
|---------|-------------|
| `make build` | Build the Docker image |
| `make up` | Start all containers |
| `make down` | Stop all containers |
| `make restart` | Restart the application |
| `make status` | Show container status |

### Logs & Debugging

| Command | Description |
|---------|-------------|
| `make logs` | View application logs (follow mode) |
| `make logs-mongo` | View MongoDB logs |
| `make shell` | Open bash shell in app container |

### Cleanup

| Command | Description |
|---------|-------------|
| `make clean` | Stop and remove containers/images |
| `make clean-all` | Clean everything including MongoDB data |

### Help

```bash
make help
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# Required - Your Gemini API key
GEMINI_API_KEY=your_gemini_api_key_here

# Optional - Customize these if needed
MONGODB_HOST=mongodb
MONGODB_PORT=27017
OLLAMA_BASE_URL=http://host.docker.internal:11434
LOG_LEVEL=INFO
```

### Using Local Ollama (for ThinkingMode)

ThinkingMode requires Ollama running on your host machine:

```bash
# 1. Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# 2. Pull required models
ollama pull phi3:14b
ollama pull deepseek-r1:14b
ollama pull qwen2.5-coder:7b
ollama pull codellama:7b
ollama pull nomic-embed-text

# 3. Start Ollama
ollama serve
```

The Docker container connects to your host's Ollama via `host.docker.internal:11434`.

## 🏗️ Architecture

### Container Network

```
┌─────────────────────────────────────────┐
│               rag-network               │
│                                         │
│  ┌─────────────┐     ┌─────────────┐   │
│  │   rag-app   │────▶│   mongodb   │   │
│  │  (port 5000)│     │ (port 27017)│   │
│  └─────────────┘     └─────────────┘   │
│         │                               │
└─────────┼───────────────────────────────┘
          │
          ▼
    ┌─────────────┐
    │   ollama    │  (host machine)
    │ (port 11434)│
    └─────────────┘
```

### Image Details

The Docker image is built in two stages:

1. **Builder Stage**: Installs Python dependencies
2. **Production Stage**: Copies only necessary files

```dockerfile
# Multi-stage build reduces final image size
FROM python:3.11-slim as builder
# ... install dependencies

FROM python:3.11-slim as production
# ... copy only what's needed
```

## 🔍 Troubleshooting

### Container won't start

```bash
# Check logs
make logs

# Check container status
make status
```

### MongoDB connection issues

```bash
# Verify MongoDB is running
sudo docker ps | grep mongodb

# Check MongoDB logs
make logs-mongo

# Test MongoDB connectivity
sudo docker exec mongodb mongosh --eval 'db.adminCommand("ping")'
```

### Port already in use

```bash
# Check what's using port 5000
sudo lsof -i :5000

# Or use a different port (edit Makefile)
```

### Ollama not connecting

```bash
# Verify Ollama is running
curl http://localhost:11434/api/tags

# Check if models are available
ollama list
```

### Reset everything

```bash
# Complete cleanup and fresh start
make clean-all
make build
make up
```

## 📂 Volume Mounts

| Host Path | Container Path | Purpose |
|-----------|----------------|---------|
| `./finalAgent/mongo_data` | `/app/mongo_data` | Pre-loaded transcript data |
| `./chat_sessions` | `/app/chat_sessions` | Conversation history |
| `mongodb_data` (Docker volume) | `/data/db` | MongoDB persistent storage |

## 🔐 Security Notes

- The `.env` file contains your API key - **do not commit to git**
- MongoDB is exposed on port 27017 - consider firewall rules in production
- For production, add authentication to MongoDB
- Consider using Docker secrets for sensitive data

## 📊 Resource Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| RAM | 4GB | 8GB+ |
| CPU | 2 cores | 4+ cores |
| Disk | 5GB | 10GB+ |

## 🔄 Updating

```bash
# Pull latest code
git pull

# Rebuild image
make build

# Restart with new image
make restart
```
