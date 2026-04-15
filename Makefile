.PHONY: build up down logs shell clean status init-db network

# Variables
APP_IMAGE = causal-ai-rag
MONGODB_IMAGE = mongo:7.0
NETWORK = rag-network
APP_CONTAINER = rag-app
MONGO_CONTAINER = mongodb
MONGO_DATA = mongodb_data

# Extract API key from .env if it exists
GEMINI_API_KEY ?= $(shell grep -oP 'GEMINI_API_KEY=\K.*' .env 2>/dev/null || echo "")

# Create docker network
network:
	@sudo docker network inspect $(NETWORK) >/dev/null 2>&1 || sudo docker network create $(NETWORK)

# Build the application image
build:
	sudo docker build -t $(APP_IMAGE) .

# Start MongoDB container
mongodb: network
	@if [ -z "$$(sudo docker ps -q -f name=$(MONGO_CONTAINER))" ]; then \
		if [ -n "$$(sudo docker ps -aq -f name=$(MONGO_CONTAINER))" ]; then \
			sudo docker start $(MONGO_CONTAINER); \
		else \
			sudo docker run -d \
				--name $(MONGO_CONTAINER) \
				--network $(NETWORK) \
				-p 27017:27017 \
				-v $(MONGO_DATA):/data/db \
				$(MONGODB_IMAGE); \
		fi; \
	fi
	@echo "MongoDB started"

# Start the application
up: network mongodb
	@if [ -z "$$(sudo docker ps -q -f name=$(APP_CONTAINER))" ]; then \
		if [ -n "$$(sudo docker ps -aq -f name=$(APP_CONTAINER))" ]; then \
			sudo docker rm $(APP_CONTAINER) 2>/dev/null; \
		fi; \
		sudo docker run -d \
			--name $(APP_CONTAINER) \
			--network $(NETWORK) \
			-p 5000:5000 \
			-e MONGODB_HOST=$(MONGO_CONTAINER) \
			-e MONGODB_PORT=27017 \
			-e GEMINI_API_KEY=$(GEMINI_API_KEY) \
			-e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
			-v $$(pwd)/data/mongoData:/app/mongo_data:ro \
			-v $$(pwd)/chatSessions:/app/chatSessions \
			$(APP_IMAGE); \
	fi
	@echo ""
	@echo "==================================="
	@echo "Application is running!"
	@echo "Open: http://localhost:5000"
	@echo "==================================="

# Stop all containers
down:
	-sudo docker stop $(APP_CONTAINER) 2>/dev/null
	-sudo docker stop $(MONGO_CONTAINER) 2>/dev/null
	@echo "Containers stopped"

# View app logs
logs:
	sudo docker logs -f $(APP_CONTAINER)

# View MongoDB logs
logs-mongo:
	sudo docker logs -f $(MONGO_CONTAINER)

# Open shell in app container
shell:
	sudo docker exec -it $(APP_CONTAINER) /bin/bash

# Check container status
status:
	@echo "=== Running Containers ==="
	@sudo docker ps --filter "name=$(APP_CONTAINER)" --filter "name=$(MONGO_CONTAINER)"
	@echo ""
	@echo "=== Network ==="
	@sudo docker network inspect $(NETWORK) --format='{{.Name}}: {{range .Containers}}{{.Name}} {{end}}' 2>/dev/null || echo "Network not created"

# Clean up everything
clean:
	-sudo docker stop $(APP_CONTAINER) $(MONGO_CONTAINER) 2>/dev/null
	-sudo docker rm $(APP_CONTAINER) $(MONGO_CONTAINER) 2>/dev/null
	-sudo docker rmi $(APP_IMAGE) 2>/dev/null
	-sudo docker network rm $(NETWORK) 2>/dev/null
	@echo "Cleanup complete"

# Clean including data
clean-all: clean
	-sudo docker volume rm $(MONGO_DATA) 2>/dev/null
	@echo "All data cleaned"

# Initialize database with transcript data
init-db: mongodb
	@echo "Initializing MongoDB with transcript data..."
	sudo docker exec $(MONGO_CONTAINER) mongosh --eval 'db.adminCommand("ping")' >/dev/null
	sudo docker run --rm \
		--network $(NETWORK) \
		-e MONGODB_URI=mongodb://$(MONGO_CONTAINER):27017 \
		-v $$(pwd)/data/mongoData:/app/mongo_data:ro \
		-v $$(pwd)/scripts:/app/scripts:ro \
		$(APP_IMAGE) \
		python /app/scripts/initMongoDb.py

# Restart the app
restart:
	-sudo docker stop $(APP_CONTAINER) 2>/dev/null
	-sudo docker rm $(APP_CONTAINER) 2>/dev/null
	@$(MAKE) up

# Show help
help:
	@echo "Multi-Agent RAG Docker Commands"
	@echo "================================"
	@echo "make build      - Build the Docker image"
	@echo "make up         - Start MongoDB and app containers"
	@echo "make down       - Stop all containers"
	@echo "make restart    - Restart the app container"
	@echo "make logs       - View app container logs"
	@echo "make logs-mongo - View MongoDB container logs"
	@echo "make status     - Show container and network status"
	@echo "make shell      - Open shell in app container"
	@echo "make init-db    - Initialize MongoDB with data"
	@echo "make clean      - Stop and remove containers/images"
	@echo "make clean-all  - Clean including MongoDB data volume"
	@echo ""
	@echo "Prerequisites:"
	@echo "  - Docker installed"
	@echo "  - .env file with GEMINI_API_KEY"
	@echo "  - Ollama running on host (for thinking mode)"
