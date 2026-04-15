#!/bin/bash
set -e

echo "=== Causal AI RAG System Starting ==="

MONGO_WAIT_HOST="${MONGODB_HOST:-mongodb}"
MONGO_WAIT_PORT="${MONGODB_PORT:-27017}"

# Wait for MongoDB to be ready
echo "Waiting for MongoDB at ${MONGO_WAIT_HOST}:${MONGO_WAIT_PORT}..."
max_retries=30
retry_count=0
until curl -s "http://${MONGO_WAIT_HOST}:${MONGO_WAIT_PORT}" > /dev/null 2>&1 || [ $retry_count -eq $max_retries ]; do
    retry_count=$((retry_count + 1))
    echo "  Waiting for MongoDB... ($retry_count/$max_retries)"
    sleep 2
done

if [ $retry_count -eq $max_retries ]; then
    echo "Warning: MongoDB may not be ready, but continuing anyway..."
else
    echo "MongoDB is ready!"
fi

# If a custom command is provided, run it instead of starting the API server.
if [ "$#" -gt 0 ]; then
    echo "=== Running custom command: $* ==="
    exec "$@"
fi

echo "=== Starting Uvicorn Server ==="
exec uvicorn server:app --host 0.0.0.0 --port 5000
