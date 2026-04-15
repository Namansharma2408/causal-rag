#!/bin/bash
set -e

echo "=== Causal AI RAG System Starting ==="

# Wait for MongoDB to be ready
echo "Waiting for MongoDB..."
max_retries=30
retry_count=0
until curl -s http://mongodb:27017 > /dev/null 2>&1 || [ $retry_count -eq $max_retries ]; do
    retry_count=$((retry_count + 1))
    echo "  Waiting for MongoDB... ($retry_count/$max_retries)"
    sleep 2
done

if [ $retry_count -eq $max_retries ]; then
    echo "Warning: MongoDB may not be ready, but continuing anyway..."
else
    echo "MongoDB is ready!"
fi

echo "=== Starting Uvicorn Server ==="
exec uvicorn server:app --host 0.0.0.0 --port 5000
