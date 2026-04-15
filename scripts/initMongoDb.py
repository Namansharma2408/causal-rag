#!/usr/bin/env python3
"""
MongoDB Data Initialization Script
Loads conversation and transcript data into MongoDB from JSON files.
"""

import json
import os
import importlib
from pathlib import Path

try:
    from bson import json_util  # type: ignore
except Exception:
    json_util = None


def _resolve_data_dir() -> Path:
    """Resolve data directory from common host/container locations."""
    candidates = []

    env_path = os.getenv("MONGO_DATA_DIR", "").strip()
    if env_path:
        candidates.append(Path(env_path))

    candidates.extend([
        Path("/app/mongo_data"),
        Path(__file__).resolve().parents[1] / "data" / "mongoData",
    ])

    for path in candidates:
        if path.exists():
            return path
    return Path("/app/mongo_data")


def _load_json(file_path: Path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Prefer Mongo Extended JSON parser so fields like {"$oid": ...} work.
    if json_util is not None:
        try:
            return json_util.loads(text)
        except Exception:
            pass

    # Fallback for plain JSON files.
    return json.loads(text)


def _insert_collection(db, collection_name: str, payload) -> int:
    """Insert list/dict payload safely and return inserted count."""
    if payload is None:
        return 0
    if isinstance(payload, list):
        if not payload:
            return 0
        db[collection_name].insert_many(payload, ordered=False)
        return len(payload)
    if isinstance(payload, dict):
        db[collection_name].insert_one(payload)
        return 1
    return 0


def initMongoDb():
    """Initialize MongoDB with data from JSON files."""
    
    # Get MongoDB URI from environment
    mongo_uri = os.getenv('MONGODB_URI', 'mongodb://localhost:27017')
    
    print(f"Connecting to MongoDB at {mongo_uri}...")
    pymongo = importlib.import_module("pymongo")
    mongo_client_cls = getattr(pymongo, "MongoClient")
    client = mongo_client_cls(mongo_uri)
    conversations_db = client["conversations_db"]
    transcripts_db = client["full_transcripts"]
    
    # Data directory path
    data_dir = _resolve_data_dir()
    
    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        return False
    
    # Check if data already exists
    if conversations_db["spans"].count_documents({}) > 0:
        print("MongoDB already has data. Skipping initialization.")
        return True
    
    print("Initializing MongoDB with data...")
    
    # Load transcripts
    transcripts_file = data_dir / "full_transcripts" / "transcripts.json"
    if transcripts_file.exists():
        print(f"Loading transcripts from {transcripts_file}...")
        transcripts = _load_json(transcripts_file)
        if transcripts:
            # Insert in batches for large files
            batch_size = 1000
            for i in range(0, len(transcripts), batch_size):
                batch = transcripts[i:i + batch_size]
                transcripts_db["transcripts"].insert_many(batch, ordered=False)
            print(f"Loaded {len(transcripts)} transcripts")
    
    # Load other collections from conversations_db folder
    conv_db_dir = data_dir / "conversations_db"
    if conv_db_dir.exists():
        for json_file in conv_db_dir.glob("*.json"):
            collection_name = json_file.stem
            print(f"Loading {collection_name}...")
            try:
                data = _load_json(json_file)
                inserted = _insert_collection(conversations_db, collection_name, data)
                print(f"  Loaded {inserted} documents into {collection_name}")
            except Exception as e:
                print(f"  Error loading {collection_name}: {e}")
    
    # Create indexes for better performance
    print("Creating indexes...")
    transcripts_db["transcripts"].create_index("transcript_id")
    conversations_db["spans"].create_index("transcript_id")
    conversations_db["spans"].create_index("cluster_id")
    
    print("MongoDB initialization complete!")
    return True

if __name__ == '__main__':
    initMongoDb()
