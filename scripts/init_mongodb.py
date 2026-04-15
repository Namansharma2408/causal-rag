#!/usr/bin/env python3
"""
MongoDB Data Initialization Script
Loads conversation and transcript data into MongoDB from JSON files.
"""

import json
import os
from pathlib import Path
from pymongo import MongoClient

def init_mongodb():
    """Initialize MongoDB with data from JSON files."""
    
    # Get MongoDB URI from environment
    mongo_uri = os.getenv('MONGODB_URI', 'mongodb://localhost:27017')
    
    print(f"Connecting to MongoDB at {mongo_uri}...")
    client = MongoClient(mongo_uri)
    db = client['conversations_db']
    
    # Data directory path
    data_dir = Path('/app/finalAgent/mongo_data')
    if not data_dir.exists():
        data_dir = Path(__file__).parent / 'finalAgent' / 'mongo_data'
    
    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        return False
    
    # Check if data already exists
    if db['transcripts'].count_documents({}) > 0:
        print("MongoDB already has data. Skipping initialization.")
        return True
    
    print("Initializing MongoDB with data...")
    
    # Load transcripts
    transcripts_file = data_dir / 'full_transcripts' / 'transcripts.json'
    if transcripts_file.exists():
        print(f"Loading transcripts from {transcripts_file}...")
        with open(transcripts_file, 'r') as f:
            transcripts = json.load(f)
            if transcripts:
                # Insert in batches for large files
                batch_size = 1000
                for i in range(0, len(transcripts), batch_size):
                    batch = transcripts[i:i+batch_size]
                    db['transcripts'].insert_many(batch)
                print(f"Loaded {len(transcripts)} transcripts")
    
    # Load other collections from conversations_db folder
    conv_db_dir = data_dir / 'conversations_db'
    if conv_db_dir.exists():
        for json_file in conv_db_dir.glob('*.json'):
            collection_name = json_file.stem
            print(f"Loading {collection_name}...")
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    if data and isinstance(data, list) and len(data) > 0:
                        db[collection_name].insert_many(data)
                        print(f"  Loaded {len(data)} documents into {collection_name}")
                    elif data and isinstance(data, dict):
                        db[collection_name].insert_one(data)
                        print(f"  Loaded 1 document into {collection_name}")
            except Exception as e:
                print(f"  Error loading {collection_name}: {e}")
    
    # Create indexes for better performance
    print("Creating indexes...")
    db['transcripts'].create_index('transcript_id')
    db['transcripts'].create_index('$**', name='text_search')
    
    print("MongoDB initialization complete!")
    return True

if __name__ == '__main__':
    init_mongodb()
