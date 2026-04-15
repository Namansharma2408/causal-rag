#!/usr/bin/env python3
"""
MongoDB Data Export Script
Exports all collections from conversations_db and full_transcripts to JSON files
Uses MongoDB Extended JSON v2 format for compatibility with mongoimport
"""

import json
import os
from datetime import datetime
from bson import ObjectId
from bson.json_util import dumps as bson_dumps
from pymongo import MongoClient


def export_collection(db, collection_name, output_dir):
    """Export a single collection to JSON file using BSON extended JSON."""
    collection = db[collection_name]
    documents = list(collection.find())
    
    output_file = os.path.join(output_dir, f"{collection_name}.json")
    
    # Use bson.json_util.dumps for MongoDB Extended JSON format
    with open(output_file, 'w') as f:
        f.write(bson_dumps(documents))
    
    print(f"  Exported {len(documents)} documents to {output_file}")
    return len(documents)

def main():
    # Connect to MongoDB
    client = MongoClient('mongodb://localhost:27017')
    
    # Output directory
    output_base = 'mongo_data'
    os.makedirs(output_base, exist_ok=True)
    
    # Databases to export
    databases = ['conversations_db', 'full_transcripts']
    
    total_docs = 0
    
    for db_name in databases:
        print(f"\n=== Exporting database: {db_name} ===")
        db = client[db_name]
        
        # Create database directory
        db_dir = os.path.join(output_base, db_name)
        os.makedirs(db_dir, exist_ok=True)
        
        # Export each collection
        for coll_name in db.list_collection_names():
            count = export_collection(db, coll_name, db_dir)
            total_docs += count
    
    print(f"\n=== Export Complete ===")
    print(f"Total documents exported: {total_docs}")
    print(f"Data saved to: {output_base}/")

if __name__ == "__main__":
    main()
