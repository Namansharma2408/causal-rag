#!/usr/bin/env python3
"""
Script to run all queries from CSV and append answers.
Handles conversation context for follow-up questions.
"""

import csv
import sys
import os
import time
from datetime import datetime

# Add parent to path so we can import finalAgent as a package
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Change to finalAgent directory for .env loading
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv(override=True)

# Now import from the package
from finalAgent.rag_system import RAGSystem

def clean_answer(answer: str) -> str:
    """Clean answer for CSV - escape quotes and handle newlines."""
    # Replace newlines with <br> for markdown
    answer = answer.replace('\n', '<br>')
    # Escape double quotes
    answer = answer.replace('"', '""')
    return answer

def run_all_queries():
    """Run all queries from CSV and save results."""
    
    input_file = "Queries - final_with_querytype_v3.csv.csv"
    output_file = f"Queries_with_answers_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    print(f"=" * 60)
    print(f"Query Runner - Processing {input_file}")
    print(f"Output will be saved to: {output_file}")
    print(f"=" * 60)
    
    # Initialize RAG system
    print("\nInitializing RAG system...")
    rag = RAGSystem()
    session_id = None
    
    # Read input CSV
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    total = len(rows)
    print(f"Found {total} queries to process\n")
    
    results = []
    current_conversation = 0
    
    for i, row in enumerate(rows):
        query_type = row['Query Type'].strip()
        query = row['Query'].strip()
        
        print(f"[{i+1}/{total}] [{query_type}] {query[:60]}...")
        
        # Reset context for Initial queries
        if query_type == "Initial":
            current_conversation += 1
            session_id = f"conv_{current_conversation}_{int(time.time())}"
            print(f"  → New conversation #{current_conversation}")
        
        try:
            start_time = time.time()
            
            # Run the query - simple call without session
            result = rag.answer(query)
            
            elapsed = time.time() - start_time
            
            # Handle different result types
            if isinstance(result, str):
                answer = result
            elif isinstance(result, dict):
                answer = result.get('answer', 'No answer generated')
            else:
                answer = str(result)
            
            print(f"  ✓ Done in {elapsed:.1f}s ({len(answer)} chars)")
            
        except Exception as e:
            answer = f"Error: {str(e)}"
            print(f"  ✗ Error: {e}")
        
        # Store result
        results.append({
            'Query Type': query_type,
            'Query': query,
            'Answers': clean_answer(answer)
        })
        
        # Save progress after each query (in case of interruption)
        with open(output_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['Query Type', 'Query', 'Answers'])
            writer.writeheader()
            writer.writerows(results)
    
    print(f"\n{'=' * 60}")
    print(f"✅ Completed! Processed {total} queries")
    print(f"📄 Results saved to: {output_file}")
    print(f"{'=' * 60}")

if __name__ == "__main__":
    run_all_queries()
