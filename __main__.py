"""
Main entry point for Causal AI RAG package.

Usage:
    python -m <package_name>                    # Interactive mode
    python -m <package_name> -q "Question?"     # Single question
    python -m <package_name> -q "Q?" --proof    # With evidence
"""

from .cli import main

if __name__ == "__main__":
    main()
