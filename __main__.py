"""
Main entry point for FinalAgent.

Usage:
    python -m finalAgent                    # Interactive mode
    python -m finalAgent -q "Question?"     # Single question
    python -m finalAgent -q "Q?" --proof    # With evidence
"""

from .cli import main

if __name__ == "__main__":
    main()
