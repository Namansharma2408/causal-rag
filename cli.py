import sys
from typing import Optional

from .config import logger
from .rag_system import RAGSystem


def print_help():
    """Print help message."""
    print("""
╔══════════════════════════════════════════════════════════════╗
║                    FinalAgent Commands                        ║
╠══════════════════════════════════════════════════════════════╣
║  help, h, ?      - Show this help                            ║
║  quit, exit, q   - Exit the program                          ║
║  clear           - Clear conversation history                 ║
║  evidence, e     - Show evidence for last answer             ║
║  history         - Show conversation history                  ║
║  session         - Show current session ID                    ║
╚══════════════════════════════════════════════════════════════╝
    """)


def format_evidence(evidence: dict) -> str:
    """Format evidence for display."""
    if not evidence:
        return "No evidence available."
    
    lines = []
    lines.append(f"\n{'='*60}")
    lines.append("📋 EVIDENCE")
    lines.append(f"{'='*60}")
    
    verified = evidence.get("verified", False)
    lines.append(f"Verified: {'✅ Yes' if verified else '❌ No'}")
    lines.append(f"Evidence spans: {evidence.get('evidence_count', 0)}")
    
    spans = evidence.get("evidence_spans", [])
    for i, span in enumerate(spans, 1):
        lines.append(f"\n[{i}] {span.get('text', '')}")
        lines.append(f"    Source: {span.get('transcript_id', 'Unknown')}")
    
    lines.append(f"\n{'='*60}")
    return "\n".join(lines)


def interactive_mode(session_id: Optional[str] = None):
    print("""
╔══════════════════════════════════════════════════════════════╗
║                    FinalAgent RAG System                      ║
║                  Type 'help' for commands                     ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    system = RAGSystem(session_id=session_id)
    print(f"Session: {system.session_id}\n")
    
    try:
        while True:
            try:
                user_input = input("\n🔍 You: ").strip()
            except EOFError:
                break
            
            if not user_input:
                continue
            
            # Handle commands
            cmd = user_input.lower()
            
            if cmd in ("quit", "exit", "q"):
                print("Goodbye! 👋")
                break
            
            if cmd in ("help", "h", "?"):
                print_help()
                continue
            
            if cmd == "clear":
                system.clear_history()
                print("✅ Conversation cleared.")
                continue
            
            if cmd in ("evidence", "e"):
                evidence = system.get_evidence()
                print(format_evidence(evidence))
                continue
            
            if cmd == "history":
                history = system.get_conversation()
                if not history:
                    print("No conversation history.")
                else:
                    for i, entry in enumerate(history, 1):
                        print(f"\n[{i}] Q: {entry['query']}")
                        print(f"    A: {entry['answer'][:100]}...")
                continue
            
            if cmd == "session":
                print(f"Session ID: {system.session_id}")
                continue
            
            # Process question
            print("\n⏳ Processing...")
            
            try:
                answer = system.answer(user_input)
                result = system.get_last_result()
                
                print(f"\n🤖 Assistant: {answer}")
                
                if result:
                    print(f"\n📊 Quality: {result.quality_score}/100")
                    if result.transcript_ids:
                        print(f"📑 Sources: {', '.join(result.transcript_ids[:3])}")
                
            except Exception as e:
                logger.error(f"Error processing question: {e}")
                print(f"❌ Error: {e}")
    
    finally:
        system.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="FinalAgent RAG System")
    parser.add_argument("--session", "-s", help="Session ID to use/resume")
    parser.add_argument("--question", "-q", help="Single question to answer")
    parser.add_argument("--proof", "-p", action="store_true", help="Include evidence")
    
    args = parser.parse_args()
    
    if args.question:
        # Single question mode
        system = RAGSystem(session_id=args.session)
        try:
            answer = system.answer(args.question, include_proof=args.proof)
            print(f"Answer: {answer}")
            
            if args.proof:
                evidence = system.get_evidence()
                print(format_evidence(evidence))
        finally:
            system.close()
    else:
        # Interactive mode
        interactive_mode(args.session)


if __name__ == "__main__":
    main()
