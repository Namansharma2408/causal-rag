"""
Streamlit Frontend for Causal AI
A modern ChatGPT/Gemini-like interface with markdown support.
"""

import streamlit as st
import uuid
import json
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Must be first Streamlit command
st.set_page_config(
    page_title="Causal AI",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import RAG system
try:
    from finalAgent.rag_system import RAGOrchestrator
    from finalAgent.config import Config
    RAG_AVAILABLE = True
except ImportError as e:
    print(f"Import error: {e}")
    RAG_AVAILABLE = False


def load_css():
    """Load modern ChatGPT/Gemini-inspired CSS."""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    :root {
        --bg-primary: #212121;
        --bg-secondary: #171717;
        --bg-tertiary: #2f2f2f;
        --bg-hover: #3a3a3a;
        --text-primary: #ececec;
        --text-secondary: #b4b4b4;
        --text-muted: #8e8e8e;
        --accent: #10a37f;
        --accent-hover: #1a7f64;
        --border: #424242;
    }
    
    * { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important; }
    
    #MainMenu, footer, header, .stDeployButton { display: none !important; }
    
    .stApp { background: var(--bg-primary) !important; }
    
    .main .block-container {
        padding-top: 2rem !important;
        padding-bottom: 6rem !important;
        max-width: 800px !important;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: var(--bg-secondary) !important;
        border-right: 1px solid var(--border) !important;
    }
    
    section[data-testid="stSidebar"] > div {
        background: var(--bg-secondary) !important;
    }
    
    section[data-testid="stSidebar"] button {
        background: transparent !important;
        border: 1px solid var(--border) !important;
        color: var(--text-primary) !important;
        border-radius: 8px !important;
        font-size: 14px !important;
    }
    
    section[data-testid="stSidebar"] button:hover {
        background: var(--bg-hover) !important;
    }
    
    section[data-testid="stSidebar"] hr {
        border-color: var(--border) !important;
    }
    
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label {
        color: var(--text-secondary) !important;
    }
    
    /* Chat messages using st.chat_message */
    [data-testid="stChatMessage"] {
        background: transparent !important;
        border: none !important;
        padding: 1.5rem 0 !important;
    }
    
    [data-testid="stChatMessageAvatarUser"] {
        background: linear-gradient(135deg, #7c3aed 0%, #a78bfa 100%) !important;
    }
    
    [data-testid="stChatMessageAvatarAssistant"] {
        background: linear-gradient(135deg, #10a37f 0%, #34d399 100%) !important;
    }
    
    /* Markdown content in messages */
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
        color: var(--text-primary) !important;
    }
    
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p {
        color: var(--text-primary) !important;
        font-size: 15px !important;
        line-height: 1.7 !important;
    }
    
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] h1,
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] h2,
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] h3 {
        color: var(--text-primary) !important;
    }
    
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] ul,
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] ol {
        color: var(--text-primary) !important;
    }
    
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] code {
        background: var(--bg-tertiary) !important;
        color: #e879f9 !important;
        padding: 2px 6px !important;
        border-radius: 4px !important;
    }
    
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] pre {
        background: var(--bg-secondary) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
    }
    
    /* Chat input */
    .stChatInput > div {
        background: var(--bg-tertiary) !important;
        border: 1px solid var(--border) !important;
        border-radius: 24px !important;
    }
    
    .stChatInput textarea {
        background: transparent !important;
        color: var(--text-primary) !important;
    }
    
    .stChatInput button {
        background: var(--accent) !important;
        border-radius: 50% !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: var(--bg-tertiary) !important;
        border: 1px solid var(--border) !important;
        color: var(--text-primary) !important;
        border-radius: 12px !important;
    }
    
    .stButton > button:hover {
        background: var(--bg-hover) !important;
        border-color: #555 !important;
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        background: var(--bg-tertiary) !important;
        color: var(--text-secondary) !important;
        border-radius: 8px !important;
    }
    
    .streamlit-expanderContent {
        background: var(--bg-tertiary) !important;
        border: 1px solid var(--border) !important;
    }
    
    /* Metrics */
    [data-testid="stMetric"] {
        background: var(--bg-secondary) !important;
        padding: 12px !important;
        border-radius: 8px !important;
        border: 1px solid var(--border) !important;
    }
    
    [data-testid="stMetric"] label { color: var(--text-muted) !important; }
    [data-testid="stMetric"] [data-testid="stMetricValue"] { color: var(--accent) !important; }
    
    /* Columns for welcome */
    [data-testid="column"] {
        background: transparent !important;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
    </style>
    """, unsafe_allow_html=True)


# Session storage
SESSIONS_DIR = Path(__file__).parent / "chat_sessions"
SESSIONS_DIR.mkdir(exist_ok=True)


def save_session(session_id: str, messages: list, title: str = None):
    """Save chat session."""
    session_file = SESSIONS_DIR / f"{session_id}.json"
    auto_title = "New Chat"
    if messages:
        auto_title = messages[0]["content"][:40]
        if len(messages[0]["content"]) > 40:
            auto_title += "..."
    
    data = {
        "id": session_id,
        "title": title or auto_title,
        "messages": messages,
        "updated_at": datetime.now().isoformat()
    }
    with open(session_file, "w") as f:
        json.dump(data, f, indent=2)


def load_session(session_id: str) -> dict:
    """Load chat session."""
    session_file = SESSIONS_DIR / f"{session_id}.json"
    if session_file.exists():
        with open(session_file, "r") as f:
            return json.load(f)
    return None


def get_all_sessions() -> list:
    """Get all sessions sorted by date."""
    sessions = []
    for file in SESSIONS_DIR.glob("*.json"):
        try:
            with open(file, "r") as f:
                sessions.append(json.load(f))
        except:
            pass
    sessions.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
    return sessions


def delete_session(session_id: str):
    """Delete a session."""
    session_file = SESSIONS_DIR / f"{session_id}.json"
    if session_file.exists():
        session_file.unlink()


def new_session_id() -> str:
    """Generate new session ID."""
    return str(uuid.uuid4())[:8]


def render_welcome():
    """Render welcome screen with Streamlit native components."""
    st.markdown("")
    
    # Center content
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Logo
        st.markdown("""
        <div style="text-align: center; padding: 40px 0 20px;">
            <div style="width: 64px; height: 64px; background: linear-gradient(135deg, #10a37f 0%, #059669 100%); 
                        border-radius: 16px; display: inline-flex; align-items: center; justify-content: center; 
                        font-size: 32px; color: white; box-shadow: 0 8px 32px rgba(16, 163, 127, 0.3);">✦</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<h1 style='text-align: center; color: #ececec; font-size: 32px; margin-bottom: 8px;'>Causal AI Assistant</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #8e8e8e; font-size: 16px; margin-bottom: 40px;'>Intelligent analysis of customer conversations.<br>Discover patterns, causes, and insights from 87,000+ interactions.</p>", unsafe_allow_html=True)
        
        # Capabilities using native columns
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("<p style='text-align: center; color: #b4b4b4;'>🎯 Causal Analysis</p>", unsafe_allow_html=True)
        with c2:
            st.markdown("<p style='text-align: center; color: #b4b4b4;'>🔄 Multi-hop Reasoning</p>", unsafe_allow_html=True)
        with c3:
            st.markdown("<p style='text-align: center; color: #b4b4b4;'>📊 Evidence-backed</p>", unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Example prompts
    col1, col2 = st.columns(2)
    
    prompts = [
        ("🔍", "What causes customer churn?"),
        ("💡", "What triggers brand loyalty?"),
        ("📈", "Compare churn vs retention factors"),
        ("🎯", "How does service quality affect satisfaction?"),
    ]
    
    for i, (icon, text) in enumerate(prompts):
        with col1 if i % 2 == 0 else col2:
            if st.button(f"{icon}  {text}", key=f"prompt_{i}", use_container_width=True):
                st.session_state.pending_input = text
                st.rerun()


def render_sidebar():
    """Render sidebar."""
    with st.sidebar:
        # Brand header
        st.markdown("""
        <div style="display: flex; align-items: center; gap: 12px; padding: 8px 0 20px; border-bottom: 1px solid #424242; margin-bottom: 16px;">
            <div style="width: 32px; height: 32px; background: linear-gradient(135deg, #10a37f 0%, #059669 100%); 
                        border-radius: 8px; display: flex; align-items: center; justify-content: center; 
                        font-size: 16px; color: white;">✦</div>
            <span style="font-size: 18px; font-weight: 600; color: #ececec;">Causal AI</span>
            <span style="font-size: 10px; color: #8e8e8e; background: #2f2f2f; padding: 2px 6px; border-radius: 4px; margin-left: auto;">v2.0</span>
        </div>
        """, unsafe_allow_html=True)
        
        # New chat
        if st.button("✚  New chat", key="new_chat", use_container_width=True):
            st.session_state.current_session = new_session_id()
            st.session_state.messages = []
            st.rerun()
        
        st.divider()
        
        # Recent chats
        st.markdown("**RECENT**")
        sessions = get_all_sessions()
        
        for session in sessions[:12]:
            col1, col2 = st.columns([5, 1])
            with col1:
                title = session.get("title", "Untitled")[:28]
                if len(session.get("title", "")) > 28:
                    title += "..."
                
                is_active = st.session_state.get("current_session") == session["id"]
                
                if st.button(
                    f"💬 {title}",
                    key=f"s_{session['id']}",
                    use_container_width=True,
                    type="primary" if is_active else "secondary"
                ):
                    st.session_state.current_session = session["id"]
                    loaded = load_session(session["id"])
                    if loaded:
                        st.session_state.messages = loaded.get("messages", [])
                    st.rerun()
            
            with col2:
                if st.button("✕", key=f"d_{session['id']}"):
                    delete_session(session["id"])
                    if st.session_state.get("current_session") == session["id"]:
                        st.session_state.current_session = new_session_id()
                        st.session_state.messages = []
                    st.rerun()
        
        if not sessions:
            st.caption("No conversations yet")
        
        # Settings
        st.divider()
        st.markdown("**SETTINGS**")
        
        st.session_state.fast_mode = st.toggle(
            "⚡ Fast mode",
            value=st.session_state.get("fast_mode", True),
            help="Skip quality checks for speed"
        )
        
        st.session_state.thinking_mode = st.toggle(
            "🧠 Thinking mode",
            value=st.session_state.get("thinking_mode", False),
            help="Multi-model consensus with peer review (slower but higher quality)"
        )
        
        st.session_state.include_proof = st.toggle(
            "📋 Show evidence",
            value=st.session_state.get("include_proof", False),
            help="Include source verification"
        )
        
        # Footer
        st.divider()
        st.caption(f"Session: `{st.session_state.get('current_session', 'N/A')[:8]}`")


def get_response(question: str, session_id: str) -> dict:
    """Get RAG system response."""
    if not RAG_AVAILABLE:
        return {
            "answer": "⚠️ RAG system not available. Please check the imports and run from the correct directory.",
            "error": True
        }
    
    try:
        config = Config()
        config.FAST_MODE = st.session_state.get("fast_mode", True)
        config.THINKING_MODE = st.session_state.get("thinking_mode", False)
        
        rag = RAGOrchestrator(config=config)
        result = rag.answer(
            question,
            session_id=session_id,
            include_proof=st.session_state.get("include_proof", False),
            thinking_mode=st.session_state.get("thinking_mode", False)
        )
        
        # Build response dict with all metadata
        response_data = {
            "answer": result.answer,
            "quality_score": getattr(result, 'quality_score', None),
            "doc_count": len(result.documents) if result.documents else 0,
            "transcript_ids": getattr(result, 'transcript_ids', []),
            "multihop": result.metadata.get("multihop", False) if result.metadata else False,
            "thinking_mode": result.metadata.get("thinking_mode", False) if result.metadata else False,
            "evidence": getattr(result, 'evidence', None)
        }
        
        # Add thinking mode specific metadata
        if result.metadata and result.metadata.get("thinking_mode"):
            response_data["winning_model"] = result.metadata.get("winning_model")
            response_data["consensus_score"] = result.metadata.get("consensus_score")
            response_data["model_scores"] = result.metadata.get("model_scores", {})
        
        return response_data
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "answer": f"Sorry, I encountered an error: {str(e)}",
            "error": True
        }


def display_message(role: str, content: str, metadata: dict = None):
    """Display a chat message with proper markdown rendering."""
    # Use supported avatar values: "user", "assistant", or emoji
    avatar = "🧑" if role == "user" else "🤖"
    
    with st.chat_message(role, avatar=avatar):
        # Render content as markdown (this properly renders markdown!)
        st.markdown(content)
        
        # Show metadata for assistant messages
        if role == "assistant" and metadata and not metadata.get("error"):
            # Show metrics in an expander
            with st.expander("📊 Response Details", expanded=False):
                # Check if thinking mode was used
                if metadata.get("thinking_mode"):
                    st.markdown("**🧠 Thinking Mode Analysis**")
                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        consensus = metadata.get("consensus_score", 0)
                        st.metric("Consensus", f"{int(consensus * 100)}%")
                    with c2:
                        st.metric("Sources", metadata.get("doc_count", 0))
                    with c3:
                        winner = metadata.get("winning_model", "N/A")
                        st.metric("Winner", winner.title())
                    with c4:
                        st.metric("Models", "4")
                    
                    # Show model scores
                    model_scores = metadata.get("model_scores", {})
                    if model_scores:
                        st.markdown("**Model Scores:**")
                        cols = st.columns(len(model_scores))
                        for i, (model, score) in enumerate(model_scores.items()):
                            with cols[i]:
                                st.metric(model.title(), f"{score:.1f}/10")
                else:
                    # Standard mode
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        score = metadata.get("quality_score")
                        st.metric("Quality", f"{score}/100" if score else "N/A")
                    with c2:
                        st.metric("Sources", metadata.get("doc_count", 0))
                    with c3:
                        mode = "Multi-hop" if metadata.get("multihop") else "Single"
                        st.metric("Mode", mode)


def process_input(question: str):
    """Process user input and get response."""
    # Add user message
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })
    
    # Get response with spinner
    with st.spinner("🔍 Analyzing..."):
        response = get_response(question, st.session_state.current_session)
    
    # Add assistant response
    st.session_state.messages.append({
        "role": "assistant",
        "content": response["answer"],
        "metadata": response
    })
    
    # Save session
    save_session(st.session_state.current_session, st.session_state.messages)
    
    st.rerun()


def main():
    """Main app."""
    load_css()
    
    # Initialize state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "current_session" not in st.session_state:
        st.session_state.current_session = new_session_id()
    if "pending_input" not in st.session_state:
        st.session_state.pending_input = None
    
    # Render sidebar
    render_sidebar()
    
    # Main content
    if not st.session_state.messages:
        render_welcome()
    else:
        # Display all messages using st.chat_message (proper markdown rendering)
        for msg in st.session_state.messages:
            display_message(
                msg["role"],
                msg["content"],
                msg.get("metadata")
            )
    
    # Handle pending input from example buttons
    if st.session_state.pending_input:
        question = st.session_state.pending_input
        st.session_state.pending_input = None
        process_input(question)
    
    # Chat input
    if user_input := st.chat_input("Message Causal AI..."):
        process_input(user_input)


if __name__ == "__main__":
    main()
