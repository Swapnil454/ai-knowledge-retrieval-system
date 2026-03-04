"""
AI Knowledge Retrieval System - Production Application
Enterprise-grade RAG (Retrieval Augmented Generation) interface
"""
import streamlit as st
import sys
import os
import json
import time
import hashlib
from datetime import datetime
from typing import Optional

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from core.pipeline import build_document_index, answer_question, merge_indices, clear_cache
from core.memory import ConversationMemory
from evaluation.query_logs import log_query

# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title="AI Knowledge Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo/ai-knowledge-retrieval',
        'Report a bug': 'https://github.com/your-repo/ai-knowledge-retrieval/issues',
        'About': '### AI Knowledge Retrieval System\nVersion 2.0 - Enterprise Edition'
    }
)

# =============================================================================
# Custom CSS for Production UI
# =============================================================================

st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Global styling */
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    h1, h2, h3, h4 {
        font-weight: 600 !important;
    }
    
    /* Chat message styling */
    .chat-message {
        padding: 1.25rem 1.5rem;
        border-radius: 1rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        animation: fadeIn 0.3s ease-out;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .user-message {
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
        color: white;
        margin-left: 15%;
        border-bottom-right-radius: 0.25rem;
    }
    
    .assistant-message {
        background: linear-gradient(135deg, #1E293B 0%, #334155 100%);
        color: #E2E8F0;
        margin-right: 15%;
        border-bottom-left-radius: 0.25rem;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .message-header {
        font-size: 0.75rem;
        opacity: 0.8;
        margin-bottom: 0.75rem;
        font-weight: 500;
        letter-spacing: 0.025em;
    }
    
    /* Source card styling */
    .source-card {
        background: linear-gradient(to bottom, #F8FAFC, #F1F5F9);
        border: 1px solid #E2E8F0;
        border-radius: 0.75rem;
        padding: 1rem;
        margin-bottom: 0.75rem;
        transition: all 0.2s ease;
    }
    
    .source-card:hover {
        border-color: #7C3AED;
        box-shadow: 0 4px 12px rgba(124,58,237,0.15);
    }
    
    /* Stats card */
    .stats-card {
        background: linear-gradient(145deg, #1E293B 0%, #0F172A 100%);
        border-radius: 1rem;
        padding: 2rem 1.5rem;
        text-align: center;
        margin: 0.75rem;
        color: white;
        border: 1px solid rgba(255,255,255,0.1);
        box-shadow: 0 8px 24px rgba(0,0,0,0.2);
        transition: transform 0.2s ease;
    }
    
    .stats-card:hover {
        transform: translateY(-4px);
    }
    
    .stats-card h3 {
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }
    
    .stats-card p {
        color: #94A3B8;
        margin-bottom: 0.25rem;
    }
    
    .stats-card small {
        color: #64748B;
        font-size: 0.8rem;
    }
    
    /* Status badges */
    .status-ready { 
        background: linear-gradient(135deg, #10B981, #059669);
        color: white;
        padding: 0.35rem 1rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 500;
        box-shadow: 0 2px 8px rgba(16,185,129,0.3);
    }
    
    .status-waiting { 
        background: linear-gradient(135deg, #F59E0B, #D97706);
        color: white;
        padding: 0.35rem 1rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 500;
        box-shadow: 0 2px 8px rgba(245,158,11,0.3);
    }
    
    /* Button styling */
    .stButton > button {
        border-radius: 0.5rem !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    /* Input styling */
    .stTextInput > div > div > input {
        border-radius: 0.75rem !important;
        padding: 0.75rem 1rem !important;
        border: 2px solid #E2E8F0 !important;
        transition: border-color 0.2s ease !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #7C3AED !important;
        box-shadow: 0 0 0 3px rgba(124,58,237,0.15) !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #F8FAFC, #F1F5F9) !important;
        border-radius: 0.5rem !important;
        font-weight: 500 !important;
    }
    
    /* Metric styling */
    [data-testid="stMetricValue"] {
        font-size: 1.75rem !important;
        font-weight: 700 !important;
        color: #7C3AED !important;
    }
    
    /* Info/success/warning boxes */
    .stAlert {
        border-radius: 0.75rem !important;
        border: none !important;
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Sidebar improvements */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0F172A 0%, #1E293B 100%);
    }
    
    section[data-testid="stSidebar"] .stMarkdown {
        color: #E2E8F0;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# Session State Initialization
# =============================================================================

def init_session_state():
    """Initialize all session state variables."""
    defaults = {
        "memory": ConversationMemory(),
        "all_chunks": [],
        "all_embeddings": [],
        "index": None,
        "processed_files": {},
        "messages": [],
        "feedback": {},
        "settings": {
            "show_sources": True,
            "show_confidence": True,
            "auto_suggest": True,
        },
        "total_queries": 0,
        "session_start": datetime.now().isoformat(),
        "suggested_questions": [],
        "is_processing": False
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# =============================================================================
# Helper Functions
# =============================================================================

def generate_message_id() -> str:
    """Generate unique message ID."""
    return hashlib.md5(f"{datetime.now().isoformat()}{len(st.session_state.messages)}".encode()).hexdigest()[:8]

def add_message(role: str, content: str, sources: list = None, confidence: float = None):
    """Add a message to chat history."""
    msg_id = generate_message_id()
    st.session_state.messages.append({
        "id": msg_id,
        "role": role,
        "content": content,
        "timestamp": datetime.now().strftime("%H:%M"),
        "sources": sources or [],
        "confidence": confidence
    })
    return msg_id

def generate_follow_up_questions(question: str, answer: str) -> list:
    """Generate suggested follow-up questions."""
    suggestions = []
    q_lower = question.lower()
    
    if any(w in q_lower for w in ["what", "define", "explain"]):
        suggestions.append("Can you provide more details?")
        suggestions.append("Why is this important?")
    elif any(w in q_lower for w in ["how", "process", "steps"]):
        suggestions.append("What are the key steps?")
        suggestions.append("Are there alternatives?")
    elif any(w in q_lower for w in ["why", "reason"]):
        suggestions.append("What are the implications?")
    else:
        suggestions.append("Summarize the key points")
        suggestions.append("What else should I know?")
    
    return suggestions[:3]

def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format."""
    for unit in ['B', 'KB', 'MB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} GB"

def export_conversation() -> str:
    """Export conversation as markdown."""
    md = f"# AI Knowledge Assistant - Conversation\n\n"
    md += f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n---\n\n"
    
    for msg in st.session_state.messages:
        role = "**You:**" if msg["role"] == "user" else "**Assistant:**"
        md += f"{role}\n\n{msg['content']}\n\n---\n\n"
    
    return md

def export_as_json() -> str:
    """Export conversation as JSON."""
    export_data = {
        "exported_at": datetime.now().isoformat(),
        "documents": list(st.session_state.processed_files.keys()),
        "total_queries": st.session_state.total_queries,
        "conversation": st.session_state.messages
    }
    return json.dumps(export_data, indent=2)

# =============================================================================
# Document Processing
# =============================================================================

def process_uploaded_files(uploaded_files):
    """Process uploaded files and build index."""
    new_files = [f for f in uploaded_files if f.name not in st.session_state.processed_files]
    
    if not new_files:
        return
    
    progress = st.progress(0, text="Processing documents...")
    
    for i, file in enumerate(new_files):
        try:
            progress.progress((i + 0.5) / len(new_files), text=f"Processing: {file.name}")
            
            chunks, embeddings, _ = build_document_index(file)
            
            st.session_state.all_chunks.extend(chunks)
            st.session_state.all_embeddings.extend(embeddings)
            
            st.session_state.processed_files[file.name] = {
                "chunks": len(chunks),
                "size": format_file_size(file.size),
                "added": datetime.now().strftime("%H:%M")
            }
            
            progress.progress((i + 1) / len(new_files))
            
        except Exception as e:
            st.error(f"❌ Error with {file.name}: {str(e)}")
    
    if st.session_state.all_embeddings:
        st.session_state.index = merge_indices(st.session_state.all_embeddings)
        clear_cache()
    
    progress.empty()
    st.success(f"✅ Added {len(new_files)} document(s)")

def remove_document(filename: str):
    """Remove a document from the index."""
    st.session_state.all_chunks = [
        c for c in st.session_state.all_chunks if c.get("source") != filename
    ]
    del st.session_state.processed_files[filename]
    
    if st.session_state.all_chunks:
        from embeddings.embedder import embed_text
        texts = [c["text"] for c in st.session_state.all_chunks]
        st.session_state.all_embeddings = list(embed_text(texts))
        st.session_state.index = merge_indices(st.session_state.all_embeddings)
    else:
        st.session_state.all_embeddings = []
        st.session_state.index = None
    
    clear_cache()

def reset_all():
    """Reset all application state."""
    st.session_state.all_chunks = []
    st.session_state.all_embeddings = []
    st.session_state.index = None
    st.session_state.processed_files = {}
    st.session_state.messages = []
    st.session_state.memory.clear()
    st.session_state.total_queries = 0
    st.session_state.feedback = {}
    st.session_state.suggested_questions = []
    clear_cache()

# =============================================================================
# Question Processing
# =============================================================================

def process_question(question: str):
    """Process a user question and generate answer."""
    add_message("user", question)
    st.session_state.total_queries += 1
    
    try:
        doc_hash = "|".join(sorted(st.session_state.processed_files.keys()))
        
        answer, context, retrieved, confidence = answer_question(
            question,
            st.session_state.all_chunks,
            st.session_state.all_embeddings,
            st.session_state.index,
            document_hash=doc_hash
        )
        
        sources = [
            {"source": c["source"], "page": c["page"], "text": c["text"][:300]}
            for c in retrieved[:5]
        ]
        
        add_message("assistant", answer, sources=sources, confidence=confidence)
        
        st.session_state.memory.add("user", question)
        st.session_state.memory.add("assistant", answer)
        st.session_state.suggested_questions = generate_follow_up_questions(question, answer)
        
        try:
            log_query(question, answer)
        except:
            pass
            
    except Exception as e:
        add_message("assistant", f"❌ Error: {str(e)}\n\nPlease try rephrasing your question.")

# =============================================================================
# UI Components
# =============================================================================

def render_sidebar():
    """Render sidebar with all controls."""
    with st.sidebar:
        # Header
        st.markdown("## 📁 Documents")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Upload files",
            type=["pdf", "txt", "docx"],
            accept_multiple_files=True,
            label_visibility="collapsed",
            help="Supports PDF, TXT, and DOCX files"
        )
        
        if uploaded_files:
            process_uploaded_files(uploaded_files)
        
        # Document list
        if st.session_state.processed_files:
            st.markdown("---")
            for fname, meta in st.session_state.processed_files.items():
                col1, col2 = st.columns([5, 1])
                with col1:
                    st.markdown(f"📄 **{fname[:25]}{'...' if len(fname) > 25 else ''}**")
                    st.caption(f"{meta['chunks']} chunks • {meta['size']}")
                with col2:
                    if st.button("✕", key=f"rm_{fname}"):
                        remove_document(fname)
                        st.rerun()
        
        # Stats
        st.markdown("---")
        st.markdown("## 📊 Statistics")
        c1, c2 = st.columns(2)
        c1.metric("Documents", len(st.session_state.processed_files))
        c2.metric("Chunks", len(st.session_state.all_chunks))
        c1.metric("Queries", st.session_state.total_queries)
        c2.metric("Messages", len(st.session_state.messages))
        
        # Settings
        st.markdown("---")
        st.markdown("## ⚙️ Settings")
        
        st.session_state.settings["show_sources"] = st.toggle(
            "Show Sources", value=st.session_state.settings["show_sources"]
        )
        st.session_state.settings["show_confidence"] = st.toggle(
            "Show Confidence", value=st.session_state.settings["show_confidence"]
        )
        st.session_state.settings["auto_suggest"] = st.toggle(
            "Suggest Questions", value=st.session_state.settings["auto_suggest"]
        )
        
        # Actions
        st.markdown("---")
        st.markdown("## 🔧 Actions")
        
        c1, c2 = st.columns(2)
        if c1.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.memory.clear()
            st.rerun()
        
        if c2.button("🔄 Reset All", use_container_width=True):
            reset_all()
            st.rerun()
        
        # Export options
        if st.session_state.messages:
            st.markdown("---")
            st.markdown("## 📥 Export")
            
            c1, c2 = st.columns(2)
            c1.download_button(
                "📝 Markdown",
                data=export_conversation(),
                file_name=f"chat_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                mime="text/markdown",
                use_container_width=True
            )
            c2.download_button(
                "📋 JSON",
                data=export_as_json(),
                file_name=f"chat_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                mime="application/json",
                use_container_width=True
            )
        
        # Help
        st.markdown("---")
        with st.expander("❓ Help"):
            st.markdown("""
            **How to use:**
            1. Upload PDF or TXT documents
            2. Ask questions in natural language
            3. Click suggested questions for follow-ups
            4. Use 👍/👎 to rate answers
            5. Export conversations when needed
            
            **Tips:**
            - Be specific in questions
            - Check source citations
            - Use follow-ups for details
            """)

def render_message(msg: dict, idx: int):
    """Render a single chat message."""
    is_user = msg["role"] == "user"
    
    if is_user:
        st.markdown(f"""
        <div class="chat-message user-message">
            <div class="message-header">You • {msg['timestamp']}</div>
            {msg['content']}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-message assistant-message">
            <div class="message-header">🤖 Assistant • {msg['timestamp']}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(msg['content'])
        
        # Action buttons
        cols = st.columns([1, 1, 1, 3])
        
        with cols[0]:
            if st.button("👍", key=f"like_{msg['id']}", help="Good answer"):
                st.session_state.feedback[msg['id']] = "positive"
                st.toast("Thanks! 👍")
        
        with cols[1]:
            if st.button("👎", key=f"dislike_{msg['id']}", help="Poor answer"):
                st.session_state.feedback[msg['id']] = "negative"
                st.toast("Thanks for feedback")
        
        with cols[2]:
            if st.button("🔄", key=f"regen_{msg['id']}", help="Regenerate"):
                if idx > 0:
                    q = st.session_state.messages[idx - 1]["content"]
                    st.session_state.messages.pop(idx)
                    process_question(q)
                    st.rerun()
        
        with cols[3]:
            if st.session_state.settings["show_confidence"] and msg.get("confidence"):
                c = msg["confidence"]
                emoji = "✅" if c >= 0.7 else "⚠️" if c >= 0.4 else "❓"
                st.caption(f"{emoji} Confidence: {c:.0%}")
        
        # Sources
        if st.session_state.settings["show_sources"] and msg.get("sources"):
            with st.expander("📚 View Sources"):
                for i, src in enumerate(msg["sources"], 1):
                    st.markdown(f"""
                    **Source {i}:** {src['source']} (Page {src['page']})  
                    > {src['text'][:200]}...
                    """)

def render_onboarding():
    """Render welcome screen for new users."""
    st.markdown("""
    <div style="text-align: center; padding: 2rem 1rem;">
        <h1 style="font-size: 2.75rem; margin-bottom: 0.5rem;">🤖 AI Knowledge Assistant</h1>
        <p style="font-size: 1.15rem; color: #94A3B8; max-width: 600px; margin: 0 auto;">
            Upload your documents and ask questions in natural language. 
            Get detailed, accurate answers with source citations.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown("""
        <div class="stats-card">
            <h3>📄</h3>
            <p><strong>Step 1: Upload</strong></p>
            <small>PDF, TXT, DOCX supported</small>
        </div>
        """, unsafe_allow_html=True)
    
    with c2:
        st.markdown("""
        <div class="stats-card">
            <h3>💬</h3>
            <p><strong>Step 2: Ask</strong></p>
            <small>Natural language questions</small>
        </div>
        """, unsafe_allow_html=True)
    
    with c3:
        st.markdown("""
        <div class="stats-card">
            <h3>✨</h3>
            <p><strong>Step 3: Discover</strong></p>
            <small>Detailed answers with sources</small>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 💡 Example Questions")
    
    examples = [
        ("🔍", "What are the key findings?"),
        ("📝", "Summarize the main points"),
        ("🔬", "What methodology was used?"),
        ("📊", "What are the conclusions?")
    ]
    
    cols = st.columns(4)
    for i, (icon, ex) in enumerate(examples):
        with cols[i]:
            if st.button(f"{icon} {ex}", key=f"ex_{i}", use_container_width=True):
                st.info("Upload documents first to ask questions")

def render_chat():
    """Render the main chat interface."""
    # Header
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown("## 💬 Chat")
    with col2:
        if st.session_state.index:
            st.markdown('<span class="status-ready">● Ready</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-waiting">● Upload docs</span>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Messages
    if not st.session_state.messages:
        st.info("👋 Upload documents and start asking questions!")
    else:
        for i, msg in enumerate(st.session_state.messages):
            render_message(msg, i)
    
    # Suggestions
    if st.session_state.settings["auto_suggest"] and st.session_state.suggested_questions:
        st.markdown("**💡 Suggested:**")
        cols = st.columns(len(st.session_state.suggested_questions))
        for i, suggestion in enumerate(st.session_state.suggested_questions):
            with cols[i]:
                if st.button(suggestion, key=f"sug_{i}", use_container_width=True):
                    process_question(suggestion)
                    st.rerun()
    
    st.markdown("---")
    
    # Input
    col1, col2 = st.columns([6, 1])
    with col1:
        question = st.text_input(
            "Question",
            placeholder="Ask anything about your documents...",
            key="q_input",
            label_visibility="collapsed"
        )
    with col2:
        send = st.button("Send ➤", type="primary", use_container_width=True)
    
    if question and send:
        if st.session_state.index:
            process_question(question)
            st.rerun()
        else:
            st.warning("⚠️ Please upload documents first")

# =============================================================================
# Main Application
# =============================================================================

def main():
    """Main application entry point."""
    render_sidebar()
    
    if not st.session_state.processed_files:
        render_onboarding()
    else:
        render_chat()

if __name__ == "__main__":
    main()
