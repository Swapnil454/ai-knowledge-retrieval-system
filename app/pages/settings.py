"""
Settings & About - Application configuration and information
"""
import streamlit as st
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from core.config import (
    MODEL_NAME, RERANK_MODEL, QA_MODEL,
    CHUNK_SIZE, CHUNK_OVERLAP,
    TOP_K_RETRIEVAL, TOP_K_CONTEXT
)

st.set_page_config(
    page_title="Settings",
    page_icon="⚙️",
    layout="wide"
)

st.title("⚙️ Settings & About")

# =============================================================================
# About Section
# =============================================================================

st.markdown("---")
st.subheader("ℹ️ About")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    ### AI Knowledge Retrieval System
    **Version 2.0 - Enterprise Edition**
    
    An intelligent document Q&A system powered by modern AI technologies:
    
    - **Retrieval Augmented Generation (RAG)** for accurate answers
    - **Semantic Search** using dense vector embeddings
    - **BM25 Keyword Search** for lexical matching
    - **Cross-Encoder Reranking** for improved relevance
    - **Natural Language Generation** for conversational responses
    
    Built with ❤️ using Python, Streamlit, and Hugging Face Transformers.
    """)

with col2:
    st.markdown("""
    **Quick Links:**
    - [Documentation](#)
    - [Report Issue](#)
    - [Feature Request](#)
    
    **Contact:**
    - support@example.com
    """)

# =============================================================================
# Current Configuration
# =============================================================================

st.markdown("---")
st.subheader("🔧 Current Configuration")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Models:**")
    st.code(f"""
Embedding Model: {MODEL_NAME}
Reranker Model: {RERANK_MODEL}
QA Model: {QA_MODEL}
    """)

with col2:
    st.markdown("**Processing:**")
    st.code(f"""
Chunk Size: {CHUNK_SIZE} words
Chunk Overlap: {CHUNK_OVERLAP} words
Top-K Retrieval: {TOP_K_RETRIEVAL}
Top-K Context: {TOP_K_CONTEXT}
    """)

# =============================================================================
# Environment Variables
# =============================================================================

st.markdown("---")
st.subheader("🌍 Environment Configuration")

st.markdown("""
You can customize the system using environment variables:
""")

env_vars = [
    ("EMBEDDING_MODEL", MODEL_NAME, "Sentence transformer model for embeddings"),
    ("RERANK_MODEL", RERANK_MODEL, "Cross-encoder model for reranking"),
    ("QA_MODEL", QA_MODEL, "Model for extractive QA"),
    ("CHUNK_SIZE", str(CHUNK_SIZE), "Words per chunk"),
    ("CHUNK_OVERLAP", str(CHUNK_OVERLAP), "Overlap between chunks"),
    ("TOP_K_RETRIEVAL", str(TOP_K_RETRIEVAL), "Chunks to retrieve"),
    ("TOP_K_CONTEXT", str(TOP_K_CONTEXT), "Chunks for answer generation"),
]

for var, default, desc in env_vars:
    col1, col2, col3 = st.columns([2, 3, 3])
    with col1:
        st.code(var)
    with col2:
        st.caption(f"Default: {default}")
    with col3:
        st.caption(desc)

# =============================================================================
# System Status
# =============================================================================

st.markdown("---")
st.subheader("🖥️ System Status")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Python:**")
    st.code(sys.version.split()[0])

with col2:
    st.markdown("**Platform:**")
    import platform
    st.code(platform.system())

with col3:
    st.markdown("**Streamlit:**")
    st.code(st.__version__)

# Model status
st.markdown("**Model Status:**")

try:
    from embeddings.embedder import MODEL_AVAILABLE as EMB_AVAILABLE
    emb_status = "✅ Loaded" if EMB_AVAILABLE else "❌ Not available"
except:
    emb_status = "❓ Unknown"

try:
    from services.qa_service import MODEL_AVAILABLE as QA_AVAILABLE, GEN_MODEL_AVAILABLE
    qa_status = "✅ Loaded" if QA_AVAILABLE else "❌ Not available"
    gen_status = "✅ Loaded" if GEN_MODEL_AVAILABLE else "❌ Not available"
except:
    qa_status = "❓ Unknown"
    gen_status = "❓ Unknown"

try:
    from ranking.reranker import MODEL_AVAILABLE as RERANK_AVAILABLE
    rerank_status = "✅ Loaded" if RERANK_AVAILABLE else "❌ Not available"
except:
    rerank_status = "❓ Unknown"

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Embedding", emb_status)
with col2:
    st.metric("QA Model", qa_status)
with col3:
    st.metric("Generator", gen_status)
with col4:
    st.metric("Reranker", rerank_status)

# =============================================================================
# Keyboard Shortcuts
# =============================================================================

st.markdown("---")
st.subheader("⌨️ Keyboard Shortcuts")

shortcuts = [
    ("Enter", "Submit question"),
    ("Ctrl + K", "Focus search (browser)"),
    ("Tab", "Navigate between elements"),
    ("Space", "Toggle expanders"),
]

col1, col2 = st.columns(2)
for i, (key, action) in enumerate(shortcuts):
    with col1 if i % 2 == 0 else col2:
        st.markdown(f"**`{key}`** — {action}")

# =============================================================================
# Tips & Best Practices
# =============================================================================

st.markdown("---")
st.subheader("💡 Tips & Best Practices")

st.markdown("""
**For Best Results:**

1. **Be Specific** - Instead of "Tell me about the document", ask "What are the main conclusions in section 3?"

2. **Use Keywords** - Include specific terms from your document for more accurate retrieval.

3. **Check Sources** - Always verify answers against the source citations provided.

4. **Use Follow-ups** - The suggested questions help explore topics in depth.

5. **Large Documents** - For very large documents, split them into smaller files by topic.

6. **Multiple Documents** - You can upload multiple related documents for cross-document queries.

**Troubleshooting:**

- If answers seem off, try rephrasing the question
- Use "Regenerate" to get a new answer
- Check the confidence score - lower scores may need verification
- Review sources to understand where information came from
""")
