"""
Document Manager - View, analyze, and manage indexed documents
"""
import streamlit as st
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

st.set_page_config(
    page_title="Document Manager",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Document Manager")
st.caption("View and analyze indexed documents")

# =============================================================================
# Check for indexed documents
# =============================================================================

if "processed_files" not in st.session_state or not st.session_state.processed_files:
    st.info("No documents indexed yet. Go to the main page to upload documents.")
    st.stop()

# =============================================================================
# Document Overview
# =============================================================================

st.markdown("---")
st.subheader("📊 Document Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Documents", len(st.session_state.processed_files))
    
with col2:
    total_chunks = len(st.session_state.get("all_chunks", []))
    st.metric("Total Chunks", total_chunks)

with col3:
    avg_chunks = total_chunks // len(st.session_state.processed_files) if st.session_state.processed_files else 0
    st.metric("Avg Chunks/Doc", avg_chunks)

# =============================================================================
# Document List
# =============================================================================

st.markdown("---")
st.subheader("📁 Indexed Documents")

for filename, metadata in st.session_state.processed_files.items():
    with st.expander(f"📄 {filename}", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Chunks", metadata.get("chunks", "N/A"))
        with col2:
            st.metric("Size", metadata.get("size", "N/A"))
        with col3:
            st.metric("Added", metadata.get("added", "N/A"))
        
        # Show chunks from this document
        doc_chunks = [c for c in st.session_state.get("all_chunks", []) if c.get("source") == filename]
        
        if doc_chunks:
            st.markdown("**Sample Content:**")
            
            # Show first 3 chunks
            for i, chunk in enumerate(doc_chunks[:3]):
                st.markdown(f"""
                **Page {chunk.get('page', 'N/A')}:**
                > {chunk.get('text', '')[:300]}...
                """)
            
            if len(doc_chunks) > 3:
                st.caption(f"...and {len(doc_chunks) - 3} more chunks")

# =============================================================================
# Chunk Browser
# =============================================================================

st.markdown("---")
st.subheader("🔍 Chunk Browser")

all_chunks = st.session_state.get("all_chunks", [])

if all_chunks:
    # Filters
    col1, col2 = st.columns(2)
    
    with col1:
        doc_filter = st.selectbox(
            "Filter by Document",
            options=["All"] + list(st.session_state.processed_files.keys())
        )
    
    with col2:
        search_term = st.text_input("Search in chunks", placeholder="Enter keyword...")
    
    # Apply filters
    filtered_chunks = all_chunks
    
    if doc_filter != "All":
        filtered_chunks = [c for c in filtered_chunks if c.get("source") == doc_filter]
    
    if search_term:
        filtered_chunks = [c for c in filtered_chunks if search_term.lower() in c.get("text", "").lower()]
    
    st.caption(f"Showing {len(filtered_chunks)} of {len(all_chunks)} chunks")
    
    # Display chunks
    for i, chunk in enumerate(filtered_chunks[:20]):
        with st.container():
            col1, col2 = st.columns([1, 5])
            with col1:
                st.caption(f"#{i+1}")
                st.caption(f"Page {chunk.get('page', 'N/A')}")
            with col2:
                text = chunk.get("text", "")
                if search_term:
                    # Highlight search term
                    import re
                    highlighted = re.sub(
                        f"({re.escape(search_term)})",
                        r"**\1**",
                        text,
                        flags=re.IGNORECASE
                    )
                    st.markdown(highlighted[:500] + "..." if len(text) > 500 else highlighted)
                else:
                    st.markdown(text[:500] + "..." if len(text) > 500 else text)
            st.markdown("---")
    
    if len(filtered_chunks) > 20:
        st.caption(f"Showing first 20 of {len(filtered_chunks)} chunks")
