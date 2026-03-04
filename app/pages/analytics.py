"""
Analytics Dashboard - Query Insights & Usage Statistics
"""
import streamlit as st
import sys
import os
import json
from datetime import datetime, timedelta
from collections import Counter

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from evaluation.query_logs import get_logs

st.set_page_config(
    page_title="Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Analytics Dashboard")
st.caption("Query insights and usage statistics")

# =============================================================================
# Load Data
# =============================================================================

logs = get_logs(limit=1000)

if not logs:
    st.info("No query data available yet. Start using the AI Assistant to generate analytics.")
    st.stop()

# =============================================================================
# Overview Metrics
# =============================================================================

st.markdown("---")
st.subheader("📈 Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Queries", len(logs))

with col2:
    # Queries today
    today = datetime.now().date()
    today_queries = sum(1 for log in logs if log.get("timestamp", "")[:10] == str(today))
    st.metric("Queries Today", today_queries)

with col3:
    # Average confidence
    confidences = [log.get("confidence", 0) for log in logs if log.get("confidence")]
    avg_conf = sum(confidences) / len(confidences) if confidences else 0
    st.metric("Avg Confidence", f"{avg_conf:.0%}")

with col4:
    # Unique days
    unique_days = len(set(log.get("timestamp", "")[:10] for log in logs))
    st.metric("Active Days", unique_days)

# =============================================================================
# Query Timeline
# =============================================================================

st.markdown("---")
st.subheader("📅 Query Timeline")

# Group queries by date
date_counts = Counter(log.get("timestamp", "")[:10] for log in logs)
if date_counts:
    dates = sorted(date_counts.keys())[-30:]  # Last 30 days
    counts = [date_counts.get(d, 0) for d in dates]
    
    chart_data = {"Date": dates, "Queries": counts}
    st.bar_chart(chart_data, x="Date", y="Queries")

# =============================================================================
# Recent Queries
# =============================================================================

st.markdown("---")
st.subheader("🕐 Recent Queries")

for log in logs[:10]:
    with st.expander(f"❓ {log.get('question', 'N/A')[:80]}..."):
        st.markdown(f"**Question:** {log.get('question', 'N/A')}")
        st.markdown(f"**Answer:** {log.get('answer', 'N/A')[:500]}...")
        
        col1, col2 = st.columns(2)
        with col1:
            st.caption(f"📅 {log.get('timestamp', 'N/A')[:19]}")
        with col2:
            conf = log.get("confidence")
            if conf:
                st.caption(f"📊 Confidence: {conf:.0%}")

# =============================================================================
# Top Topics
# =============================================================================

st.markdown("---")
st.subheader("🏷️ Common Words in Queries")

# Extract common words
all_words = []
stop_words = {'what', 'how', 'why', 'when', 'where', 'who', 'is', 'are', 'the', 'a', 'an', 'in', 'on', 'to', 'for', 'of', 'and', 'or', 'this', 'that', 'it', 'me', 'my', 'can', 'you', 'do', 'does', '?'}

for log in logs:
    question = log.get("question", "").lower()
    words = [w.strip('.,?!') for w in question.split() if len(w) > 3 and w.lower() not in stop_words]
    all_words.extend(words)

word_counts = Counter(all_words).most_common(20)

if word_counts:
    cols = st.columns(5)
    for i, (word, count) in enumerate(word_counts[:10]):
        with cols[i % 5]:
            st.metric(word.capitalize(), count)

# =============================================================================
# Export Options
# =============================================================================

st.markdown("---")
st.subheader("📥 Export Data")

col1, col2 = st.columns(2)

with col1:
    # Export as JSON
    export_json = json.dumps(logs, indent=2)
    st.download_button(
        "📋 Download JSON",
        data=export_json,
        file_name=f"analytics_{datetime.now().strftime('%Y%m%d')}.json",
        mime="application/json",
        use_container_width=True
    )

with col2:
    # Export as CSV
    csv_lines = ["timestamp,question,answer,confidence"]
    for log in logs:
        q = log.get("question", "").replace('"', "'")
        a = log.get("answer", "")[:100].replace('"', "'").replace('\n', ' ')
        c = log.get("confidence", "")
        t = log.get("timestamp", "")
        csv_lines.append(f'"{t}","{q}","{a}","{c}"')
    
    st.download_button(
        "📊 Download CSV",
        data="\n".join(csv_lines),
        file_name=f"analytics_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        use_container_width=True
    )
