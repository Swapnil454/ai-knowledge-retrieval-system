import streamlit as st
import sys
import os
import time


sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from core.pipeline import build_document_index, answer_question
from core.memory import ConversationMemory

st.set_page_config(page_title="Enterprise RAG Assistant")

st.title("AI Knowledge Retrieval System")

memory = ConversationMemory()

uploaded_files = st.file_uploader(
    "Upload PDF documents",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files:

    all_chunks = []
    all_embeddings = []
    index = None

    for file in uploaded_files:

        chunks, embeddings, index = build_document_index(file)

        all_chunks.extend(chunks)
        all_embeddings.extend(embeddings)

    question = st.text_input("Ask a question")

    if question:

        answer, context, retrieved, confidence = answer_question(
            question,
            all_chunks,
            all_embeddings,
            index
        )

        memory.add("user", question)
        memory.add("assistant", answer)

        st.subheader("Answer")
        st.write(answer)

        st.sidebar.title("System Stats")

        st.sidebar.write(
            f"Documents indexed: {len(uploaded_files)}"
        )

        st.sidebar.write(
            f"Chunks stored: {len(all_chunks)}"
        )
        st.write(f"Confidence score: {confidence:.2f}")

        with st.expander("Sources"):

            for chunk in retrieved[:3]:
                st.write(
                    f"📄 {chunk['source']} — Page {chunk['page']}"
                )

        st.subheader("Chat History")

        for msg in memory.get():
            st.write(f"**{msg['role']}**: {msg['message']}")