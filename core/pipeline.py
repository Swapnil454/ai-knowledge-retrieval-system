import re
import difflib

from loaders.pdf_loader import load_pdf
from processing.chunker import chunk_text
from embeddings.embedder import embed_text
from vector_store.faiss_store import build_index, search_index
from ranking.reranker import rerank
from ranking.bm25_search import BM25Search
from services.qa_service import generate_answer
from core.config import TOP_K_RETRIEVAL, TOP_K_CONTEXT
from core.cache import SemanticCache
from services.query_expansion import generate_queries


cache = SemanticCache()


# -------------------------
# Build FAISS Index
# -------------------------

def build_document_index(file):

    pages = load_pdf(file)

    chunks = chunk_text(pages)

    texts = [chunk["text"] for chunk in chunks]

    embeddings = embed_text(texts)

    index = build_index(embeddings)

    return chunks, embeddings, index


# -------------------------
# Row Extraction Helper
# -------------------------

def extract_rows(text):

    rows = re.split(r'\s(?=\d+\s)', text)

    clean_rows = []

    for row in rows:
        row = row.strip()

        if len(row) > 5:
            clean_rows.append(row)

    return clean_rows


# -------------------------
# Row Matching
# -------------------------

def find_best_row(question, texts):

    query = question.lower()

    best_row = ""
    best_score = 0

    for text in texts:

        rows = extract_rows(text)

        for row in rows:

            score = difflib.SequenceMatcher(
                None,
                query,
                row.lower()
            ).ratio()

            if score > best_score:

                best_score = score
                best_row = row

    if best_row:
        return best_row

    return "No matching row found."


# -------------------------
# Answer Question
# -------------------------

def answer_question(question, chunks, embeddings, index):

    queries = generate_queries(question)

    query_embeddings = embed_text(queries)

    primary_query_embedding = query_embeddings[0]

    # -------------------------
    # Semantic Cache
    # -------------------------

    is_question = (
            "?" in question or
            question.lower().startswith(("who", "what", "when", "where", "why", "how"))
    )

    if is_question:
        cached = cache.search(primary_query_embedding)

        if cached:
            return cached, "", [], 1.0

    # -------------------------
    # Vector Search
    # -------------------------

    vector_results = []
    distances = None

    for query_emb in query_embeddings:

        distances, vector_indices = search_index(
            index,
            [query_emb],
            TOP_K_RETRIEVAL
        )

        results = [chunks[i] for i in vector_indices[0]]

        vector_results.extend(results)

    # -------------------------
    # BM25 Search
    # -------------------------

    bm25 = BM25Search(chunks)

    bm25_indices = bm25.search(question, TOP_K_RETRIEVAL)

    bm25_results = [chunks[i] for i in bm25_indices]

    # -------------------------
    # Combine Results
    # -------------------------

    combined = vector_results + bm25_results

    unique_chunks = []
    seen = set()

    for chunk in combined:

        text = chunk["text"]

        if text not in seen:

            seen.add(text)
            unique_chunks.append(chunk)

    # -------------------------
    # Rerank
    # -------------------------

    texts = [chunk["text"] for chunk in unique_chunks]

    reranked_texts = rerank(question, texts)

    # -------------------------
    # Build Context
    # -------------------------

    context = "\n".join(reranked_texts[:TOP_K_CONTEXT])

    # -------------------------
    # Generate Answer
    # -------------------------

    if "?" in question or question.lower().startswith(
        ("who", "what", "when", "where", "why", "how")
    ):

        answer = generate_answer(question, context)

    else:

        answer = find_best_row(question, reranked_texts)

        # highlight query words
        words = re.findall(r'\w+', question.lower())
        for word in words:

            answer = re.sub(
                word,
                f"**{word}**",
                answer,
                flags=re.IGNORECASE
            )

    # -------------------------
    # Save to Cache
    # -------------------------

    if is_question:
        cache.add(primary_query_embedding, answer)
    # -------------------------
    # Confidence
    # -------------------------

    confidence = float(1 / (1 + distances[0][0])) if distances is not None else 0.0

    return answer, context, unique_chunks, confidence