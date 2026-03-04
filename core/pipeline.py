import re
import difflib
import logging

from loaders.pdf_loader import load_pdf
from processing.chunker import chunk_text
from processing.cleaner import clean_text
from embeddings.embedder import embed_text
from vector_store.faiss_store import build_index, search_index, merge_indices
from ranking.reranker import rerank
from ranking.bm25_search import BM25Search
from services.qa_service import generate_answer
from core.config import TOP_K_RETRIEVAL, TOP_K_CONTEXT
from core.cache import SemanticCache
from services.query_expansion import generate_queries

logger = logging.getLogger(__name__)
cache = SemanticCache()


def clear_cache():
    """Clear the semantic cache."""
    cache.clear()


# -------------------------
# Build FAISS Index
# -------------------------

def build_document_index(file):
    """
    Process a PDF file and build a FAISS index.
    
    Args:
        file: Uploaded PDF file object
        
    Returns:
        Tuple of (chunks, embeddings, index)
    """
    try:
        pages = load_pdf(file)
        
        if not pages:
            raise ValueError(f"No text could be extracted from {file.name}")

        chunks = chunk_text(pages)
        
        # Clean the text in each chunk
        for chunk in chunks:
            chunk["text"] = clean_text(chunk["text"])

        texts = [chunk["text"] for chunk in chunks]

        embeddings = embed_text(texts)

        index = build_index(embeddings)

        logger.info(f"Processed {file.name}: {len(chunks)} chunks created")
        return chunks, embeddings, index
        
    except Exception as e:
        logger.error(f"Error processing document {file.name}: {e}")
        raise


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

def answer_question(question, chunks, embeddings, index, document_hash=None):
    """
    Answer a question using the RAG pipeline.
    
    Args:
        question: User's question string
        chunks: List of document chunks
        embeddings: List of chunk embeddings
        index: FAISS index
        document_hash: Optional hash to track document changes for cache invalidation
        
    Returns:
        Tuple of (answer, context, unique_chunks, confidence)
    """
    try:
        # Update cache document hash if provided
        if document_hash:
            cache.set_document_hash(document_hash)
        
        queries = generate_queries(question)

        query_embeddings = embed_text(queries)

        primary_query_embedding = query_embeddings[0]

        # -------------------------
        # Check if this is a question
        # -------------------------
        is_question = (
            "?" in question or
            question.lower().startswith(("who", "what", "when", "where", "why", "how", "is", "are", "can", "could", "would", "should", "do", "does"))
        )

        # -------------------------
        # Semantic Cache Check
        # -------------------------
        if is_question:
            cached = cache.search(primary_query_embedding)
            if cached:
                logger.info("Cache hit - returning cached answer")
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

            results = [chunks[i] for i in vector_indices[0] if i < len(chunks)]

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

        if is_question:
            answer = generate_answer(question, context)
        else:
            answer = find_best_row(question, reranked_texts)

            # highlight query words
            words = re.findall(r'\w+', question.lower())
            for word in words:
                if len(word) > 2:  # Only highlight longer words
                    answer = re.sub(
                        rf'\b{word}\b',
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

        confidence = float(1 / (1 + distances[0][0])) if distances is not None and len(distances[0]) > 0 else 0.0

        return answer, context, unique_chunks, confidence
        
    except Exception as e:
        logger.error(f"Error in answer_question: {e}")
        raise