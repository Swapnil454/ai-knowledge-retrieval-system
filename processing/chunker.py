from core.config import CHUNK_SIZE, CHUNK_OVERLAP
import logging

logger = logging.getLogger(__name__)


def chunk_text(pages: list, size: int = None, overlap: int = None) -> list:
    """
    Split document pages into overlapping chunks.
    
    Args:
        pages: List of page dicts with 'text', 'page', 'source' keys
        size: Chunk size in words (default from config)
        overlap: Overlap size in words (default from config)
        
    Returns:
        List of chunk dicts with text, page, and source info
    """
    size = size or CHUNK_SIZE
    overlap = overlap or CHUNK_OVERLAP
    
    if overlap >= size:
        raise ValueError("Overlap must be less than chunk size")
    
    chunks = []
    step = size - overlap

    for page in pages:
        text = page.get("text", "")
        
        if not text.strip():
            continue
            
        words = text.split()
        
        # If page is smaller than chunk size, keep it as one chunk
        if len(words) <= size:
            chunks.append({
                "text": text,
                "page": page.get("page", 0),
                "source": page.get("source", "unknown")
            })
            continue

        for i in range(0, len(words), step):
            chunk_words = words[i:i + size]
            chunk_text = " ".join(chunk_words)
            
            if chunk_text.strip():  # Only add non-empty chunks
                chunks.append({
                    "text": chunk_text,
                    "page": page.get("page", 0),
                    "source": page.get("source", "unknown")
                })

    logger.info(f"Created {len(chunks)} chunks from {len(pages)} pages")
    return chunks