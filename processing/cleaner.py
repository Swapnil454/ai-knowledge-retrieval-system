"""
Text cleaning utilities for preprocessing document content.
"""
import re
import unicodedata


def clean_text(text: str) -> str:
    """
    Clean and normalize text extracted from documents.
    
    Args:
        text: Raw text to clean
        
    Returns:
        Cleaned and normalized text
    """
    if not text:
        return ""
    
    # Normalize unicode characters
    text = unicodedata.normalize("NFKC", text)
    
    # Remove excessive whitespace and newlines
    text = re.sub(r'\n{3,}', '\n\n', text)  # Max 2 consecutive newlines
    text = re.sub(r'[ \t]{2,}', ' ', text)  # Collapse multiple spaces
    
    # Remove page numbers (common patterns)
    text = re.sub(r'\n\d+\n', '\n', text)
    text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
    
    # Remove common PDF artifacts
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    
    # Fix common OCR errors
    text = text.replace('ﬁ', 'fi').replace('ﬂ', 'fl')
    text = text.replace('™', '').replace('®', '')
    
    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+\.\S+', '', text)
    
    # Normalize quotes and dashes
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace(''', "'").replace(''', "'")
    text = text.replace('–', '-').replace('—', '-')
    
    # Remove excessive punctuation
    text = re.sub(r'\.{4,}', '...', text)
    text = re.sub(r'-{3,}', '--', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def remove_stopwords(text: str, custom_stopwords: list = None) -> str:
    """
    Remove common stopwords from text.
    
    Args:
        text: Text to process
        custom_stopwords: Additional stopwords to remove
        
    Returns:
        Text with stopwords removed
    """
    # Basic English stopwords
    stopwords = {
        'a', 'an', 'the', 'and', 'or', 'but', 'is', 'are', 'was', 'were',
        'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'as', 'be',
        'this', 'that', 'it', 'its', 'from', 'has', 'have', 'had'
    }
    
    if custom_stopwords:
        stopwords.update(custom_stopwords)
    
    words = text.split()
    filtered = [w for w in words if w.lower() not in stopwords]
    
    return ' '.join(filtered)


def extract_sentences(text: str) -> list:
    """
    Split text into sentences.
    
    Args:
        text: Text to split
        
    Returns:
        List of sentences
    """
    # Split on sentence endings
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # Filter out very short sentences
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    
    return sentences