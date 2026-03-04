"""
Multi-format Document Loader
Supports: PDF, TXT, DOCX
"""
from pypdf import PdfReader
import logging
import os

logger = logging.getLogger(__name__)


def load_pdf(file):
    """
    Extract text from various document formats.
    
    Args:
        file: File object (uploaded file or file path)
        
    Returns:
        List of dicts with text, page number, and source
    """
    filename = getattr(file, 'name', str(file))
    extension = os.path.splitext(filename)[1].lower()
    
    if extension == '.pdf':
        return _load_pdf(file)
    elif extension == '.txt':
        return _load_txt(file)
    elif extension == '.docx':
        return _load_docx(file)
    else:
        logger.warning(f"Unsupported file format: {extension}")
        raise ValueError(f"Unsupported file format: {extension}")


def _load_pdf(file) -> list:
    """Load PDF document."""
    try:
        if hasattr(file, 'seek'):
            file.seek(0)

        reader = PdfReader(file)
        
        if len(reader.pages) == 0:
            logger.warning(f"PDF {file.name} has no pages")
            return []

        pages = []
        filename = getattr(file, 'name', 'unknown')

        for i, page in enumerate(reader.pages):
            try:
                text = page.extract_text()
                if text and text.strip():
                    pages.append({
                        "text": text,
                        "page": i + 1,
                        "source": filename
                    })
            except Exception as e:
                logger.warning(f"Could not extract text from page {i+1}: {e}")
                continue

        logger.info(f"Extracted {len(pages)} pages from PDF: {filename}")
        return pages
        
    except Exception as e:
        logger.error(f"Error loading PDF: {e}")
        raise ValueError(f"Could not read PDF file: {str(e)}")


def _load_txt(file) -> list:
    """Load plain text document."""
    try:
        if hasattr(file, 'seek'):
            file.seek(0)
        
        filename = getattr(file, 'name', 'unknown')
        
        # Read content
        if hasattr(file, 'read'):
            content = file.read()
            if isinstance(content, bytes):
                content = content.decode('utf-8', errors='ignore')
        else:
            with open(file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        
        if not content.strip():
            return []
        
        # Split into pages (every ~2000 chars or by section markers)
        pages = []
        chunk_size = 2000
        
        # Try to split by natural breaks first
        sections = content.split('\n\n\n')
        
        if len(sections) > 1:
            for i, section in enumerate(sections):
                if section.strip():
                    pages.append({
                        "text": section.strip(),
                        "page": i + 1,
                        "source": filename
                    })
        else:
            # Split by size
            for i in range(0, len(content), chunk_size):
                chunk = content[i:i + chunk_size]
                if chunk.strip():
                    pages.append({
                        "text": chunk.strip(),
                        "page": (i // chunk_size) + 1,
                        "source": filename
                    })
        
        logger.info(f"Extracted {len(pages)} sections from TXT: {filename}")
        return pages
        
    except Exception as e:
        logger.error(f"Error loading TXT: {e}")
        raise ValueError(f"Could not read TXT file: {str(e)}")


def _load_docx(file) -> list:
    """Load DOCX document."""
    try:
        try:
            from docx import Document
        except ImportError:
            logger.warning("python-docx not installed. Install with: pip install python-docx")
            raise ValueError("DOCX support requires python-docx: pip install python-docx")
        
        if hasattr(file, 'seek'):
            file.seek(0)
        
        filename = getattr(file, 'name', 'unknown')
        doc = Document(file)
        
        pages = []
        current_text = []
        page_num = 1
        char_count = 0
        
        for para in doc.paragraphs:
            text = para.text.strip()
            if text:
                current_text.append(text)
                char_count += len(text)
                
                # Create a new "page" every ~2000 chars
                if char_count >= 2000:
                    pages.append({
                        "text": "\n".join(current_text),
                        "page": page_num,
                        "source": filename
                    })
                    current_text = []
                    char_count = 0
                    page_num += 1
        
        # Add remaining text
        if current_text:
            pages.append({
                "text": "\n".join(current_text),
                "page": page_num,
                "source": filename
            })
        
        logger.info(f"Extracted {len(pages)} sections from DOCX: {filename}")
        return pages
        
    except ImportError:
        raise
    except Exception as e:
        logger.error(f"Error loading DOCX: {e}")
        raise ValueError(f"Could not read DOCX file: {str(e)}")