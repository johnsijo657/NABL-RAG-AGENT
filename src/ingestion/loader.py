import pymupdf
import logging
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

def load_pdf(file_path: str) -> List[Dict[str, Any]]:
    """
    Loads a PDF file and extracts text page by page.
    Returns a list of dictionaries, where each dict represents a page.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    pages = []
    try:
        doc = pymupdf.open(file_path)
        total_pages = len(doc)
        
        for i, page in enumerate(doc):
            text = page.get_text("text").strip()
            # Basic cleanup: remove excessive newlines
            text = " ".join(text.split())
            
            pages.append({
                "page_number": i + 1,
                "content": text,
                "source_file": path.name,
                "total_pages": total_pages
            })
            
        doc.close()
        logger.info(f"Loaded {total_pages} pages from {path.name}")
        return pages
        
    except Exception as e:
        logger.error(f"Error loading PDF {file_path}: {str(e)}")
        raise
