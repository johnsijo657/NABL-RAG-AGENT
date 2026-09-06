import pymupdf
import logging
import base64
import re
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class PDFTooLargeError(Exception):
    pass

def load_pdf(file_path: str, extract_images: bool = True) -> List[Dict[str, Any]]:
    """
    Loads a PDF file and extracts text page by page.
    Returns a list of dictionaries, where each dict represents a page.
    If extract_images is True, it also renders each page to a base64 encoded PNG image.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    pages = []
    try:
        doc = pymupdf.open(file_path)
        total_pages = len(doc)
        
        # if total_pages > 10:
        #     doc.close()
        #     raise PDFTooLargeError(f"PDF exceeds the maximum allowed size of 10 pages. This document has {total_pages} pages.")
        
        for i, page in enumerate(doc):
            text = page.get_text("text")
            # Clean up whitespace per line while preserving paragraph structure
            lines = [line.strip() for line in text.splitlines()]
            text = "\n".join(lines)
            text = re.sub(r'\n{3,}', '\n\n', text).strip()
            
            page_data = {
                "page_number": i + 1,
                "content": text,
                "source_file": path.name,
                "total_pages": total_pages
            }
            
            if extract_images:
                # Render page to an image (scale of 1.0 to save tokens on Cloud APIs)
                matrix = pymupdf.Matrix(1.0, 1.0)
                pix = page.get_pixmap(matrix=matrix)
                img_data = pix.tobytes("png")
                b64_img = base64.b64encode(img_data).decode("utf-8")
                page_data["image_base64"] = b64_img
            
            pages.append(page_data)
            
        doc.close()
        logger.info(f"Loaded {total_pages} pages from {path.name}")
        return pages
        
    except PDFTooLargeError:
        raise
    except Exception as e:
        logger.error(f"Error loading PDF {file_path}: {str(e)}")
        raise
