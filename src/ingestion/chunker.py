from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict, Any
from src.config import settings
import logging

logger = logging.getLogger(__name__)

def chunk_pages(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Takes a list of page dictionaries and splits their content into smaller,
    clause-aware chunks with contextual metadata headers.
    Preserves page number and source file metadata for each chunk.
    """
    # Clause-aware separators for standards documents (ISO/IEC 17025, NABL policies)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=[
            "\n\n\n",
            "\n\n",
            r"\n(?=[0-9]+\.[0-9]+)",    # Numbered clauses (e.g. 7.8, 6.5)
            r"\n(?=[a-z]\))",            # Sub-clause requirement items (e.g. a), b), c))
            "\n",
            ";\n",
            ". ",
            " ",
            ""
        ],
        is_separator_regex=True
    )
    
    chunks = []
    for page in pages:
        if not page.get("content"):
            continue
            
        page_chunks = text_splitter.split_text(page["content"])
        doc_name = page.get("source_file", "Unknown Document")
        page_num = page.get("page_number", 1)
        
        for chunk_text in page_chunks:
            cleaned_text = chunk_text.strip()
            if not cleaned_text:
                continue
                
            # Add contextual document header to improve embedding quality and BM25 search
            contextual_content = f"[{doc_name} | Page {page_num}]\n{cleaned_text}"
            
            chunks.append({
                "content": contextual_content,
                "page_number": page_num,
                "source_file": doc_name
            })
            
    logger.info(f"Generated {len(chunks)} clause-aware chunks from {len(pages)} pages.")
    return chunks
