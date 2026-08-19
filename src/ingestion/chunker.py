from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict, Any
from src.config import settings

def chunk_pages(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Takes a list of page dictionaries and splits their content into smaller chunks.
    Preserves metadata (page number, source file) for each chunk.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    
    chunks = []
    for page in pages:
        if not page["content"]:
            continue
            
        page_chunks = text_splitter.split_text(page["content"])
        
        for chunk_text in page_chunks:
            chunks.append({
                "content": chunk_text,
                "page_number": page["page_number"],
                "source_file": page["source_file"]
            })
            
    return chunks
