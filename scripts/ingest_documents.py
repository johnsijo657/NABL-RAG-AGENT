import os
import sys
import logging
from pathlib import Path

# Add project root to python path to allow importing src
project_root = str(Path(__file__).parent.parent.absolute())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.database import SessionLocal
from src.models import Document, DocumentChunk
from src.ingestion.loader import load_pdf
from src.ingestion.chunker import chunk_pages
from src.ingestion.embedder import generate_embeddings
from src.retrieval.bm25_store import bm25_retriever

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DOCUMENTS_DIR = Path(project_root) / "NABL_DOCUMENTS"

def ingest_directory():
    db = SessionLocal()
    ingested_count = 0
    try:
        if not DOCUMENTS_DIR.exists():
            logger.error(f"Directory not found: {DOCUMENTS_DIR}")
            return
            
        pdf_files = list(DOCUMENTS_DIR.glob("*.pdf"))
        if not pdf_files:
            logger.info(f"No PDF files found in {DOCUMENTS_DIR}")
            return
            
        logger.info(f"Found {len(pdf_files)} PDFs in {DOCUMENTS_DIR}")
        
        for pdf_path in pdf_files:
            filename = pdf_path.name
            try:
                # Smart checking: see if document already exists
                existing_doc = db.query(Document).filter(Document.filename == filename).first()
                if existing_doc:
                    logger.info(f"Skipping {filename}: Already ingested (ID: {existing_doc.id}).")
                    continue
                    
                logger.info(f"Processing new document: {filename}")
                
                # 1. Load PDF
                pages = load_pdf(str(pdf_path))
                if not pages:
                    logger.warning(f"No pages extracted from {filename}.")
                    continue
                    
                total_pages = pages[0]["total_pages"]
                
                # 2. Chunk text
                chunks_data = chunk_pages(pages)
                logger.info(f"Created {len(chunks_data)} chunks from {filename}")
                
                if not chunks_data:
                    logger.warning(f"No chunks generated for {filename}.")
                    continue
                    
                # 3. Generate embeddings
                texts_to_embed = [chunk["content"] for chunk in chunks_data]
                embeddings = generate_embeddings(texts_to_embed)
                
                # 4. Save to Database
                # Create Document record
                db_doc = Document(filename=filename, total_pages=total_pages)
                db.add(db_doc)
                db.flush() # flush to get db_doc.id
                
                # Create DocumentChunk records
                db_chunks = []
                for chunk_data, embedding in zip(chunks_data, embeddings):
                    db_chunks.append(
                        DocumentChunk(
                            document_id=db_doc.id,
                            page_number=chunk_data["page_number"],
                            content=chunk_data["content"],
                            embedding=embedding
                        )
                    )
                    
                db.add_all(db_chunks)
                db.commit()
                ingested_count += 1
                logger.info(f"Successfully ingested {filename} into database.")
                
            except Exception as e:
                db.rollback()
                logger.error(f"Failed to ingest {filename}: {str(e)}")
                
        if ingested_count > 0:
            bm25_retriever.invalidate()
            logger.info(f"Ingested {ingested_count} new document(s). BM25 index invalidated for next query.")
            
    finally:
        db.close()

if __name__ == "__main__":
    ingest_directory()
