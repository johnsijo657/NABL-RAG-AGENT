from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text
from src.database import get_db

router = APIRouter(prefix="/api/v1", tags=["health"])

@router.get("/health")
def health_check(db: Session = Depends(get_db)):
    # Check DB connectivity
    db_status = "ok"
    try:
        db.execute(text("SELECT 1"))
    except Exception:
        db_status = "failed"
        
    return {
        "status": "online",
        "database": db_status
    }
