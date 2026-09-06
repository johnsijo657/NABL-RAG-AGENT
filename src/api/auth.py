from datetime import datetime, timedelta, timezone
from typing import Optional
import jwt
from jwt.exceptions import InvalidTokenError
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from src.config import settings
from src.database import get_db
from src.models import User, ApiKey
from src.api.schemas import TokenData
from src.api.rate_limiter import rate_limiter
import hashlib
from fastapi.security import APIKeyHeader
from fastapi import Security

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/token")
api_key_header_scheme = APIKeyHeader(name="X-API-Key", auto_error=False)
auth_header_scheme = APIKeyHeader(name="Authorization", auto_error=False)

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def hash_api_key(raw_key: str) -> str:
    """Computes SHA-256 hash of a raw API key for secure storage and comparison."""
    return hashlib.sha256(raw_key.strip().encode("utf-8")).hexdigest()

def get_api_key(
    x_api_key: Optional[str] = Security(api_key_header_scheme),
    auth_header: Optional[str] = Security(auth_header_scheme),
    db: Session = Depends(get_db)
) -> ApiKey:
    """
    Validates the X-API-Key header upon request arrival.
    Verifies key validity, active status, expiration, and enforces the key's rate limit.
    """
    raw_key = None
    if x_api_key:
        raw_key = x_api_key.strip()
    elif auth_header:
        # Also support Authorization: Bearer <api_key> or Authorization: ApiKey <api_key>
        parts = auth_header.strip().split()
        if len(parts) == 2 and parts[0].lower() in ("bearer", "apikey"):
            raw_key = parts[1]
        elif len(parts) == 1:
            raw_key = parts[0]

    if not raw_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API Key. Please provide a valid developer API key in the 'X-API-Key' header.",
            headers={"WWW-Authenticate": "ApiKey"}
        )

    # Hash raw key to compare against database
    key_hash = hash_api_key(raw_key)
    api_key = db.query(ApiKey).filter(ApiKey.key_hash == key_hash).first()

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key provided.",
            headers={"WWW-Authenticate": "ApiKey"}
        )

    if not api_key.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This API Key has been revoked or deactivated."
        )

    if api_key.expires_at:
        # Normalize timezone for comparison
        expires = api_key.expires_at if api_key.expires_at.tzinfo else api_key.expires_at.replace(tzinfo=timezone.utc)
        if expires < datetime.now(timezone.utc):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="This API Key has expired."
            )

    # Enforce rate limit immediately upon request arrival
    rate_limiter.check_rate_limit(api_key.id, api_key.rate_limit)

    # Update last_used_at timestamp
    try:
        api_key.last_used_at = datetime.now(timezone.utc)
        db.commit()
    except Exception:
        db.rollback()

    return api_key

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt

def get_user(db: Session, username: str):
    return db.query(User).filter(User.username == username).first()

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except InvalidTokenError:
        raise credentials_exception
        
    user = get_user(db, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user
