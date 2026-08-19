import sys
from pathlib import Path

# Add project root to python path
project_root = str(Path(__file__).parent.parent.absolute())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.database import SessionLocal
from src.models import User
from src.api.auth import get_password_hash
import os

def create_users():
    db = SessionLocal()
    users_to_create = ["admin", "tester"]
    password = os.getenv("DEFAULT_USER_PASSWORD", "changeme123")
    
    try:
        for username in users_to_create:
            existing = db.query(User).filter(User.username == username).first()
            if existing:
                print(f"User '{username}' already exists.")
                continue
                
            hashed_password = get_password_hash(password)
            user = User(username=username, hashed_password=hashed_password)
            db.add(user)
            db.commit()
            print(f"User '{username}' created successfully.")
    except Exception as e:
        print(f"Error creating users: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    create_users()
