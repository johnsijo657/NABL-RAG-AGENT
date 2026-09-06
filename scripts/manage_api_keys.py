import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import secrets
from datetime import datetime, timedelta, timezone
from src.database import SessionLocal
from src.models import User, ApiKey
from src.api.auth import hash_api_key

def create_key(username: str, name: str, rate_limit: int, days_valid: int = None):
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.username == username).first()
        if not user:
            existing = [u.username for u in db.query(User).all()]
            print(f"[ERROR] User '{username}' does not exist.")
            print(f"Available users in database: {', '.join(existing)}")
            print(f"Tip: Use an existing username like '--username test_api_admin' or '--username admin'.")
            return


        # Generate cryptographically secure API key
        random_token = secrets.token_hex(24)
        raw_key = f"nabl_live_{random_token}"
        key_prefix = raw_key[:14]  # e.g., "nabl_live_a1b2"
        key_hash = hash_api_key(raw_key)

        expires_at = None
        if days_valid:
            expires_at = datetime.now(timezone.utc) + timedelta(days=days_valid)

        new_key = ApiKey(
            name=name,
            key_prefix=key_prefix,
            key_hash=key_hash,
            owner_id=user.id,
            rate_limit=rate_limit,
            is_active=True,
            expires_at=expires_at,
        )
        db.add(new_key)
        db.commit()
        db.refresh(new_key)

        print("\n" + "=" * 65)
        print("[SUCCESS] NEW API KEY GENERATED SUCCESSFULLY")
        print("=" * 65)
        print(f"Key Secret   : {raw_key}")
        print("[IMPORTANT]  : Copy this key now! It is NOT stored in plaintext and cannot be recovered.")
        print("-" * 65)
        print(f"Key ID       : {new_key.id}")
        print(f"Name         : {new_key.name}")
        print(f"Prefix       : {new_key.key_prefix}...")
        print(f"Owner        : {user.username} (ID: {user.id})")
        print(f"Rate Limit   : {new_key.rate_limit} requests/minute")
        print(f"Expires At   : {new_key.expires_at or 'Never (Permanent until revoked)'}")
        print("=" * 65 + "\n")
    finally:
        db.close()

def list_keys():
    db = SessionLocal()
    try:
        keys = db.query(ApiKey).join(User).all()
        if not keys:
            print("No API keys found in database.")
            return

        print("\n" + "=" * 80)
        print(f"{'ID':<4} | {'Prefix':<16} | {'Name':<24} | {'Owner':<12} | {'Limit':<7} | {'Active':<6} | {'Last Used'}")
        print("-" * 80)
        for k in keys:
            last_used = k.last_used_at.strftime("%Y-%m-%d %H:%M") if k.last_used_at else "Never"
            active_str = "YES" if k.is_active else "REVOKED"
            print(f"{k.id:<4} | {k.key_prefix:<16} | {k.name[:23]:<24} | {k.owner.username[:11]:<12} | {k.rate_limit:<7} | {active_str:<6} | {last_used}")
        print("=" * 80 + "\n")
    finally:
        db.close()

def revoke_key(key_id: int):
    db = SessionLocal()
    try:
        key = db.query(ApiKey).filter(ApiKey.id == key_id).first()
        if not key:
            print(f"[ERROR] API Key with ID {key_id} not found.")
            return

        key.is_active = False
        db.commit()
        print(f"[REVOKED] Successfully revoked API Key ID {key_id} ('{key.name}'). Any incoming requests will be rejected with HTTP 403.")
    finally:
        db.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NABL RAG Agent API Key Management CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # create
    create_parser = subparsers.add_parser("create", help="Create a new developer API key")
    create_parser.add_argument("--username", required=True, help="Username of the key owner")
    create_parser.add_argument("--name", required=True, help="Descriptive name for the integration (e.g. 'Hematology Lab LIMS')")
    create_parser.add_argument("--rate-limit", type=int, default=30, help="Allowed requests per minute (default: 30)")
    create_parser.add_argument("--days", type=int, default=None, help="Validity in days (default: None = infinite)")

    # list
    subparsers.add_parser("list", help="List all API keys")

    # revoke
    revoke_parser = subparsers.add_parser("revoke", help="Revoke an existing API key")
    revoke_parser.add_argument("--id", "--key-id", dest="id", type=int, required=True, help="ID of the API key to revoke")

    args = parser.parse_args()

    if args.command == "create":
        create_key(args.username, args.name, args.rate_limit, args.days)
    elif args.command == "list":
        list_keys()
    elif args.command == "revoke":
        revoke_key(args.id)
