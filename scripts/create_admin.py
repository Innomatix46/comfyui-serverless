#!/usr/bin/env python3
"""Create admin user and API token for testing."""

import asyncio
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sqlalchemy.orm import Session
from src.core.database import SessionLocal, create_tables
from src.models.database import User, APIKey
from src.core.security import get_password_hash, create_api_key
from datetime import datetime, timedelta


async def create_admin_user():
    """Create admin user and API key."""
    # Ensure tables exist
    create_tables()
    
    with SessionLocal() as db:
        # Check if admin user exists
        admin_user = db.query(User).filter(User.email == "admin@comfyui.local").first()
        
        if not admin_user:
            # Create admin user
            admin_user = User(
                email="admin@comfyui.local",
                username="admin",
                hashed_password=get_password_hash("admin123"),
                is_active=True,
                is_superuser=True
            )
            db.add(admin_user)
            db.commit()
            db.refresh(admin_user)
            print("✅ Admin user created: admin@comfyui.local (password: admin123)")
        else:
            print("✅ Admin user already exists")
        
        # Create or get API key
        existing_key = db.query(APIKey).filter(
            APIKey.user_id == admin_user.id,
            APIKey.name == "test-key"
        ).first()
        
        if not existing_key:
            api_key_data = create_api_key()
            api_key = APIKey(
                user_id=admin_user.id,
                key_hash=api_key_data["key_hash"],
                name="test-key",
                is_active=True,
                expires_at=datetime.utcnow() + timedelta(days=365)
            )
            db.add(api_key)
            db.commit()
            
            print(f"✅ API Key created: {api_key_data['key']}")
            print(f"📋 Use this token in Authorization header: Bearer {api_key_data['key']}")
            
            # Write to file for easy access
            with open("api_token.txt", "w") as f:
                f.write(api_key_data['key'])
            print("💾 Token saved to api_token.txt")
        else:
            print("✅ API key already exists - check api_token.txt")


if __name__ == "__main__":
    asyncio.run(create_admin_user())