"""Authentication service."""
import jwt
from datetime import datetime, timedelta
from passlib.context import CryptContext
from sqlalchemy.orm import Session
from typing import Optional, Dict
import structlog
import redis
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from src.config.settings import settings
from src.core.database import get_db, SessionLocal
from src.models.database import User

logger = structlog.get_logger()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()


class AuthService:
    """Authentication service class."""
    
    def __init__(self):
        self.redis_client = redis.from_url(settings.REDIS_URL)
        self.blacklist_key_prefix = "blacklist_token:"
        self.refresh_token_prefix = "refresh_token:"
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash."""
        return pwd_context.verify(plain_password, hashed_password)
    
    def get_password_hash(self, password: str) -> str:
        """Generate password hash."""
        return pwd_context.hash(password)
    
    async def create_user(
        self, 
        email: str, 
        password: str, 
        username: Optional[str] = None
    ) -> Optional[User]:
        """Create a new user."""
        try:
            with SessionLocal() as db:
                # Check if user already exists
                existing_user = db.query(User).filter(User.email == email).first()
                if existing_user:
                    logger.warning("User creation failed - email already exists", email=email)
                    return None
                
                # Create new user
                hashed_password = self.get_password_hash(password)
                user = User(
                    email=email,
                    username=username,
                    hashed_password=hashed_password,
                    is_active=True
                )
                
                db.add(user)
                db.commit()
                db.refresh(user)
                
                logger.info("User created successfully", user_id=user.id, email=email)
                return user
                
        except Exception as e:
            logger.error("Failed to create user", email=email, error=str(e))
            return None
    
    async def authenticate_user(self, email: str, password: str) -> Optional[User]:
        """Authenticate user with email and password."""
        try:
            with SessionLocal() as db:
                user = db.query(User).filter(User.email == email).first()
                
                if not user:
                    logger.warning("Authentication failed - user not found", email=email)
                    return None
                
                if not self.verify_password(password, user.hashed_password):
                    logger.warning("Authentication failed - invalid password", email=email)
                    return None
                
                if not user.is_active:
                    logger.warning("Authentication failed - user inactive", email=email)
                    return None
                
                logger.info("User authenticated successfully", user_id=user.id, email=email)
                return user
                
        except Exception as e:
            logger.error("Authentication error", email=email, error=str(e))
            return None
    
    async def create_access_token(self, user_id: int) -> str:
        """Create JWT access token."""
        try:
            expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
            
            payload = {
                "user_id": user_id,
                "exp": expire,
                "iat": datetime.utcnow(),
                "type": "access"
            }
            
            token = jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
            logger.info("Access token created", user_id=user_id)
            return token
            
        except Exception as e:
            logger.error("Failed to create access token", user_id=user_id, error=str(e))
            raise
    
    async def create_refresh_token(self, user_id: int) -> str:
        """Create JWT refresh token."""
        try:
            expire = datetime.utcnow() + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
            
            payload = {
                "user_id": user_id,
                "exp": expire,
                "iat": datetime.utcnow(),
                "type": "refresh"
            }
            
            token = jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
            
            # Store refresh token in Redis
            redis_key = f"{self.refresh_token_prefix}{user_id}:{token[-10:]}"
            self.redis_client.setex(
                redis_key, 
                settings.REFRESH_TOKEN_EXPIRE_DAYS * 24 * 3600, 
                token
            )
            
            logger.info("Refresh token created", user_id=user_id)
            return token
            
        except Exception as e:
            logger.error("Failed to create refresh token", user_id=user_id, error=str(e))
            raise
    
    async def verify_token(self, token: str) -> Optional[User]:
        """Verify JWT token and return user."""
        try:
            # Quick check if token looks like JWT (has dots)
            if '.' not in token:
                logger.debug("Token is not JWT format - no dots found")
                return None
            
            # Check if token is blacklisted
            if await self.is_token_blacklisted(token):
                logger.warning("Token verification failed - blacklisted token")
                return None
            
            # Decode JWT token
            payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
            user_id = payload.get("user_id")
            token_type = payload.get("type", "access")
            
            if not user_id or token_type != "access":
                logger.warning("Token verification failed - invalid payload")
                return None
            
            # Get user from database
            with SessionLocal() as db:
                user = db.query(User).filter(User.id == user_id).first()
                
                if not user or not user.is_active:
                    logger.warning("Token verification failed - user not found or inactive", user_id=user_id)
                    return None
                
                return user
                
        except jwt.ExpiredSignatureError:
            logger.warning("Token verification failed - expired token")
            return None
        except jwt.InvalidTokenError as e:
            logger.debug("JWT token verification failed - likely not a JWT", error=str(e))
            return None
        except Exception as e:
            logger.debug("Token verification error", error=str(e))
            return None
    
    async def refresh_tokens(self, refresh_token: str) -> Optional[Dict[str, str]]:
        """Refresh access and refresh tokens."""
        try:
            # Verify refresh token
            payload = jwt.decode(refresh_token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
            user_id = payload.get("user_id")
            token_type = payload.get("type", "access")
            
            if not user_id or token_type != "refresh":
                logger.warning("Token refresh failed - invalid payload")
                return None
            
            # Check if refresh token exists in Redis
            redis_key = f"{self.refresh_token_prefix}{user_id}:{refresh_token[-10:]}"
            stored_token = self.redis_client.get(redis_key)
            
            if not stored_token or stored_token.decode() != refresh_token:
                logger.warning("Token refresh failed - token not found in Redis", user_id=user_id)
                return None
            
            # Verify user exists and is active
            with SessionLocal() as db:
                user = db.query(User).filter(User.id == user_id).first()
                
                if not user or not user.is_active:
                    logger.warning("Token refresh failed - user not found or inactive", user_id=user_id)
                    return None
            
            # Create new tokens
            new_access_token = await self.create_access_token(user_id)
            new_refresh_token = await self.create_refresh_token(user_id)
            
            # Remove old refresh token from Redis
            self.redis_client.delete(redis_key)
            
            logger.info("Tokens refreshed successfully", user_id=user_id)
            
            return {
                "access_token": new_access_token,
                "refresh_token": new_refresh_token
            }
            
        except jwt.ExpiredSignatureError:
            logger.warning("Token refresh failed - expired refresh token")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning("Token refresh failed - invalid refresh token", error=str(e))
            return None
        except Exception as e:
            logger.error("Token refresh error", error=str(e))
            return None
    
    async def blacklist_token(self, token: str):
        """Add token to blacklist."""
        try:
            # Decode token to get expiration
            payload = jwt.decode(
                token, 
                settings.SECRET_KEY, 
                algorithms=[settings.ALGORITHM],
                options={"verify_exp": False}
            )
            
            exp = payload.get("exp")
            if exp:
                exp_datetime = datetime.fromtimestamp(exp)
                ttl = int((exp_datetime - datetime.utcnow()).total_seconds())
                
                if ttl > 0:
                    blacklist_key = f"{self.blacklist_key_prefix}{token[-10:]}"
                    self.redis_client.setex(blacklist_key, ttl, "blacklisted")
                    
                    logger.info("Token blacklisted successfully")
            
        except Exception as e:
            logger.error("Failed to blacklist token", error=str(e))
    
    async def is_token_blacklisted(self, token: str) -> bool:
        """Check if token is blacklisted."""
        try:
            blacklist_key = f"{self.blacklist_key_prefix}{token[-10:]}"
            return self.redis_client.exists(blacklist_key) > 0
        except Exception as e:
            logger.error("Failed to check token blacklist", error=str(e))
            return False
    
    async def update_password(self, user_id: int, new_password: str) -> bool:
        """Update user password."""
        try:
            with SessionLocal() as db:
                user = db.query(User).filter(User.id == user_id).first()
                
                if not user:
                    logger.warning("Password update failed - user not found", user_id=user_id)
                    return False
                
                user.hashed_password = self.get_password_hash(new_password)
                db.commit()
                
                logger.info("Password updated successfully", user_id=user_id)
                return True
                
        except Exception as e:
            logger.error("Failed to update password", user_id=user_id, error=str(e))
            return False


# Global auth service instance
auth_service = AuthService()


async def verify_api_key(api_key: str) -> Optional[User]:
    """Verify API key and return user."""
    try:
        import hashlib
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        logger.info("Verifying API key", key_preview=api_key[:10], key_hash=key_hash[:16])
        
        with SessionLocal() as db:
            from src.models.database import APIKey
            api_key_record = db.query(APIKey).filter(
                APIKey.key_hash == key_hash,
                APIKey.is_active == True
            ).first()
            
            if not api_key_record:
                logger.warning("API key not found", key_hash=key_hash[:16])
                return None
            
            logger.info("API key found", api_key_id=api_key_record.id, user_id=api_key_record.user_id)
            
            # Check expiration
            from datetime import datetime as dt_mod
            if api_key_record.expires_at and api_key_record.expires_at < dt_mod.utcnow():
                logger.warning("API key expired", expires_at=api_key_record.expires_at)
                return None
            
            # Get user
            user = db.query(User).filter(User.id == api_key_record.user_id).first()
            if not user or not user.is_active:
                logger.warning("User not found or inactive", user_id=api_key_record.user_id)
                return None
            
            # Update last used
            api_key_record.last_used = dt_mod.utcnow()
            db.commit()
            
            logger.info("API key verification successful", user_id=user.id, email=user.email)
            return user
            
    except Exception as e:
        logger.error("API key verification error", error=str(e))
        return None


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> User:
    """Dependency to get current authenticated user."""
    token = credentials.credentials
    logger.info("Authenticating user", token_preview=token[:10])
    
    # Try API key first (simpler and what we're using)
    logger.info("Attempting API key verification")
    user = await verify_api_key(token)
    
    if user:
        logger.info("API key authentication successful", user_id=user.id)
        return user
    
    # If API key fails, try JWT token
    logger.info("API key failed, attempting JWT verification")
    user = await auth_service.verify_token(token)
    
    if user:
        logger.info("JWT authentication successful", user_id=user.id)
        return user
    
    logger.warning("All authentication methods failed")
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired token",
        headers={"WWW-Authenticate": "Bearer"},
    )