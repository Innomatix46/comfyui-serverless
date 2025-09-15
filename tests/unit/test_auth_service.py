"""Unit tests for authentication service."""

import pytest
import jwt
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
import fakeredis

try:
    from src.services.auth import AuthService
    from src.models.database import User
    from src.config.settings import settings
except ImportError:
    pytest.skip("Source modules not available", allow_module_level=True)


class TestAuthService:
    """Test cases for AuthService."""
    
    @pytest.fixture
    def auth_service(self, mock_redis):
        """Create AuthService instance with mocked Redis."""
        with patch('redis.from_url', return_value=mock_redis):
            return AuthService()
    
    @pytest.fixture
    def sample_user(self):
        """Sample user for testing."""
        return User(
            id=1,
            email="test@example.com",
            username="testuser",
            hashed_password="$2b$12$test.hash",
            is_active=True
        )
    
    def test_verify_password_success(self, auth_service):
        """Test successful password verification."""
        plain_password = "testpassword123"
        # Use a real bcrypt hash for testing
        hashed_password = auth_service.get_password_hash(plain_password)
        
        result = auth_service.verify_password(plain_password, hashed_password)
        assert result is True
    
    def test_verify_password_failure(self, auth_service):
        """Test failed password verification."""
        plain_password = "wrongpassword"
        hashed_password = "$2b$12$different.hash"
        
        result = auth_service.verify_password(plain_password, hashed_password)
        assert result is False
    
    def test_get_password_hash(self, auth_service):
        """Test password hashing."""
        password = "testpassword123"
        hashed = auth_service.get_password_hash(password)
        
        assert hashed != password
        assert hashed.startswith("$2b$")
        assert auth_service.verify_password(password, hashed)
    
    @pytest.mark.asyncio
    async def test_create_user_success(self, auth_service):
        """Test successful user creation."""
        with patch('src.core.database.SessionLocal') as mock_session:
            mock_db = Mock()
            mock_session.return_value.__enter__.return_value = mock_db
            mock_db.query.return_value.filter.return_value.first.return_value = None
            
            # Mock the user creation
            created_user = Mock(spec=User)
            created_user.id = 1
            created_user.email = "new@example.com"
            
            mock_db.add.return_value = None
            mock_db.commit.return_value = None
            mock_db.refresh.return_value = None
            
            with patch('src.models.database.User', return_value=created_user):
                user = await auth_service.create_user(
                    email="new@example.com",
                    password="password123",
                    username="newuser"
                )
            
            assert user is not None
            assert user.email == "new@example.com"
            mock_db.add.assert_called_once()
            mock_db.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_user_duplicate_email(self, auth_service):
        """Test user creation with duplicate email."""
        with patch('src.core.database.SessionLocal') as mock_session:
            mock_db = Mock()
            mock_session.return_value.__enter__.return_value = mock_db
            
            # Mock existing user
            existing_user = Mock(spec=User)
            mock_db.query.return_value.filter.return_value.first.return_value = existing_user
            
            user = await auth_service.create_user(
                email="existing@example.com",
                password="password123"
            )
            
            assert user is None
            mock_db.add.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_authenticate_user_success(self, auth_service, sample_user):
        """Test successful user authentication."""
        password = "testpassword123"
        sample_user.hashed_password = auth_service.get_password_hash(password)
        
        with patch('src.core.database.SessionLocal') as mock_session:
            mock_db = Mock()
            mock_session.return_value.__enter__.return_value = mock_db
            mock_db.query.return_value.filter.return_value.first.return_value = sample_user
            
            user = await auth_service.authenticate_user(
                email="test@example.com",
                password=password
            )
            
            assert user is not None
            assert user.email == "test@example.com"
    
    @pytest.mark.asyncio
    async def test_authenticate_user_wrong_password(self, auth_service, sample_user):
        """Test authentication with wrong password."""
        with patch('src.core.database.SessionLocal') as mock_session:
            mock_db = Mock()
            mock_session.return_value.__enter__.return_value = mock_db
            mock_db.query.return_value.filter.return_value.first.return_value = sample_user
            
            user = await auth_service.authenticate_user(
                email="test@example.com", 
                password="wrongpassword"
            )
            
            assert user is None
    
    @pytest.mark.asyncio
    async def test_authenticate_user_not_found(self, auth_service):
        """Test authentication with non-existent user."""
        with patch('src.core.database.SessionLocal') as mock_session:
            mock_db = Mock()
            mock_session.return_value.__enter__.return_value = mock_db
            mock_db.query.return_value.filter.return_value.first.return_value = None
            
            user = await auth_service.authenticate_user(
                email="nonexistent@example.com",
                password="password123"
            )
            
            assert user is None
    
    @pytest.mark.asyncio
    async def test_authenticate_user_inactive(self, auth_service, sample_user):
        """Test authentication with inactive user."""
        sample_user.is_active = False
        
        with patch('src.core.database.SessionLocal') as mock_session:
            mock_db = Mock()
            mock_session.return_value.__enter__.return_value = mock_db
            mock_db.query.return_value.filter.return_value.first.return_value = sample_user
            
            user = await auth_service.authenticate_user(
                email="test@example.com",
                password="testpassword123"
            )
            
            assert user is None
    
    @pytest.mark.asyncio 
    async def test_create_access_token(self, auth_service):
        """Test access token creation."""
        user_id = 1
        token = await auth_service.create_access_token(user_id)
        
        assert token is not None
        
        # Decode and verify token
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM]
        )
        
        assert payload["user_id"] == user_id
        assert payload["type"] == "access"
        assert "exp" in payload
        assert "iat" in payload
    
    @pytest.mark.asyncio
    async def test_create_refresh_token(self, auth_service):
        """Test refresh token creation."""
        user_id = 1
        token = await auth_service.create_refresh_token(user_id)
        
        assert token is not None
        
        # Decode and verify token
        payload = jwt.decode(
            token,
            settings.SECRET_KEY, 
            algorithms=[settings.ALGORITHM]
        )
        
        assert payload["user_id"] == user_id
        assert payload["type"] == "refresh"
        
        # Check token is stored in Redis
        redis_key = f"{auth_service.refresh_token_prefix}{user_id}:{token[-10:]}"
        stored_token = auth_service.redis_client.get(redis_key)
        assert stored_token == token
    
    @pytest.mark.asyncio
    async def test_verify_token_success(self, auth_service, sample_user):
        """Test successful token verification."""
        user_id = sample_user.id
        token = await auth_service.create_access_token(user_id)
        
        with patch('src.core.database.SessionLocal') as mock_session:
            mock_db = Mock()
            mock_session.return_value.__enter__.return_value = mock_db
            mock_db.query.return_value.filter.return_value.first.return_value = sample_user
            
            user = await auth_service.verify_token(token)
            
            assert user is not None
            assert user.id == user_id
    
    @pytest.mark.asyncio
    async def test_verify_token_expired(self, auth_service):
        """Test verification of expired token."""
        # Create expired token
        expire = datetime.utcnow() - timedelta(hours=1)
        payload = {
            "user_id": 1,
            "exp": expire,
            "iat": datetime.utcnow() - timedelta(hours=2),
            "type": "access"
        }
        
        token = jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
        
        user = await auth_service.verify_token(token)
        assert user is None
    
    @pytest.mark.asyncio
    async def test_verify_token_blacklisted(self, auth_service):
        """Test verification of blacklisted token."""
        token = await auth_service.create_access_token(1)
        
        # Blacklist the token
        await auth_service.blacklist_token(token)
        
        user = await auth_service.verify_token(token)
        assert user is None
    
    @pytest.mark.asyncio
    async def test_refresh_tokens_success(self, auth_service, sample_user):
        """Test successful token refresh."""
        user_id = sample_user.id
        refresh_token = await auth_service.create_refresh_token(user_id)
        
        with patch('src.core.database.SessionLocal') as mock_session:
            mock_db = Mock()
            mock_session.return_value.__enter__.return_value = mock_db
            mock_db.query.return_value.filter.return_value.first.return_value = sample_user
            
            tokens = await auth_service.refresh_tokens(refresh_token)
            
            assert tokens is not None
            assert "access_token" in tokens
            assert "refresh_token" in tokens
            assert tokens["access_token"] != refresh_token
            assert tokens["refresh_token"] != refresh_token
    
    @pytest.mark.asyncio
    async def test_refresh_tokens_invalid_token(self, auth_service):
        """Test refresh with invalid token."""
        invalid_token = "invalid.token.here"
        
        tokens = await auth_service.refresh_tokens(invalid_token)
        assert tokens is None
    
    @pytest.mark.asyncio
    async def test_blacklist_token(self, auth_service):
        """Test token blacklisting."""
        token = await auth_service.create_access_token(1)
        
        # Blacklist token
        await auth_service.blacklist_token(token)
        
        # Check if blacklisted
        is_blacklisted = await auth_service.is_token_blacklisted(token)
        assert is_blacklisted is True
    
    @pytest.mark.asyncio
    async def test_is_token_blacklisted_false(self, auth_service):
        """Test checking non-blacklisted token."""
        token = await auth_service.create_access_token(1)
        
        is_blacklisted = await auth_service.is_token_blacklisted(token)
        assert is_blacklisted is False
    
    @pytest.mark.asyncio
    async def test_update_password_success(self, auth_service, sample_user):
        """Test successful password update."""
        new_password = "newpassword123"
        
        with patch('src.core.database.SessionLocal') as mock_session:
            mock_db = Mock()
            mock_session.return_value.__enter__.return_value = mock_db
            mock_db.query.return_value.filter.return_value.first.return_value = sample_user
            
            result = await auth_service.update_password(sample_user.id, new_password)
            
            assert result is True
            mock_db.commit.assert_called_once()
            # Verify password was hashed
            assert sample_user.hashed_password != new_password
            assert sample_user.hashed_password.startswith("$2b$")
    
    @pytest.mark.asyncio
    async def test_update_password_user_not_found(self, auth_service):
        """Test password update for non-existent user."""
        with patch('src.core.database.SessionLocal') as mock_session:
            mock_db = Mock()
            mock_session.return_value.__enter__.return_value = mock_db
            mock_db.query.return_value.filter.return_value.first.return_value = None
            
            result = await auth_service.update_password(999, "newpassword")
            
            assert result is False
            mock_db.commit.assert_not_called()


class TestAuthServiceEdgeCases:
    """Edge case tests for AuthService."""
    
    @pytest.fixture
    def auth_service(self):
        """Create AuthService instance."""
        with patch('redis.from_url', return_value=fakeredis.FakeRedis()):
            return AuthService()
    
    @pytest.mark.asyncio
    async def test_create_user_database_error(self, auth_service):
        """Test user creation with database error."""
        with patch('src.core.database.SessionLocal') as mock_session:
            mock_session.side_effect = Exception("Database connection failed")
            
            user = await auth_service.create_user(
                email="test@example.com",
                password="password123"
            )
            
            assert user is None
    
    @pytest.mark.asyncio
    async def test_verify_token_invalid_format(self, auth_service):
        """Test token verification with invalid format."""
        invalid_token = "not.a.valid.jwt"
        
        user = await auth_service.verify_token(invalid_token)
        assert user is None
    
    @pytest.mark.asyncio
    async def test_verify_token_missing_claims(self, auth_service):
        """Test token verification with missing claims."""
        # Create token without required claims
        payload = {"exp": datetime.utcnow() + timedelta(hours=1)}
        token = jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
        
        user = await auth_service.verify_token(token)
        assert user is None
    
    @pytest.mark.asyncio
    async def test_blacklist_token_invalid_token(self, auth_service):
        """Test blacklisting invalid token."""
        invalid_token = "invalid.token"
        
        # Should not raise exception
        await auth_service.blacklist_token(invalid_token)
        
        # Check Redis doesn't contain invalid entries
        keys = auth_service.redis_client.keys(f"{auth_service.blacklist_key_prefix}*")
        assert len(keys) == 0
    
    def test_password_hash_consistency(self, auth_service):
        """Test that same password produces different hashes."""
        password = "samepassword123"
        
        hash1 = auth_service.get_password_hash(password)
        hash2 = auth_service.get_password_hash(password)
        
        # Hashes should be different (due to salt)
        assert hash1 != hash2
        
        # But both should verify correctly
        assert auth_service.verify_password(password, hash1)
        assert auth_service.verify_password(password, hash2)
    
    @pytest.mark.asyncio
    async def test_token_expiry_edge_case(self, auth_service):
        """Test token that expires exactly now."""
        # Create token that expires in 1 second
        expire = datetime.utcnow() + timedelta(seconds=1)
        payload = {
            "user_id": 1,
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "access"
        }
        
        token = jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
        
        # Wait for token to expire
        import time
        time.sleep(2)
        
        user = await auth_service.verify_token(token)
        assert user is None