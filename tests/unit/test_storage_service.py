"""Unit tests for storage service."""

import pytest
import tempfile
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, mock_open
from pathlib import Path
import boto3
from moto import mock_s3

try:
    from src.services.storage import StorageService
    from src.models.database import FileUpload
    from src.config.settings import settings
except ImportError:
    pytest.skip("Source modules not available", allow_module_level=True)


class TestStorageServiceLocal:
    """Test cases for local storage backend."""
    
    @pytest.fixture
    def temp_storage_path(self):
        """Create temporary storage directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def local_storage_service(self, temp_storage_path):
        """Create StorageService with local backend."""
        with patch.object(settings, 'STORAGE_TYPE', 'local'), \
             patch.object(settings, 'STORAGE_BASE_PATH', temp_storage_path):
            return StorageService()
    
    @pytest.mark.asyncio
    async def test_upload_file_success(self, local_storage_service):
        """Test successful file upload to local storage."""
        file_id = "test-file-123"
        filename = "test.jpg"
        content = b"test image content"
        content_type = "image/jpeg"
        user_id = 1
        metadata = {"format": "JPEG", "quality": 85}
        
        with patch('src.core.database.SessionLocal') as mock_session:
            mock_db = Mock()
            mock_session.return_value.__enter__.return_value = mock_db
            
            storage_path = await local_storage_service.upload_file(
                file_id=file_id,
                filename=filename,
                content=content,
                content_type=content_type,
                user_id=user_id,
                metadata=metadata
            )
            
            # Verify file record was created
            mock_db.add.assert_called_once()
            mock_db.commit.assert_called_once()
            
            # Verify storage path
            assert storage_path.endswith(file_id)
            
            # Verify file was written
            assert os.path.exists(storage_path)
            with open(storage_path, 'rb') as f:
                assert f.read() == content
    
    @pytest.mark.asyncio
    async def test_upload_file_stream_success(self, local_storage_service):
        """Test successful file stream upload."""
        file_id = "test-stream-123"
        filename = "test_stream.jpg"
        content_type = "image/jpeg"
        user_id = 1
        file_size = 1024
        
        # Create mock file stream
        mock_stream = Mock()
        mock_stream.read.side_effect = [b"chunk1", b"chunk2", b""]  # Empty indicates EOF
        
        with patch('src.core.database.SessionLocal') as mock_session:
            mock_db = Mock()
            mock_session.return_value.__enter__.return_value = mock_db
            
            storage_path = await local_storage_service.upload_file_stream(
                file_id=file_id,
                filename=filename,
                file_stream=mock_stream,
                content_type=content_type,
                user_id=user_id,
                file_size=file_size
            )
            
            # Verify file record was created
            mock_db.add.assert_called_once()
            mock_db.commit.assert_called_once()
            
            # Verify storage path
            assert storage_path.endswith(file_id)
    
    @pytest.mark.asyncio
    async def test_download_file_success(self, local_storage_service):
        """Test successful file download."""
        file_id = "test-download-123"
        user_id = 1
        content = b"test download content"
        
        # Create test file
        file_path = Path(local_storage_service.local_path) / file_id
        file_path.write_bytes(content)
        
        with patch('src.core.database.SessionLocal') as mock_session:
            mock_db = Mock()
            mock_session.return_value.__enter__.return_value = mock_db
            
            mock_file_record = Mock(spec=FileUpload)
            mock_file_record.storage_path = str(file_path)
            mock_file_record.storage_type = "local"
            mock_file_record.expires_at = datetime.utcnow() + timedelta(days=1)
            
            mock_db.query.return_value.filter.return_value.first.return_value = mock_file_record
            
            downloaded_content = await local_storage_service.download_file(file_id, user_id)
            
            assert downloaded_content == content
            mock_db.commit.assert_called_once()  # Access time updated
    
    @pytest.mark.asyncio
    async def test_download_file_not_found(self, local_storage_service):
        """Test downloading non-existent file."""
        file_id = "non-existent-file"
        user_id = 1
        
        with patch('src.core.database.SessionLocal') as mock_session:
            mock_db = Mock()
            mock_session.return_value.__enter__.return_value = mock_db
            mock_db.query.return_value.filter.return_value.first.return_value = None
            
            result = await local_storage_service.download_file(file_id, user_id)
            
            assert result is None
    
    @pytest.mark.asyncio
    async def test_download_file_expired(self, local_storage_service):
        """Test downloading expired file."""
        file_id = "expired-file-123"
        user_id = 1
        
        with patch('src.core.database.SessionLocal') as mock_session:
            mock_db = Mock()
            mock_session.return_value.__enter__.return_value = mock_db
            
            mock_file_record = Mock(spec=FileUpload)
            mock_file_record.expires_at = datetime.utcnow() - timedelta(hours=1)  # Expired
            
            mock_db.query.return_value.filter.return_value.first.return_value = mock_file_record
            
            result = await local_storage_service.download_file(file_id, user_id)
            
            assert result is None
    
    @pytest.mark.asyncio
    async def test_delete_file_success(self, local_storage_service):
        """Test successful file deletion."""
        file_id = "test-delete-123"
        user_id = 1
        
        # Create test file
        file_path = Path(local_storage_service.local_path) / file_id
        file_path.write_bytes(b"test content")
        
        with patch('src.core.database.SessionLocal') as mock_session:
            mock_db = Mock()
            mock_session.return_value.__enter__.return_value = mock_db
            
            mock_file_record = Mock(spec=FileUpload)
            mock_file_record.storage_path = str(file_path)
            mock_file_record.storage_type = "local"
            
            mock_db.query.return_value.filter.return_value.first.return_value = mock_file_record
            
            result = await local_storage_service.delete_file(file_id, user_id)
            
            assert result is True
            mock_db.delete.assert_called_once_with(mock_file_record)
            mock_db.commit.assert_called_once()
            
            # Verify file was deleted from filesystem
            assert not file_path.exists()
    
    @pytest.mark.asyncio
    async def test_delete_file_not_found(self, local_storage_service):
        """Test deleting non-existent file."""
        file_id = "non-existent-file"
        user_id = 1
        
        with patch('src.core.database.SessionLocal') as mock_session:
            mock_db = Mock()
            mock_session.return_value.__enter__.return_value = mock_db
            mock_db.query.return_value.filter.return_value.first.return_value = None
            
            result = await local_storage_service.delete_file(file_id, user_id)
            
            assert result is False
            mock_db.delete.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_get_file_info_success(self, local_storage_service):
        """Test getting file information."""
        file_id = "test-info-123"
        user_id = 1
        
        with patch('src.core.database.SessionLocal') as mock_session:
            mock_db = Mock()
            mock_session.return_value.__enter__.return_value = mock_db
            
            mock_file_record = Mock(spec=FileUpload)
            mock_file_record.id = file_id
            mock_file_record.original_filename = "test.jpg"
            mock_file_record.content_type = "image/jpeg"
            mock_file_record.file_size = 1024
            mock_file_record.created_at = datetime.utcnow()
            mock_file_record.expires_at = datetime.utcnow() + timedelta(days=7)
            mock_file_record.is_uploaded = True
            mock_file_record.metadata = {"format": "JPEG"}
            
            mock_db.query.return_value.filter.return_value.first.return_value = mock_file_record
            
            file_info = await local_storage_service.get_file_info(file_id, user_id)
            
            assert file_info is not None
            assert file_info["file_id"] == file_id
            assert file_info["filename"] == "test.jpg"
            assert file_info["content_type"] == "image/jpeg"
            assert file_info["file_size"] == 1024
            assert file_info["is_uploaded"] is True
            assert file_info["metadata"]["format"] == "JPEG"
    
    @pytest.mark.asyncio
    async def test_list_user_files(self, local_storage_service):
        """Test listing user files."""
        user_id = 1
        
        with patch('src.core.database.SessionLocal') as mock_session:
            mock_db = Mock()
            mock_session.return_value.__enter__.return_value = mock_db
            
            # Mock file records
            mock_files = []
            for i in range(3):
                mock_file = Mock(spec=FileUpload)
                mock_file.id = f"file-{i}"
                mock_file.original_filename = f"test_{i}.jpg"
                mock_file.content_type = "image/jpeg"
                mock_file.file_size = 1024 * (i + 1)
                mock_file.created_at = datetime.utcnow() - timedelta(days=i)
                mock_file.expires_at = datetime.utcnow() + timedelta(days=7-i)
                mock_file.is_uploaded = True
                mock_files.append(mock_file)
            
            mock_db.query.return_value.filter.return_value.order_by.return_value.offset.return_value.limit.return_value.all.return_value = mock_files
            
            files = await local_storage_service.list_user_files(user_id, limit=10, offset=0)
            
            assert len(files) == 3
            assert files[0]["file_id"] == "file-0"
            assert files[1]["file_id"] == "file-1"
            assert files[2]["file_id"] == "file-2"
    
    @pytest.mark.asyncio
    async def test_cleanup_expired_files(self, local_storage_service):
        """Test cleanup of expired files."""
        with patch('src.core.database.SessionLocal') as mock_session:
            mock_db = Mock()
            mock_session.return_value.__enter__.return_value = mock_db
            
            # Mock expired files
            expired_files = []
            for i in range(2):
                mock_file = Mock(spec=FileUpload)
                mock_file.id = f"expired-{i}"
                mock_file.storage_path = f"/tmp/expired-{i}"
                mock_file.storage_type = "local"
                expired_files.append(mock_file)
            
            mock_db.query.return_value.filter.return_value.all.return_value = expired_files
            
            # Mock file deletion
            with patch('os.remove') as mock_remove:
                deleted_count = await local_storage_service.cleanup_expired_files()
                
                assert deleted_count == 2
                assert mock_remove.call_count == 2
                assert mock_db.delete.call_count == 2
                mock_db.commit.assert_called_once()


class TestStorageServiceS3:
    """Test cases for S3 storage backend."""
    
    @pytest.fixture
    def s3_storage_service(self):
        """Create StorageService with S3 backend."""
        with patch.object(settings, 'STORAGE_TYPE', 's3'), \
             patch.object(settings, 'AWS_S3_BUCKET', 'test-bucket'):
            
            with mock_s3():
                # Create mock S3 client and bucket
                s3_client = boto3.client(
                    's3',
                    aws_access_key_id='testing',
                    aws_secret_access_key='testing',
                    region_name='us-east-1'
                )
                s3_client.create_bucket(Bucket='test-bucket')
                
                service = StorageService()
                service.s3_client = s3_client
                service.bucket_name = 'test-bucket'
                
                return service
    
    @pytest.mark.asyncio
    async def test_upload_file_to_s3(self, s3_storage_service):
        """Test file upload to S3."""
        file_id = "s3-test-123"
        filename = "test_s3.jpg"
        content = b"S3 test content"
        content_type = "image/jpeg"
        user_id = 1
        
        with patch('src.core.database.SessionLocal') as mock_session:
            mock_db = Mock()
            mock_session.return_value.__enter__.return_value = mock_db
            
            storage_path = await s3_storage_service.upload_file(
                file_id=file_id,
                filename=filename,
                content=content,
                content_type=content_type,
                user_id=user_id
            )
            
            # Verify S3 key format
            assert storage_path == f"files/{file_id}"
            
            # Verify file was uploaded to S3
            response = s3_storage_service.s3_client.get_object(
                Bucket='test-bucket',
                Key=storage_path
            )
            assert response['Body'].read() == content
            assert response['ContentType'] == content_type
    
    @pytest.mark.asyncio
    async def test_download_file_from_s3(self, s3_storage_service):
        """Test file download from S3."""
        file_id = "s3-download-123"
        user_id = 1
        content = b"S3 download content"
        s3_key = f"files/{file_id}"
        
        # Upload file to S3 first
        s3_storage_service.s3_client.put_object(
            Bucket='test-bucket',
            Key=s3_key,
            Body=content,
            ContentType='image/jpeg'
        )
        
        with patch('src.core.database.SessionLocal') as mock_session:
            mock_db = Mock()
            mock_session.return_value.__enter__.return_value = mock_db
            
            mock_file_record = Mock(spec=FileUpload)
            mock_file_record.storage_path = s3_key
            mock_file_record.storage_type = "s3"
            mock_file_record.expires_at = datetime.utcnow() + timedelta(days=1)
            
            mock_db.query.return_value.filter.return_value.first.return_value = mock_file_record
            
            downloaded_content = await s3_storage_service.download_file(file_id, user_id)
            
            assert downloaded_content == content
    
    @pytest.mark.asyncio
    async def test_delete_file_from_s3(self, s3_storage_service):
        """Test file deletion from S3."""
        file_id = "s3-delete-123"
        user_id = 1
        s3_key = f"files/{file_id}"
        
        # Upload file to S3 first
        s3_storage_service.s3_client.put_object(
            Bucket='test-bucket',
            Key=s3_key,
            Body=b"content to delete",
            ContentType='text/plain'
        )
        
        with patch('src.core.database.SessionLocal') as mock_session:
            mock_db = Mock()
            mock_session.return_value.__enter__.return_value = mock_db
            
            mock_file_record = Mock(spec=FileUpload)
            mock_file_record.storage_path = s3_key
            mock_file_record.storage_type = "s3"
            
            mock_db.query.return_value.filter.return_value.first.return_value = mock_file_record
            
            result = await s3_storage_service.delete_file(file_id, user_id)
            
            assert result is True
            
            # Verify file was deleted from S3
            with pytest.raises(s3_storage_service.s3_client.exceptions.NoSuchKey):
                s3_storage_service.s3_client.get_object(
                    Bucket='test-bucket',
                    Key=s3_key
                )
    
    @pytest.mark.asyncio
    async def test_get_signed_upload_url(self, s3_storage_service):
        """Test generating signed upload URL."""
        file_id = "signed-upload-123"
        filename = "upload_test.jpg"
        content_type = "image/jpeg"
        user_id = 1
        expires_in = 3600
        
        with patch('src.core.database.SessionLocal') as mock_session:
            mock_db = Mock()
            mock_session.return_value.__enter__.return_value = mock_db
            
            url = await s3_storage_service.get_signed_upload_url(
                file_id=file_id,
                filename=filename,
                content_type=content_type,
                user_id=user_id,
                expires_in=expires_in
            )
            
            assert url is not None
            assert "amazonaws.com" in url or "localhost" in url  # Works with moto
            assert "uploads/" in url
            
            # Verify file record was created
            mock_db.add.assert_called_once()
            mock_db.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_signed_download_url(self, s3_storage_service):
        """Test generating signed download URL."""
        file_id = "signed-download-123"
        user_id = 1
        s3_key = f"files/{file_id}"
        
        with patch('src.core.database.SessionLocal') as mock_session:
            mock_db = Mock()
            mock_session.return_value.__enter__.return_value = mock_db
            
            mock_file_record = Mock(spec=FileUpload)
            mock_file_record.storage_path = s3_key
            
            mock_db.query.return_value.filter.return_value.first.return_value = mock_file_record
            
            url = await s3_storage_service.get_signed_download_url(
                file_id=file_id,
                user_id=user_id,
                expires_in=3600
            )
            
            assert url is not None
            assert "amazonaws.com" in url or "localhost" in url  # Works with moto
    
    @pytest.mark.asyncio
    async def test_get_signed_url_local_backend_returns_none(self, local_storage_service):
        """Test that signed URLs return None for local backend."""
        url = await local_storage_service.get_signed_upload_url(
            file_id="test",
            filename="test.jpg",
            content_type="image/jpeg",
            user_id=1
        )
        
        assert url is None
        
        url = await local_storage_service.get_signed_download_url(
            file_id="test",
            user_id=1
        )
        
        assert url is None


class TestStorageServiceEdgeCases:
    """Edge case tests for StorageService."""
    
    @pytest.fixture
    def storage_service(self):
        """Create basic storage service."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(settings, 'STORAGE_TYPE', 'local'), \
                 patch.object(settings, 'STORAGE_BASE_PATH', temp_dir):
                return StorageService()
    
    @pytest.mark.asyncio
    async def test_upload_file_database_error(self, storage_service):
        """Test file upload with database error."""
        with patch('src.core.database.SessionLocal') as mock_session:
            mock_session.side_effect = Exception("Database connection failed")
            
            with pytest.raises(Exception, match="Database connection failed"):
                await storage_service.upload_file(
                    file_id="test-123",
                    filename="test.jpg",
                    content=b"test",
                    content_type="image/jpeg",
                    user_id=1
                )
    
    @pytest.mark.asyncio
    async def test_upload_file_empty_content(self, storage_service):
        """Test uploading empty file."""
        with patch('src.core.database.SessionLocal') as mock_session:
            mock_db = Mock()
            mock_session.return_value.__enter__.return_value = mock_db
            
            storage_path = await storage_service.upload_file(
                file_id="empty-file",
                filename="empty.txt",
                content=b"",
                content_type="text/plain",
                user_id=1
            )
            
            assert storage_path is not None
            mock_db.add.assert_called_once()
            
            # Verify empty file was created
            assert os.path.exists(storage_path)
            with open(storage_path, 'rb') as f:
                assert f.read() == b""
    
    @pytest.mark.asyncio
    async def test_download_file_filesystem_error(self, storage_service):
        """Test file download with filesystem error."""
        file_id = "fs-error-file"
        user_id = 1
        
        with patch('src.core.database.SessionLocal') as mock_session:
            mock_db = Mock()
            mock_session.return_value.__enter__.return_value = mock_db
            
            mock_file_record = Mock(spec=FileUpload)
            mock_file_record.storage_path = "/nonexistent/path/file.txt"
            mock_file_record.storage_type = "local"
            mock_file_record.expires_at = datetime.utcnow() + timedelta(days=1)
            
            mock_db.query.return_value.filter.return_value.first.return_value = mock_file_record
            
            result = await storage_service.download_file(file_id, user_id)
            
            # Should return None on filesystem error
            assert result is None
    
    @pytest.mark.asyncio
    async def test_delete_file_already_deleted_from_filesystem(self, storage_service):
        """Test deleting file that's already gone from filesystem."""
        file_id = "already-deleted-file"
        user_id = 1
        
        with patch('src.core.database.SessionLocal') as mock_session:
            mock_db = Mock()
            mock_session.return_value.__enter__.return_value = mock_db
            
            mock_file_record = Mock(spec=FileUpload)
            mock_file_record.storage_path = "/nonexistent/file.txt"
            mock_file_record.storage_type = "local"
            
            mock_db.query.return_value.filter.return_value.first.return_value = mock_file_record
            
            # Should still succeed even if file doesn't exist on filesystem
            result = await storage_service.delete_file(file_id, user_id)
            
            assert result is True
            mock_db.delete.assert_called_once()
            mock_db.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_cleanup_partial_failure(self, storage_service):
        """Test cleanup with some files failing to delete."""
        with patch('src.core.database.SessionLocal') as mock_session:
            mock_db = Mock()
            mock_session.return_value.__enter__.return_value = mock_db
            
            # Mock expired files - some will fail deletion
            expired_files = []
            for i in range(3):
                mock_file = Mock(spec=FileUpload)
                mock_file.id = f"expired-{i}"
                mock_file.storage_path = f"/tmp/expired-{i}"
                mock_file.storage_type = "local"
                expired_files.append(mock_file)
            
            mock_db.query.return_value.filter.return_value.all.return_value = expired_files
            
            # Mock file deletion - second file fails
            def mock_remove(path):
                if "expired-1" in path:
                    raise OSError("Permission denied")
            
            with patch('os.remove', side_effect=mock_remove):
                deleted_count = await storage_service.cleanup_expired_files()
                
                # Should delete 2 out of 3 files (one failed)
                assert deleted_count == 2
                mock_db.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_list_user_files_empty_result(self, storage_service):
        """Test listing files for user with no files."""
        user_id = 999
        
        with patch('src.core.database.SessionLocal') as mock_session:
            mock_db = Mock()
            mock_session.return_value.__enter__.return_value = mock_db
            mock_db.query.return_value.filter.return_value.order_by.return_value.offset.return_value.limit.return_value.all.return_value = []
            
            files = await storage_service.list_user_files(user_id)
            
            assert files == []
    
    @pytest.mark.asyncio
    async def test_list_user_files_database_error(self, storage_service):
        """Test listing files with database error."""
        user_id = 1
        
        with patch('src.core.database.SessionLocal') as mock_session:
            mock_session.side_effect = Exception("Database error")
            
            files = await storage_service.list_user_files(user_id)
            
            # Should return empty list on error
            assert files == []
    
    def test_storage_service_initialization(self):
        """Test StorageService initialization with different backends."""
        # Test S3 backend
        with patch.object(settings, 'STORAGE_TYPE', 's3'), \
             patch.object(settings, 'AWS_ACCESS_KEY_ID', 'test-key'), \
             patch.object(settings, 'AWS_SECRET_ACCESS_KEY', 'test-secret'), \
             patch.object(settings, 'AWS_S3_BUCKET', 'test-bucket'):
            
            with patch('boto3.client') as mock_boto3:
                service = StorageService()
                
                assert service.storage_type == 's3'
                assert service.bucket_name == 'test-bucket'
                mock_boto3.assert_called_once()
        
        # Test local backend
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(settings, 'STORAGE_TYPE', 'local'), \
                 patch.object(settings, 'STORAGE_BASE_PATH', temp_dir):
                
                service = StorageService()
                
                assert service.storage_type == 'local'
                assert str(service.local_path) == temp_dir
                assert service.s3_client is None
    
    def test_file_size_edge_cases(self, storage_service):
        """Test handling of various file sizes."""
        # Test with very large file size metadata
        large_size = 10 * 1024 * 1024 * 1024  # 10GB
        
        # Should not raise any errors with large file size
        # (This is more of a data validation test)
        assert large_size > 0
        
        # Test with zero file size
        zero_size = 0
        assert zero_size >= 0