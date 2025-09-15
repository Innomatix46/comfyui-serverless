"""Storage service for file management."""
import asyncio
import os
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, BinaryIO
import aiofiles
try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError:
    boto3 = None
    ClientError = Exception
import structlog
from sqlalchemy.orm import Session

from src.config.settings import settings
from src.core.database import SessionLocal
from src.models.database import FileUpload

logger = structlog.get_logger()


class StorageService:
    """File storage service supporting multiple backends."""
    
    def __init__(self):
        self.storage_type = settings.STORAGE_TYPE
        self.local_path = Path(settings.STORAGE_BASE_PATH)
        self.local_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize S3 client if needed
        if self.storage_type == "s3":
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                region_name=settings.AWS_S3_REGION
            )
            self.bucket_name = settings.AWS_S3_BUCKET
        else:
            self.s3_client = None
            self.bucket_name = None
    
    async def upload_file(
        self,
        file_id: str,
        filename: str,
        content: bytes,
        content_type: str,
        user_id: int,
        metadata: Optional[Dict] = None
    ) -> str:
        """Upload file to storage backend."""
        try:
            if self.storage_type == "s3":
                storage_path = await self._upload_to_s3(file_id, content, content_type, metadata)
            else:
                storage_path = await self._upload_to_local(file_id, content)
            
            # Save file record to database
            with SessionLocal() as db:
                file_record = FileUpload(
                    id=file_id,
                    user_id=user_id,
                    filename=file_id,
                    original_filename=filename,
                    content_type=content_type,
                    file_size=len(content),
                    storage_path=storage_path,
                    storage_type=self.storage_type,
                    is_uploaded=True,
                    file_metadata=metadata or {},
                    expires_at=datetime.utcnow() + timedelta(days=settings.RESULT_RETENTION_DAYS)
                )
                db.add(file_record)
                db.commit()
            
            logger.info(
                "File uploaded successfully",
                file_id=file_id,
                storage_path=storage_path,
                size_bytes=len(content)
            )
            
            return storage_path
            
        except Exception as e:
            logger.error("File upload failed", file_id=file_id, error=str(e))
            raise
    
    async def upload_file_stream(
        self,
        file_id: str,
        filename: str,
        file_stream: BinaryIO,
        content_type: str,
        user_id: int,
        file_size: int,
        metadata: Optional[Dict] = None
    ) -> str:
        """Upload file from stream."""
        try:
            if self.storage_type == "s3":
                storage_path = await self._upload_stream_to_s3(
                    file_id, file_stream, content_type, file_size, metadata
                )
            else:
                storage_path = await self._upload_stream_to_local(file_id, file_stream)
            
            # Save file record to database
            with SessionLocal() as db:
                file_record = FileUpload(
                    id=file_id,
                    user_id=user_id,
                    filename=file_id,
                    original_filename=filename,
                    content_type=content_type,
                    file_size=file_size,
                    storage_path=storage_path,
                    storage_type=self.storage_type,
                    is_uploaded=True,
                    file_metadata=metadata or {},
                    expires_at=datetime.utcnow() + timedelta(days=settings.RESULT_RETENTION_DAYS)
                )
                db.add(file_record)
                db.commit()
            
            logger.info(
                "File stream uploaded successfully",
                file_id=file_id,
                storage_path=storage_path,
                size_bytes=file_size
            )
            
            return storage_path
            
        except Exception as e:
            logger.error("File stream upload failed", file_id=file_id, error=str(e))
            raise
    
    async def download_file(self, file_id: str, user_id: int) -> Optional[bytes]:
        """Download file content."""
        try:
            # Get file record
            with SessionLocal() as db:
                file_record = db.query(FileUpload).filter(
                    FileUpload.id == file_id,
                    FileUpload.user_id == user_id
                ).first()
                
                if not file_record:
                    logger.warning("File not found", file_id=file_id, user_id=user_id)
                    return None
                
                if file_record.expires_at and file_record.expires_at < datetime.utcnow():
                    logger.warning("File expired", file_id=file_id)
                    return None
                
                # Update access time
                file_record.accessed_at = datetime.utcnow()
                db.commit()
                
                storage_path = file_record.storage_path
                storage_type = file_record.storage_type
            
            # Download from storage backend
            if storage_type == "s3":
                content = await self._download_from_s3(storage_path)
            else:
                content = await self._download_from_local(storage_path)
            
            logger.info("File downloaded successfully", file_id=file_id)
            return content
            
        except Exception as e:
            logger.error("File download failed", file_id=file_id, error=str(e))
            return None
    
    async def get_file_info(self, file_id: str, user_id: int) -> Optional[Dict]:
        """Get file information."""
        try:
            with SessionLocal() as db:
                file_record = db.query(FileUpload).filter(
                    FileUpload.id == file_id,
                    FileUpload.user_id == user_id
                ).first()
                
                if not file_record:
                    return None
                
                return {
                    "file_id": file_record.id,
                    "filename": file_record.original_filename,
                    "content_type": file_record.content_type,
                    "file_size": file_record.file_size,
                    "created_at": file_record.created_at,
                    "expires_at": file_record.expires_at,
                    "is_uploaded": file_record.is_uploaded,
                    "metadata": file_record.file_metadata
                }
                
        except Exception as e:
            logger.error("Failed to get file info", file_id=file_id, error=str(e))
            return None
    
    async def delete_file(self, file_id: str, user_id: int) -> bool:
        """Delete file from storage."""
        try:
            # Get file record
            with SessionLocal() as db:
                file_record = db.query(FileUpload).filter(
                    FileUpload.id == file_id,
                    FileUpload.user_id == user_id
                ).first()
                
                if not file_record:
                    logger.warning("File not found for deletion", file_id=file_id)
                    return False
                
                storage_path = file_record.storage_path
                storage_type = file_record.storage_type
                
                # Delete from database first
                db.delete(file_record)
                db.commit()
            
            # Delete from storage backend
            if storage_type == "s3":
                await self._delete_from_s3(storage_path)
            else:
                await self._delete_from_local(storage_path)
            
            logger.info("File deleted successfully", file_id=file_id)
            return True
            
        except Exception as e:
            logger.error("File deletion failed", file_id=file_id, error=str(e))
            return False
    
    async def list_user_files(
        self,
        user_id: int,
        limit: int = 20,
        offset: int = 0
    ) -> List[Dict]:
        """List user's files."""
        try:
            with SessionLocal() as db:
                files = db.query(FileUpload).filter(
                    FileUpload.user_id == user_id
                ).order_by(
                    FileUpload.created_at.desc()
                ).offset(offset).limit(limit).all()
                
                return [
                    {
                        "file_id": f.id,
                        "filename": f.original_filename,
                        "content_type": f.content_type,
                        "file_size": f.file_size,
                        "created_at": f.created_at,
                        "expires_at": f.expires_at,
                        "is_uploaded": f.is_uploaded
                    }
                    for f in files
                ]
                
        except Exception as e:
            logger.error("Failed to list user files", user_id=user_id, error=str(e))
            return []
    
    async def cleanup_expired_files(self) -> int:
        """Clean up expired files."""
        try:
            deleted_count = 0
            
            with SessionLocal() as db:
                # Find expired files
                expired_files = db.query(FileUpload).filter(
                    FileUpload.expires_at < datetime.utcnow()
                ).all()
                
                for file_record in expired_files:
                    try:
                        # Delete from storage
                        if file_record.storage_type == "s3":
                            await self._delete_from_s3(file_record.storage_path)
                        else:
                            await self._delete_from_local(file_record.storage_path)
                        
                        # Delete from database
                        db.delete(file_record)
                        deleted_count += 1
                        
                    except Exception as e:
                        logger.error(
                            "Failed to delete expired file",
                            file_id=file_record.id,
                            error=str(e)
                        )
                
                db.commit()
            
            logger.info("Expired files cleanup completed", deleted_count=deleted_count)
            return deleted_count
            
        except Exception as e:
            logger.error("Cleanup failed", error=str(e))
            return 0
    
    async def get_signed_upload_url(
        self,
        file_id: str,
        filename: str,
        content_type: str,
        user_id: int,
        expires_in: int = 3600
    ) -> Optional[str]:
        """Generate signed URL for direct upload (S3 only)."""
        if self.storage_type != "s3":
            return None
        
        try:
            key = f"uploads/{user_id}/{file_id}"
            
            # Generate presigned URL
            url = self.s3_client.generate_presigned_url(
                'put_object',
                Params={
                    'Bucket': self.bucket_name,
                    'Key': key,
                    'ContentType': content_type
                },
                ExpiresIn=expires_in
            )
            
            # Create file record with placeholder
            with SessionLocal() as db:
                file_record = FileUpload(
                    id=file_id,
                    user_id=user_id,
                    filename=file_id,
                    original_filename=filename,
                    content_type=content_type,
                    file_size=0,
                    storage_path=key,
                    storage_type="s3",
                    is_uploaded=False,
                    expires_at=datetime.utcnow() + timedelta(seconds=expires_in)
                )
                db.add(file_record)
                db.commit()
            
            logger.info("Generated signed upload URL", file_id=file_id, expires_in=expires_in)
            return url
            
        except Exception as e:
            logger.error("Failed to generate signed URL", file_id=file_id, error=str(e))
            return None
    
    async def get_signed_download_url(
        self,
        file_id: str,
        user_id: int,
        expires_in: int = 3600
    ) -> Optional[str]:
        """Generate signed URL for direct download (S3 only)."""
        if self.storage_type != "s3":
            return None
        
        try:
            with SessionLocal() as db:
                file_record = db.query(FileUpload).filter(
                    FileUpload.id == file_id,
                    FileUpload.user_id == user_id
                ).first()
                
                if not file_record:
                    return None
                
                # Generate presigned URL
                url = self.s3_client.generate_presigned_url(
                    'get_object',
                    Params={
                        'Bucket': self.bucket_name,
                        'Key': file_record.storage_path
                    },
                    ExpiresIn=expires_in
                )
                
                return url
                
        except Exception as e:
            logger.error("Failed to generate signed download URL", file_id=file_id, error=str(e))
            return None
    
    # Private methods for storage backends
    
    async def _upload_to_s3(
        self,
        file_id: str,
        content: bytes,
        content_type: str,
        metadata: Optional[Dict] = None
    ) -> str:
        """Upload content to S3."""
        key = f"files/{file_id}"
        
        extra_args = {
            'ContentType': content_type,
            'Metadata': metadata or {}
        }
        
        self.s3_client.put_object(
            Bucket=self.bucket_name,
            Key=key,
            Body=content,
            **extra_args
        )
        
        return key
    
    async def _upload_stream_to_s3(
        self,
        file_id: str,
        file_stream: BinaryIO,
        content_type: str,
        file_size: int,
        metadata: Optional[Dict] = None
    ) -> str:
        """Upload file stream to S3."""
        key = f"files/{file_id}"
        
        extra_args = {
            'ContentType': content_type,
            'ContentLength': file_size,
            'Metadata': metadata or {}
        }
        
        self.s3_client.upload_fileobj(
            file_stream,
            self.bucket_name,
            key,
            ExtraArgs=extra_args
        )
        
        return key
    
    async def _download_from_s3(self, key: str) -> bytes:
        """Download content from S3."""
        response = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
        return response['Body'].read()
    
    async def _delete_from_s3(self, key: str):
        """Delete object from S3."""
        self.s3_client.delete_object(Bucket=self.bucket_name, Key=key)
    
    async def _upload_to_local(self, file_id: str, content: bytes) -> str:
        """Upload content to local storage."""
        file_path = self.local_path / file_id
        
        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(content)
        
        return str(file_path)
    
    async def _upload_stream_to_local(self, file_id: str, file_stream: BinaryIO) -> str:
        """Upload file stream to local storage."""
        file_path = self.local_path / file_id
        
        async with aiofiles.open(file_path, 'wb') as f:
            while True:
                chunk = file_stream.read(8192)
                if not chunk:
                    break
                await f.write(chunk)
        
        return str(file_path)
    
    async def _download_from_local(self, file_path: str) -> bytes:
        """Download content from local storage."""
        async with aiofiles.open(file_path, 'rb') as f:
            return await f.read()
    
    async def _delete_from_local(self, file_path: str):
        """Delete file from local storage."""
        try:
            os.remove(file_path)
        except FileNotFoundError:
            pass  # File already deleted


# Global storage service instance
storage_service = StorageService()
