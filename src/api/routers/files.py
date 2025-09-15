"""File management API endpoints."""
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query, Path
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from typing import List, Optional
import uuid
import io

from src.core.database import get_db
from src.models.schemas import FileUploadResponse, FileInfo
from src.models.database import User
from src.services.storage import storage_service
from src.services.auth import get_current_user

router = APIRouter()


@router.post("/upload", response_model=FileUploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user)
):
    """Upload a file."""
    try:
        # Validate file size
        content = await file.read()
        if len(content) > 100 * 1024 * 1024:  # 100MB limit
            raise HTTPException(
                status_code=413,
                detail="File too large. Maximum size is 100MB."
            )
        
        # Generate file ID
        file_id = str(uuid.uuid4())
        
        # Upload file
        storage_path = await storage_service.upload_file(
            file_id=file_id,
            filename=file.filename,
            content=content,
            content_type=file.content_type or "application/octet-stream",
            user_id=current_user.id
        )
        
        return FileUploadResponse(
            file_id=file_id,
            filename=file.filename,
            size=len(content),
            content_type=file.content_type or "application/octet-stream"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to upload file: {str(e)}"
        )


@router.get("/{file_id}/download")
async def download_file(
    file_id: str = Path(..., description="File ID"),
    current_user: User = Depends(get_current_user)
):
    """Download a file."""
    try:
        # Get file info
        file_info = await storage_service.get_file_info(file_id, current_user.id)
        if not file_info:
            raise HTTPException(
                status_code=404,
                detail="File not found"
            )
        
        # Download file content
        content = await storage_service.download_file(file_id, current_user.id)
        if not content:
            raise HTTPException(
                status_code=404,
                detail="File content not found"
            )
        
        # Return file as streaming response
        return StreamingResponse(
            io.BytesIO(content),
            media_type=file_info["content_type"],
            headers={
                "Content-Disposition": f"attachment; filename={file_info['filename']}"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to download file: {str(e)}"
        )


@router.get("/{file_id}", response_model=FileInfo)
async def get_file_info(
    file_id: str = Path(..., description="File ID"),
    current_user: User = Depends(get_current_user)
):
    """Get file information."""
    try:
        file_info = await storage_service.get_file_info(file_id, current_user.id)
        
        if not file_info:
            raise HTTPException(
                status_code=404,
                detail="File not found"
            )
        
        return FileInfo(**file_info)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get file info: {str(e)}"
        )


@router.delete("/{file_id}")
async def delete_file(
    file_id: str = Path(..., description="File ID"),
    current_user: User = Depends(get_current_user)
):
    """Delete a file."""
    try:
        success = await storage_service.delete_file(file_id, current_user.id)
        
        if success:
            return {"message": "File deleted successfully"}
        else:
            raise HTTPException(
                status_code=404,
                detail="File not found"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete file: {str(e)}"
        )


@router.get("/", response_model=List[FileInfo])
async def list_files(
    limit: int = Query(20, ge=1, le=100, description="Number of files to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    current_user: User = Depends(get_current_user)
):
    """List user's files."""
    try:
        files = await storage_service.list_user_files(
            user_id=current_user.id,
            limit=limit,
            offset=offset
        )
        
        return [FileInfo(**file_data) for file_data in files]
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list files: {str(e)}"
        )


@router.post("/upload/presigned")
async def get_presigned_upload_url(
    filename: str = Query(..., description="Original filename"),
    content_type: str = Query(..., description="File content type"),
    current_user: User = Depends(get_current_user)
):
    """Get presigned URL for direct file upload (S3 only)."""
    try:
        file_id = str(uuid.uuid4())
        
        upload_url = await storage_service.get_signed_upload_url(
            file_id=file_id,
            filename=filename,
            content_type=content_type,
            user_id=current_user.id,
            expires_in=3600  # 1 hour
        )
        
        if not upload_url:
            raise HTTPException(
                status_code=501,
                detail="Presigned URLs not supported with current storage backend"
            )
        
        return {
            "file_id": file_id,
            "upload_url": upload_url,
            "expires_in": 3600
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate upload URL: {str(e)}"
        )


@router.get("/{file_id}/download/presigned")
async def get_presigned_download_url(
    file_id: str = Path(..., description="File ID"),
    expires_in: int = Query(3600, ge=60, le=86400, description="URL expiry in seconds"),
    current_user: User = Depends(get_current_user)
):
    """Get presigned URL for direct file download (S3 only)."""
    try:
        download_url = await storage_service.get_signed_download_url(
            file_id=file_id,
            user_id=current_user.id,
            expires_in=expires_in
        )
        
        if not download_url:
            raise HTTPException(
                status_code=501,
                detail="Presigned URLs not supported with current storage backend"
            )
        
        return {
            "file_id": file_id,
            "download_url": download_url,
            "expires_in": expires_in
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate download URL: {str(e)}"
        )