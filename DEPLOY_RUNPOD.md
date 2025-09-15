# RunPod Serverless Deployment

This guide shows how to package and deploy the API as a RunPod Serverless endpoint using the provided Dockerfile and handler.

## Prerequisites
- Docker installed and logged in to your registry (e.g., GHCR or Docker Hub).
- A RunPod account with Serverless enabled.
- External services available or planned: Postgres, Redis, and a reachable ComfyUI instance.

## Build and Push Image
- Build locally:
  - `bash scripts/deploy-runpod.sh -i comfyui-serverless -t runpod`
- Push to your registry (example):
  - `bash scripts/deploy-runpod.sh -i comfyui-serverless -t runpod -r ghcr.io/<org> -p`
- Dockerfile used: `Dockerfile.runpod` (includes `runpod_handler.py`).

## Local Smoke Test
- Run the built image locally and check `/health`:
  - `bash scripts/test-runpod-local.sh comfyui-serverless:runpod`
- Docs appear at `/docs` only if `DEBUG=true`.

## RunPod Endpoint Setup
- Image: `<registry>/comfyui-serverless:runpod`
- Command: `python runpod_handler.py`
- Recommended env vars:
  - `SECRET_KEY` (required, strong secret)
  - `DATABASE_URL` (Postgres; dev-only: `sqlite:////workspace/data.db`)
  - `REDIS_URL`, `CELERY_BROKER_URL`, `CELERY_RESULT_BACKEND`
  - `COMFYUI_API_URL` (URL of your ComfyUI server)
  - `STORAGE_TYPE` (`s3` or `local`), `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_S3_BUCKET`
- GPU: Usually not required for the API if ComfyUI runs separately on GPU.

## Example Serverless Request
```
{
  "input": {
    "endpoint": "/health",
    "method": "GET"
  }
}
```
Workflow execution example (replace payload as needed):
```
{
  "input": {
    "endpoint": "/workflows/execute",
    "method": "POST",
    "api_key": "<your_api_key>",
    "payload": {
      "workflow": { "nodes": { /* ... */ } },
      "priority": "normal",
      "webhook_url": "https://your-webhook/callback",
      "timeout_minutes": 30
    }
  }
}
```

## Notes
- Celery workers should run outside the Serverless endpoint (e.g., a separate service using the same Redis broker).
- Use managed Postgres/Redis in production.
- Ensure CORS and secrets are configured for production; the app refuses to start with a default `SECRET_KEY` when `DEBUG=false`.
