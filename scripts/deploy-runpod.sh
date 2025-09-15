#!/usr/bin/env bash

set -euo pipefail

echo "🚀 Build and (optionally) push RunPod Serverless image"

# Configurable via env or flags
IMAGE_NAME="comfyui-serverless"
IMAGE_TAG="runpod"
REGISTRY=""
PUSH="0"

usage() {
  cat <<USAGE
Usage: $0 [-i image_name] [-t tag] [-r registry] [-p]
  -i  Image name (default: ${IMAGE_NAME})
  -t  Image tag (default: ${IMAGE_TAG})
  -r  Registry (e.g. ghcr.io/ORG or docker.io/USER)
  -p  Push after build (requires docker login)
USAGE
}

while getopts ":i:t:r:ph" opt; do
  case $opt in
    i) IMAGE_NAME="$OPTARG" ;;
    t) IMAGE_TAG="$OPTARG" ;;
    r) REGISTRY="$OPTARG" ;;
    p) PUSH="1" ;;
    h) usage; exit 0 ;;
    *) usage; exit 1 ;;
  esac
done

if ! command -v docker >/dev/null 2>&1; then
  echo "❌ Docker not found. Please install Docker." >&2
  exit 1
fi

FULL_IMAGE="$IMAGE_NAME:$IMAGE_TAG"
if [[ -n "${REGISTRY}" ]]; then
  FULL_IMAGE="${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"
fi

echo "🧱 Building image: ${FULL_IMAGE} (Dockerfile.runpod)"
docker build -t "${FULL_IMAGE}" -f Dockerfile.runpod .

if [[ "${PUSH}" == "1" ]]; then
  echo "📤 Pushing image: ${FULL_IMAGE}"
  docker push "${FULL_IMAGE}"
fi

echo "✅ Image ready: ${FULL_IMAGE}"
echo
echo "Next steps (RunPod Serverless):"
echo "1) Create a new Serverless Endpoint in RunPod."
echo "2) Use image: ${FULL_IMAGE}"
echo "3) Entrypoint/CMD: python runpod_handler.py"
echo "4) Set env vars (SECRET_KEY, DATABASE_URL, REDIS_URL, CELERY_*, COMFYUI_API_URL, STORAGE_*)"
echo
echo "Quick test payload (health):"
cat <<'JSON'
{
  "input": {
    "endpoint": "/health",
    "method": "GET"
  }
}
JSON

