#!/usr/bin/env bash

set -euo pipefail

echo "🧪 Run RunPod image locally for a quick smoke test"

IMAGE="${1:-comfyui-serverless:runpod}"
PORT="${PORT:-8000}"

if ! command -v docker >/dev/null 2>&1; then
  echo "❌ Docker not found. Please install Docker." >&2
  exit 1
fi

echo "➡️  Starting container: ${IMAGE} (port ${PORT})"
docker run --rm -p ${PORT}:8000 \
  -e SECRET_KEY="local-dev-secret" \
  -e DATABASE_URL="sqlite:////workspace/data.db" \
  -e REDIS_URL="redis://localhost:6379/0" \
  -e CELERY_BROKER_URL="redis://localhost:6379/1" \
  -e CELERY_RESULT_BACKEND="redis://localhost:6379/1" \
  -e COMFYUI_API_URL="http://localhost:8188" \
  "${IMAGE}" &

PID=$!
trap 'echo "🛑 Stopping..."; kill ${PID} >/dev/null 2>&1 || true' EXIT

echo "⏳ Waiting for API to be ready..."
for i in {1..30}; do
  if curl -fsS "http://localhost:${PORT}/health" >/dev/null; then
    echo "✅ API is up: http://localhost:${PORT}"
    break
  fi
  sleep 1
done

echo "🔎 Health check:"
curl -fsS "http://localhost:${PORT}/health" | jq '.' 2>/dev/null || curl -fsS "http://localhost:${PORT}/health" || true

echo
echo "You can now hit docs at: http://localhost:${PORT}/docs (if DEBUG=true)"
echo "Press Ctrl+C to stop."
wait ${PID}

