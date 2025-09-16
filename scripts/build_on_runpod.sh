#!/bin/bash
# Build Docker Image directly on RunPod Pod

echo "🚀 Building Docker Image on RunPod..."

# Docker Hub credentials (set these as environment variables)
DOCKER_USERNAME=${DOCKER_USERNAME:-"innomatix46"}
DOCKER_PASSWORD=${DOCKER_PASSWORD:-"your_docker_password"}
IMAGE_NAME="innomatix46/comfyui-serverless"
IMAGE_TAG="latest"

# Login to Docker Hub
echo "🔐 Logging into Docker Hub..."
echo $DOCKER_PASSWORD | docker login -u $DOCKER_USERNAME --password-stdin

# Clone repository
echo "📦 Cloning repository..."
cd /workspace
rm -rf comfyui-serverless
git clone https://github.com/Innomatix46/comfyui-serverless.git
cd comfyui-serverless

# Build Docker image
echo "🔨 Building Docker image..."
docker build -f Dockerfile.runpod -t $IMAGE_NAME:$IMAGE_TAG .

# Push to Docker Hub
echo "📤 Pushing to Docker Hub..."
docker push $IMAGE_NAME:$IMAGE_TAG

echo "✅ Docker image built and pushed successfully!"
echo "📋 Image: $IMAGE_NAME:$IMAGE_TAG"
echo ""
echo "🎯 Next steps:"
echo "1. Create new RunPod endpoint"
echo "2. Use Docker image: $IMAGE_NAME:$IMAGE_TAG"
echo "3. No build needed - image is ready!"