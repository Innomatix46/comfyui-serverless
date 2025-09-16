#!/usr/bin/env python3
"""Generate secure secrets for RunPod deployment"""
import secrets
import string
import hashlib
from datetime import datetime

def generate_secret_key(length: int = 64) -> str:
    """Generate a secure secret key"""
    alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
    return ''.join(secrets.choice(alphabet) for _ in range(length))

def generate_api_key(length: int = 32) -> str:
    """Generate a secure API key"""
    return secrets.token_urlsafe(length)

def generate_password(length: int = 16) -> str:
    """Generate a secure password"""
    alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
    password = ''.join(secrets.choice(alphabet) for _ in range(length))
    return password

def main():
    print("🔐 Generating Secure Secrets for RunPod Deployment")
    print("=" * 60)
    
    # Generate secrets
    secret_key = generate_secret_key(64)
    comfyui_api_key = generate_api_key(32)
    admin_password = generate_password(16)
    postgres_password = generate_password(20)
    jwt_secret = generate_secret_key(64)
    
    print("📋 Environment Variables for RunPod:")
    print("-" * 40)
    print(f"SECRET_KEY={secret_key}")
    print(f"COMFYUI_API_KEY={comfyui_api_key}")
    print(f"ADMIN_PASSWORD={admin_password}")
    print(f"POSTGRES_PASSWORD={postgres_password}")
    print(f"JWT_SECRET={jwt_secret}")
    print()
    
    print("🔗 For local .env file:")
    print("-" * 25)
    env_content = f"""# Generated secrets - {datetime.now().isoformat()}
SECRET_KEY={secret_key}
COMFYUI_API_KEY={comfyui_api_key}
ADMIN_PASSWORD={admin_password}
POSTGRES_PASSWORD={postgres_password}
JWT_SECRET={jwt_secret}

# Database
DATABASE_URL=sqlite:///./comfyui_serverless.db

# ComfyUI Paths (RunPod)
COMFYUI_PATH=/workspace/comfyui
COMFYUI_MODELS_PATH=/workspace/comfyui/models
COMFYUI_OUTPUT_PATH=/workspace/comfyui/output
COMFYUI_TEMP_PATH=/workspace/comfyui/temp
COMFYUI_API_URL=http://localhost:8188

# API Configuration
DEBUG=false
HOST=0.0.0.0
PORT=8000
API_TITLE=ComfyUI Serverless API
API_VERSION=1.0.0

# Python Configuration
PYTHONPATH=/workspace/src
PYTHONDONTWRITEBYTECODE=1
PYTHONUNBUFFERED=1
"""
    
    # Save to file
    with open('.env.production', 'w') as f:
        f.write(env_content)
    
    print("💾 Secrets saved to .env.production")
    print()
    
    print("🚨 IMPORTANT SECURITY NOTES:")
    print("1. Never commit .env.production to git")
    print("2. Use different secrets for each environment")
    print("3. Rotate secrets regularly")
    print("4. Store secrets securely (password manager)")
    print("5. Monitor for leaked secrets in logs")
    print()
    
    print("📋 Next Steps:")
    print("1. Copy environment variables to RunPod console")
    print("2. Test deployment with test_client.py")
    print("3. Monitor logs for any issues")
    print("4. Set up monitoring and alerts")
    
    print("=" * 60)
    print("✅ Secrets generated successfully!")

if __name__ == "__main__":
    main()