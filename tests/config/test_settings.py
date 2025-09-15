"""Test-specific settings and configuration."""
import os
from pathlib import Path

# Test database configuration
TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL", 
    "sqlite:///./test_comfyui.db"
)

# Test Redis configuration  
TEST_REDIS_URL = os.environ.get(
    "TEST_REDIS_URL",
    "redis://localhost:6379/15"  # Use database 15 for tests
)

# Test Celery configuration
TEST_CELERY_BROKER_URL = os.environ.get(
    "TEST_CELERY_BROKER_URL",
    "redis://localhost:6379/14"  # Use database 14 for test broker
)

TEST_CELERY_RESULT_BACKEND = os.environ.get(
    "TEST_CELERY_RESULT_BACKEND", 
    "redis://localhost:6379/13"  # Use database 13 for test results
)

# Test storage configuration
TEST_S3_BUCKET = os.environ.get("TEST_S3_BUCKET", "test-comfyui-bucket")
TEST_S3_ACCESS_KEY = os.environ.get("TEST_S3_ACCESS_KEY", "test-access-key")
TEST_S3_SECRET_KEY = os.environ.get("TEST_S3_SECRET_KEY", "test-secret-key")
TEST_S3_ENDPOINT = os.environ.get("TEST_S3_ENDPOINT", "http://localhost:9000")  # MinIO

# Test file paths
TEST_FILES_DIR = Path(__file__).parent / "files"
TEST_MODELS_DIR = Path(__file__).parent / "models"
TEST_OUTPUTS_DIR = Path(__file__).parent / "outputs"

# Create test directories
for directory in [TEST_FILES_DIR, TEST_MODELS_DIR, TEST_OUTPUTS_DIR]:
    directory.mkdir(exist_ok=True)

# Test API configuration
TEST_API_HOST = os.environ.get("TEST_API_HOST", "localhost")
TEST_API_PORT = int(os.environ.get("TEST_API_PORT", 8000))
TEST_API_BASE_URL = f"http://{TEST_API_HOST}:{TEST_API_PORT}"

# Test ComfyUI configuration
TEST_COMFYUI_HOST = os.environ.get("TEST_COMFYUI_HOST", "localhost")
TEST_COMFYUI_PORT = int(os.environ.get("TEST_COMFYUI_PORT", 8188))
TEST_COMFYUI_BASE_URL = f"http://{TEST_COMFYUI_HOST}:{TEST_COMFYUI_PORT}"

# Test authentication
TEST_JWT_SECRET = os.environ.get("TEST_JWT_SECRET", "test-jwt-secret-key-for-testing-only")
TEST_JWT_ALGORITHM = "HS256"
TEST_JWT_EXPIRATION = 3600  # 1 hour

# Test performance settings
PERFORMANCE_TEST_ENABLED = os.environ.get("PERFORMANCE_TEST_ENABLED", "false").lower() == "true"
LOAD_TEST_USERS = int(os.environ.get("LOAD_TEST_USERS", 10))
LOAD_TEST_DURATION = int(os.environ.get("LOAD_TEST_DURATION", 60))  # seconds

# Test data settings
TEST_DATA_CLEANUP = os.environ.get("TEST_DATA_CLEANUP", "true").lower() == "true"
TEST_DATA_PERSIST = os.environ.get("TEST_DATA_PERSIST", "false").lower() == "true"

# External service test settings
SKIP_EXTERNAL_TESTS = os.environ.get("SKIP_EXTERNAL_TESTS", "false").lower() == "true"
MOCK_EXTERNAL_SERVICES = os.environ.get("MOCK_EXTERNAL_SERVICES", "true").lower() == "true"

# GPU test settings
GPU_TESTS_ENABLED = os.environ.get("GPU_TESTS_ENABLED", "false").lower() == "true"
MOCK_GPU_FUNCTIONS = os.environ.get("MOCK_GPU_FUNCTIONS", "true").lower() == "true"

# Logging settings for tests
TEST_LOG_LEVEL = os.environ.get("TEST_LOG_LEVEL", "INFO")
TEST_LOG_FILE = os.environ.get("TEST_LOG_FILE", "tests/logs/test.log")

# Coverage settings
COVERAGE_MIN_PERCENTAGE = int(os.environ.get("COVERAGE_MIN_PERCENTAGE", 80))
COVERAGE_FAIL_UNDER = os.environ.get("COVERAGE_FAIL_UNDER", "true").lower() == "true"

# Test timeouts
DEFAULT_TEST_TIMEOUT = int(os.environ.get("DEFAULT_TEST_TIMEOUT", 30))  # seconds
SLOW_TEST_TIMEOUT = int(os.environ.get("SLOW_TEST_TIMEOUT", 300))  # 5 minutes
INTEGRATION_TEST_TIMEOUT = int(os.environ.get("INTEGRATION_TEST_TIMEOUT", 120))  # 2 minutes

# Docker test settings (for CI/CD)
DOCKER_TEST_MODE = os.environ.get("DOCKER_TEST_MODE", "false").lower() == "true"
DOCKER_COMPOSE_FILE = os.environ.get("DOCKER_COMPOSE_FILE", "docker-compose.test.yml")

# CI/CD settings
CI_MODE = os.environ.get("CI", "false").lower() == "true"
GITHUB_ACTIONS = os.environ.get("GITHUB_ACTIONS", "false").lower() == "true"

# Test parallelization
PARALLEL_TEST_WORKERS = int(os.environ.get("PARALLEL_TEST_WORKERS", 1))

# Memory and resource limits for tests
MAX_MEMORY_USAGE_MB = int(os.environ.get("MAX_MEMORY_USAGE_MB", 1024))
MAX_CPU_USAGE_PERCENT = int(os.environ.get("MAX_CPU_USAGE_PERCENT", 80))

# Feature flags for testing
ENABLE_FEATURE_TESTS = {
    "webhook_retry": os.environ.get("TEST_WEBHOOK_RETRY", "true").lower() == "true",
    "model_caching": os.environ.get("TEST_MODEL_CACHING", "true").lower() == "true",
    "workflow_validation": os.environ.get("TEST_WORKFLOW_VALIDATION", "true").lower() == "true",
    "metrics_collection": os.environ.get("TEST_METRICS_COLLECTION", "true").lower() == "true",
    "auth_middleware": os.environ.get("TEST_AUTH_MIDDLEWARE", "true").lower() == "true"
}

# Test environment validation
def validate_test_environment():
    """Validate test environment configuration."""
    errors = []
    
    # Check required directories
    for directory in [TEST_FILES_DIR, TEST_MODELS_DIR, TEST_OUTPUTS_DIR]:
        if not directory.exists():
            errors.append(f"Test directory does not exist: {directory}")
    
    # Check database URL format
    if not TEST_DATABASE_URL.startswith(("sqlite://", "postgresql://", "mysql://")):
        errors.append("Invalid TEST_DATABASE_URL format")
    
    # Check Redis URL format
    if not TEST_REDIS_URL.startswith("redis://"):
        errors.append("Invalid TEST_REDIS_URL format")
    
    # Validate performance test settings
    if PERFORMANCE_TEST_ENABLED:
        if LOAD_TEST_USERS < 1:
            errors.append("LOAD_TEST_USERS must be at least 1")
        if LOAD_TEST_DURATION < 10:
            errors.append("LOAD_TEST_DURATION must be at least 10 seconds")
    
    # Validate timeout settings
    if DEFAULT_TEST_TIMEOUT < 1:
        errors.append("DEFAULT_TEST_TIMEOUT must be at least 1 second")
    
    # Validate coverage settings
    if COVERAGE_MIN_PERCENTAGE < 0 or COVERAGE_MIN_PERCENTAGE > 100:
        errors.append("COVERAGE_MIN_PERCENTAGE must be between 0 and 100")
    
    return errors


# Environment-specific settings
if CI_MODE:
    # Adjust settings for CI environment
    TEST_DATA_CLEANUP = True
    MOCK_EXTERNAL_SERVICES = True
    DEFAULT_TEST_TIMEOUT = 60  # Longer timeouts in CI
    
if DOCKER_TEST_MODE:
    # Adjust settings for Docker testing
    TEST_API_HOST = "testapi"  # Docker service name
    TEST_COMFYUI_HOST = "comfyui"  # Docker service name
    TEST_REDIS_URL = "redis://redis:6379/15"
    TEST_DATABASE_URL = "postgresql://test:test@postgres:5432/test_comfyui"


class TestConfig:
    """Test configuration class."""
    
    # Database
    DATABASE_URL = TEST_DATABASE_URL
    
    # Redis
    REDIS_URL = TEST_REDIS_URL
    
    # Celery
    CELERY_BROKER_URL = TEST_CELERY_BROKER_URL
    CELERY_RESULT_BACKEND = TEST_CELERY_RESULT_BACKEND
    CELERY_TASK_ALWAYS_EAGER = True  # Run tasks synchronously in tests
    
    # Storage
    S3_BUCKET = TEST_S3_BUCKET
    S3_ACCESS_KEY = TEST_S3_ACCESS_KEY
    S3_SECRET_KEY = TEST_S3_SECRET_KEY
    S3_ENDPOINT = TEST_S3_ENDPOINT
    
    # API
    API_HOST = TEST_API_HOST
    API_PORT = TEST_API_PORT
    API_BASE_URL = TEST_API_BASE_URL
    
    # ComfyUI
    COMFYUI_BASE_URL = TEST_COMFYUI_BASE_URL
    
    # Authentication
    JWT_SECRET = TEST_JWT_SECRET
    JWT_ALGORITHM = TEST_JWT_ALGORITHM
    JWT_EXPIRATION = TEST_JWT_EXPIRATION
    
    # Paths
    FILES_DIR = TEST_FILES_DIR
    MODELS_DIR = TEST_MODELS_DIR
    OUTPUTS_DIR = TEST_OUTPUTS_DIR
    
    # Features
    FEATURES = ENABLE_FEATURE_TESTS
    
    # Environment
    DEBUG = True
    TESTING = True
    
    @classmethod
    def validate(cls):
        """Validate configuration."""
        return validate_test_environment()


# Export test configuration
test_config = TestConfig()

# Validate on import
validation_errors = test_config.validate()
if validation_errors:
    print("Test environment validation errors:")
    for error in validation_errors:
        print(f"  - {error}")
    print("\nSome tests may fail due to configuration issues.")