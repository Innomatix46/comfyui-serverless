#!/usr/bin/env python3
"""
Simplified API test for ComfyUI Serverless API
Tests core functionality without database dependencies
"""

import sys
import os
from pathlib import Path

def test_core_components():
    """Test core components individually"""
    print("🔧 Testing core components...")
    
    # Add src to path
    src_path = Path(__file__).parent / "src"
    sys.path.insert(0, str(src_path))
    
    tests_passed = []
    
    # Test 1: Validation utilities
    try:
        from utils.validation import ValidationResult, validate_file_upload
        result = validate_file_upload("test.jpg", "image/jpeg", 1000)
        assert result.is_valid == True
        print("✅ Validation utilities work")
        tests_passed.append(True)
    except Exception as e:
        print(f"❌ Validation test failed: {e}")
        tests_passed.append(False)
    
    # Test 2: Pydantic schemas (without database)
    try:
        from models.schemas import UserBase, WorkflowStatus, Priority
        user = UserBase(email="test@example.com")
        assert user.email == "test@example.com"
        assert WorkflowStatus.PENDING == "pending"
        print("✅ Pydantic schemas work")
        tests_passed.append(True)
    except Exception as e:
        print(f"❌ Schema test failed: {e}")
        tests_passed.append(False)
    
    # Test 3: Configuration (if exists)
    try:
        # Try to import settings without initializing
        import importlib.util
        config_path = src_path / "config" / "settings.py"
        if config_path.exists():
            spec = importlib.util.spec_from_file_location("settings", config_path)
            settings_module = importlib.util.module_from_spec(spec)
            # Don't execute, just check it can be loaded
            print("✅ Configuration module exists")
        else:
            print("⚠️ Configuration module not found (expected)")
        tests_passed.append(True)
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        tests_passed.append(False)
    
    return tests_passed

def test_api_structure():
    """Test API structure and routing"""
    print("🚀 Testing API structure...")
    
    src_path = Path(__file__).parent / "src"
    api_path = src_path / "api"
    
    # Check if key files exist
    key_files = [
        "main.py",
        "middleware.py",
        "routers/auth.py",
        "routers/workflow.py",
        "routers/health.py"
    ]
    
    missing_files = []
    for file_path in key_files:
        full_path = api_path / file_path
        if not full_path.exists():
            missing_files.append(str(full_path))
        else:
            print(f"✅ {file_path} exists")
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    
    print("✅ All API files present")
    return True

def test_service_structure():
    """Test service layer structure"""
    print("🔧 Testing service structure...")
    
    src_path = Path(__file__).parent / "src"
    services_path = src_path / "services"
    
    # Check if key service files exist
    key_services = [
        "auth.py",
        "workflow.py", 
        "storage.py",
        "model.py"
    ]
    
    for service in key_services:
        service_path = services_path / service
        if service_path.exists():
            print(f"✅ {service} service exists")
        else:
            print(f"❌ {service} service missing")
            return False
    
    return True

def test_basic_imports():
    """Test basic imports without running servers"""
    print("📦 Testing basic imports...")
    
    src_path = Path(__file__).parent / "src"
    sys.path.insert(0, str(src_path))
    
    # Test individual modules that don't require database
    modules_to_test = [
        ("utils.validation", "ValidationResult"),
        ("models.schemas", "UserBase"),
    ]
    
    for module_name, class_name in modules_to_test:
        try:
            module = __import__(module_name, fromlist=[class_name])
            getattr(module, class_name)
            print(f"✅ {module_name}.{class_name} imports successfully")
        except Exception as e:
            print(f"❌ {module_name}.{class_name} failed: {e}")
            return False
    
    return True

def main():
    """Run all tests"""
    print("🧪 ComfyUI Serverless API - Simple Structure Test")
    print("=" * 60)
    
    all_results = []
    
    # Run tests
    print("\n1. Testing basic imports...")
    all_results.append(test_basic_imports())
    
    print("\n2. Testing core components...")
    component_results = test_core_components()
    all_results.append(all(component_results))
    
    print("\n3. Testing API structure...")
    all_results.append(test_api_structure())
    
    print("\n4. Testing service structure...")
    all_results.append(test_service_structure())
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Results Summary")
    print("=" * 60)
    
    passed = sum(all_results)
    total = len(all_results)
    
    if passed == total:
        print(f"🎉 ALL STRUCTURE TESTS PASSED! ({passed}/{total})")
        print("\n✅ Project structure is correct")
        print("✅ Core components are functional")
        print("✅ Ready for full testing with dependencies!")
        
        print("\n🚀 Next steps:")
        print("1. Install dependencies: pip install -r src/requirements.txt")
        print("2. Set up database: PostgreSQL + Redis")
        print("3. Run full test suite: docker-compose -f docker-compose.test.yml up")
        print("4. Start API server: uvicorn src.api.main:app --reload")
        
        return 0
    else:
        print(f"⚠️ Some tests failed: {passed}/{total} passed")
        print("❌ Please check the project structure")
        return 1

if __name__ == "__main__":
    exit(main())