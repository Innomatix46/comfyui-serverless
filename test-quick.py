#!/usr/bin/env python3
"""
Quick validation test for ComfyUI Serverless API
Run this to test core functionality without external dependencies
"""

import sys
import os
import importlib.util
from pathlib import Path

def test_imports():
    """Test that all modules can be imported"""
    print("🔍 Testing imports...")
    
    # Add src to path
    src_path = Path(__file__).parent / "src"
    sys.path.insert(0, str(src_path))
    
    try:
        # Test core imports
        from utils.validation import ValidationResult, WorkflowValidator
        from models.schemas import UserBase, WorkflowStatus
        print("✅ Core modules imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_validation_functions():
    """Test validation utilities"""
    print("🧪 Testing validation functions...")
    
    try:
        from utils.validation import ValidationResult, validate_file_upload, validate_webhook_url
        
        # Test file validation
        result = validate_file_upload("test.jpg", "image/jpeg", 1000)
        assert result.is_valid == True
        print("✅ File validation works")
        
        # Test webhook validation
        result = validate_webhook_url("https://example.com/webhook")
        assert result.is_valid == True
        
        result = validate_webhook_url("invalid-url")
        assert result.is_valid == False
        print("✅ Webhook validation works")
        
        return True
    except Exception as e:
        print(f"❌ Validation test error: {e}")
        return False

def test_schemas():
    """Test Pydantic schemas"""
    print("📋 Testing Pydantic schemas...")
    
    try:
        from models.schemas import UserBase, WorkflowStatus, Priority, ModelType
        
        # Test user schema  
        user_data = {
            "email": "test@example.com",
            "username": "testuser"
        }
        
        user_request = UserBase(**user_data)
        assert user_request.email == "test@example.com"
        print("✅ User schema validation works")
        
        # Test enum schemas
        assert WorkflowStatus.PENDING == "pending"
        assert Priority.HIGH == "high"
        assert ModelType.CHECKPOINT == "checkpoint"
        print("✅ Enum schemas work")
        
        return True
    except Exception as e:
        print(f"❌ Schema test error: {e}")
        return False

def main():
    """Run all quick tests"""
    print("🚀 ComfyUI Serverless API - Quick Validation Test")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_validation_functions, 
        test_schemas
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append(False)
        print()
    
    # Summary
    print("📊 Test Summary")
    print("=" * 50)
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"🎉 ALL TESTS PASSED! ({passed}/{total})")
        print("✅ Core functionality is working correctly")
        print("🚀 Ready to run full test suite!")
        return 0
    else:
        print(f"⚠️  Some tests failed: {passed}/{total} passed")
        print("❌ Please check the errors above")
        return 1

if __name__ == "__main__":
    exit(main())