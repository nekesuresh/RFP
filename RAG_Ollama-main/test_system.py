#!/usr/bin/env python3
"""
Test script for the Multi-Agent RFP Assistant
"""

import requests
import time
import json
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

API_BASE_URL = "http://localhost:8000"

def test_api_health():
    """Test if the API is running"""
    print("Testing API health...")
    try:
        response = requests.get(f"{API_BASE_URL}/ping", timeout=5)
        if response.status_code == 200:
            print("✅ API is running")
            return True
        else:
            print(f"❌ API returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ API health check failed: {e}")
        return False

def test_config():
    """Test configuration endpoint"""
    print("\nTesting configuration...")
    try:
        response = requests.get(f"{API_BASE_URL}/config", timeout=5)
        if response.status_code == 200:
            config = response.json()
            print("✅ Configuration retrieved successfully")
            print(f"   Model: {config.get('ollama_model', 'N/A')}")
            print(f"   Temperature: {config.get('temperature', 'N/A')}")
            print(f"   Top K Results: {config.get('top_k_results', 'N/A')}")
            return True
        else:
            print(f"❌ Config test failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False

def test_legacy_ask():
    """Test the legacy ask endpoint"""
    print("\nTesting legacy ask endpoint...")
    try:
        query = "What is this document about?"
        response = requests.get(f"{API_BASE_URL}/ask/", params={"q": query}, timeout=30)
        if response.status_code == 200:
            result = response.json()
            print("✅ Legacy ask endpoint working")
            print(f"   Response length: {len(result.get('response', ''))} characters")
            return True
        else:
            print(f"❌ Legacy ask failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Legacy ask test failed: {e}")
        return False

def test_multi_agent_ask():
    """Test the multi-agent ask endpoint"""
    print("\nTesting multi-agent ask endpoint...")
    try:
        payload = {
            "query": "Help me improve the project scope section"
        }
        response = requests.post(f"{API_BASE_URL}/ask/", json=payload, timeout=60)
        if response.status_code == 200:
            result = response.json()
            print("✅ Multi-agent ask endpoint working")
            print(f"   Status: {result.get('status', 'N/A')}")
            print(f"   Documents retrieved: {result.get('retrieval_result', {}).get('num_documents', 0)}")
            print(f"   Agent log steps: {len(result.get('agent_log', []))}")
            return True
        else:
            print(f"❌ Multi-agent ask failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Multi-agent ask test failed: {e}")
        return False

def test_feedback():
    """Test the feedback endpoint"""
    print("\nTesting feedback endpoint...")
    try:
        payload = {
            "query": "Help me improve the project scope section",
            "feedback": "Make it more specific and measurable",
            "original_suggestion": "This is a test suggestion that needs improvement."
        }
        response = requests.post(f"{API_BASE_URL}/feedback/", json=payload, timeout=60)
        if response.status_code == 200:
            result = response.json()
            print("✅ Feedback endpoint working")
            print(f"   Status: {result.get('status', 'N/A')}")
            return True
        else:
            print(f"❌ Feedback test failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Feedback test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Multi-Agent RFP Assistant System Test")
    print("=" * 50)
    
    tests = [
        test_api_health,
        test_config,
        test_legacy_ask,
        test_multi_agent_ask,
        test_feedback
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The system is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the system configuration.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 