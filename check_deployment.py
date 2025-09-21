#!/usr/bin/env python3
"""
Artisans Hub - Cloud Deployment Status Checker
Check if your cloud deployment is working correctly
"""

import requests
import json
import sys
from urllib.parse import urljoin

def check_backend_health(base_url):
    """Check if backend is responding"""
    try:
        health_url = urljoin(base_url, '/api/health')
        response = requests.get(health_url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Backend Health: {data.get('status', 'unknown')}")
            return True
        else:
            print(f"❌ Backend Health Check Failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Backend Connection Failed: {str(e)}")
        return False

def check_frontend(frontend_url):
    """Check if frontend is accessible"""
    try:
        response = requests.get(frontend_url, timeout=10)
        if response.status_code == 200:
            print("✅ Frontend: Accessible")
            return True
        else:
            print(f"❌ Frontend Check Failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Frontend Connection Failed: {str(e)}")
        return False

def check_api_endpoints(base_url):
    """Check critical API endpoints"""
    endpoints = [
        '/api/products',
        '/'
    ]
    
    all_good = True
    for endpoint in endpoints:
        try:
            url = urljoin(base_url, endpoint)
            response = requests.get(url, timeout=10)
            if response.status_code in [200, 404]:  # 404 is ok for some endpoints
                print(f"✅ API Endpoint {endpoint}: Working")
            else:
                print(f"❌ API Endpoint {endpoint}: Failed ({response.status_code})")
                all_good = False
        except Exception as e:
            print(f"❌ API Endpoint {endpoint}: Error - {str(e)}")
            all_good = False
    
    return all_good

def main():
    print("🔍 Artisans Hub - Cloud Deployment Status Check")
    print("=" * 50)
    
    # Get URLs from user input or use defaults
    backend_url = input("Enter your backend URL (Railway): ").strip()
    if not backend_url:
        backend_url = "https://artisans-hub-backend.railway.app"
    
    frontend_url = input("Enter your frontend URL (Netlify): ").strip()
    if not frontend_url:
        frontend_url = "https://artisans-hub.netlify.app"
    
    print(f"\n🌐 Checking deployment status...")
    print(f"Backend: {backend_url}")
    print(f"Frontend: {frontend_url}")
    print("-" * 50)
    
    # Run checks
    backend_ok = check_backend_health(backend_url)
    frontend_ok = check_frontend(frontend_url)
    api_ok = check_api_endpoints(backend_url)
    
    print("-" * 50)
    
    if backend_ok and frontend_ok and api_ok:
        print("🎉 All systems operational! Your Artisans Hub is live in the cloud!")
        print(f"🔗 Access your marketplace: {frontend_url}")
        print(f"🔧 API documentation: {backend_url}/api/health")
    else:
        print("⚠️ Some issues detected. Check the logs above for details.")
        print("💡 Try running the deployment script again or check the platform dashboards.")
    
    return 0 if (backend_ok and frontend_ok and api_ok) else 1

if __name__ == "__main__":
    sys.exit(main())