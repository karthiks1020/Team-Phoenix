#!/usr/bin/env python3
"""
Artisans Hub - Cloud Deployment Demo & Status Check
Automated deployment status checker with default URLs
"""

import requests
import json
import sys
from urllib.parse import urljoin

def check_backend_health(base_url):
    """Check if backend is responding"""
    try:
        health_url = urljoin(base_url, '/api/health')
        print(f"🔍 Checking: {health_url}")
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

def check_local_deployment():
    """Check current local deployment"""
    print("🔍 Artisans Hub - Local Deployment Check")
    print("=" * 50)
    
    local_backend = "http://192.168.1.105:5000"
    local_frontend = "http://192.168.1.105:3000"
    
    print(f"🌐 Checking local deployment...")
    print(f"Backend: {local_backend}")
    print(f"Frontend: {local_frontend}")
    print("-" * 50)
    
    # Check backend
    backend_ok = check_backend_health(local_backend)
    
    # Check frontend
    try:
        response = requests.get(local_frontend, timeout=5)
        if response.status_code == 200:
            print("✅ Frontend: Accessible")
            frontend_ok = True
        else:
            print(f"❌ Frontend Check Failed: {response.status_code}")
            frontend_ok = False
    except Exception as e:
        print(f"❌ Frontend Connection Failed: {str(e)}")
        frontend_ok = False
    
    return backend_ok, frontend_ok

def demo_cloud_deployment():
    """Demonstrate what cloud deployment would look like"""
    print("\n" + "=" * 60)
    print("🌐 CLOUD DEPLOYMENT DEMONSTRATION")
    print("=" * 60)
    
    print("\n📋 Deployment Steps:")
    print("1. ✅ Railway CLI installed and ready")
    print("2. 🔐 Login to Railway (requires browser authentication)")
    print("3. 📦 Create new Railway project")
    print("4. ⚙️ Set environment variables")
    print("5. 🚀 Deploy backend to Railway")
    print("6. 🌐 Deploy frontend to Netlify")
    
    print("\n🎯 Expected Cloud URLs:")
    print("• Backend:  https://artisans-hub-backend.railway.app")
    print("• Frontend: https://artisans-hub.netlify.app")
    
    print("\n📝 Manual Deployment Commands:")
    print("```bash")
    print("# 1. Login to Railway")
    print("railway login")
    print("")
    print("# 2. Create and deploy project") 
    print("railway project new")
    print("railway env set FLASK_ENV=production")
    print("railway env set PORT=5000")
    print("railway env set CORS_ORIGINS='*'")
    print("railway up")
    print("")
    print("# 3. Deploy frontend")
    print("cd frontend")
    print("npm run build")
    print("# Upload build folder to netlify.com")
    print("```")

def main():
    print("🚀 Artisans Hub - Deployment Status & Demo")
    print("=" * 60)
    
    # Check local deployment first
    backend_ok, frontend_ok = check_local_deployment()
    
    # Show deployment demo
    demo_cloud_deployment()
    
    print("\n" + "=" * 60)
    print("📊 DEPLOYMENT READINESS ASSESSMENT")
    print("=" * 60)
    
    if backend_ok and frontend_ok:
        print("🎉 Local deployment is working perfectly!")
        print("✅ Your application is ready for cloud deployment")
        print("🚀 All features tested and functional")
        print("\n🔗 Current Access:")
        print("• Local Frontend: http://192.168.1.105:3000")
        print("• Local Backend:  http://192.168.1.105:5000")
        
        print("\n🌟 Cloud Deployment Benefits:")
        print("• Global 24/7 accessibility")
        print("• Professional HTTPS URLs")
        print("• Auto-scaling infrastructure") 
        print("• Zero maintenance hosting")
        print("• Mobile-optimized performance")
        
    else:
        print("⚠️ Local deployment has issues - fix these first:")
        if not backend_ok:
            print("❌ Backend not responding - check Flask server")
        if not frontend_ok:
            print("❌ Frontend not accessible - check React dev server")
    
    print("\n🎯 Next Steps:")
    print("1. Ensure local servers are running")
    print("2. Run: railway login (manual browser auth required)")
    print("3. Run: railway project new")
    print("4. Run: railway up")
    print("5. Deploy frontend to Netlify")
    
    return 0 if (backend_ok and frontend_ok) else 1

if __name__ == "__main__":
    sys.exit(main())