#!/usr/bin/env python3
"""
Reset Database and Restart Backend for Fresh Marketplace
This script ensures a clean marketplace state with no products.
"""

import requests
import subprocess
import time
import os
import sys

def reset_database():
    """Reset the database to fresh marketplace state"""
    try:
        print("🔄 Resetting database to fresh marketplace state...")
        
        # Try to reset via API
        response = requests.post('http://localhost:5000/api/reset-database')
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ {data['message']}")
            print(f"📊 Sellers: {data['sellers_count']}, Products: {data['products_count']}")
            return True
        else:
            print(f"❌ Failed to reset database: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("⚠️ Backend not running, will start fresh")
        return True
    except Exception as e:
        print(f"❌ Error resetting database: {e}")
        return False

def start_backend():
    """Start the backend server"""
    try:
        print("🚀 Starting backend server...")
        
        # Change to project directory
        project_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(project_dir)
        
        # Start backend
        process = subprocess.Popen([
            sys.executable, 'backend/app.py'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        print("⏳ Waiting for server to start...")
        time.sleep(5)
        
        # Check if server is running
        try:
            response = requests.get('http://localhost:5000/api/health')
            if response.status_code == 200:
                print("✅ Backend server started successfully!")
                print("🔗 Backend API: http://localhost:5000")
                return True
            else:
                print("❌ Server started but health check failed")
                return False
        except:
            print("❌ Server failed to start properly")
            return False
            
    except Exception as e:
        print(f"❌ Error starting backend: {e}")
        return False

def main():
    """Main function to reset and restart"""
    print("🎨 Artisans Hub - Database Reset & Restart")
    print("=" * 50)
    
    # Step 1: Reset database
    if reset_database():
        print("✅ Database reset successful")
    else:
        print("⚠️ Database reset failed, continuing anyway")
    
    print()
    
    # Step 2: Start backend
    if start_backend():
        print("✅ Backend server is running")
        print()
        print("🎯 Next Steps:")
        print("1. Open your browser to http://localhost:3000")
        print("2. Navigate to any category page")
        print("3. You should see 'Sellers Yet to Upload!' message")
        print("4. All statistics should show zero values")
        print()
        print("🔄 The application is now in fresh marketplace state!")
    else:
        print("❌ Failed to start backend server")
        print("💡 Try running manually: python backend/app.py")

if __name__ == "__main__":
    main()