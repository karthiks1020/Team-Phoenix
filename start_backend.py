#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.dirname(__file__))

from backend.app import app

if __name__ == '__main__':
    print("🚀 Starting AI-Powered Marketplace Backend...")
    print("📊 Server running at: http://localhost:5000")
    print("🎯 API endpoints available")
    app.run(debug=True, host='0.0.0.0', port=5000)
