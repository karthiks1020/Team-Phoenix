#!/bin/bash

# Artisans Hub - Cloud Deployment Script
echo "🚀 Deploying Artisans Hub to the Cloud"
echo "======================================"

# Check if Railway CLI is installed
if ! command -v railway &> /dev/null; then
    echo "❌ Railway CLI not found. Installing..."
    npm install -g @railway/cli
fi

# Login to Railway (if not already logged in)
echo "🔐 Logging into Railway..."
railway login

# Create new Railway project for backend
echo "📦 Creating Railway project..."
railway project new

# Set environment variables
echo "⚙️ Setting environment variables..."
railway env set FLASK_ENV=production
railway env set PORT=5000
railway env set CORS_ORIGINS="*"
railway env set USE_FALLBACK_AI=true
railway env set MAX_CONTENT_LENGTH=16777216

# Deploy the backend
echo "🚀 Deploying backend to Railway..."
railway up

echo "✅ Backend deployment initiated!"
echo "📝 Note: Frontend will be deployed separately to Netlify/Vercel"
echo "🌐 Your Railway dashboard: https://railway.app/dashboard"