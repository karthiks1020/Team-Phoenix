# 🚀 Artisans Hub - Cloud Deployment Guide

## Overview
This guide will help you deploy the Artisans Hub marketplace to the cloud using Railway (backend) and Netlify/Vercel (frontend).

## 🎯 Deployment Architecture
- **Backend**: Flask API deployed on Railway
- **Frontend**: React app deployed on Netlify/Vercel
- **Database**: SQLite (upgradeable to PostgreSQL on Railway)
- **File Storage**: Local storage (upgradeable to cloud storage)

## 📋 Prerequisites
1. **Node.js** (v14+) installed
2. **Python** (v3.11+) installed
3. **Git** installed
4. **Railway CLI** (will be installed automatically)

## 🚀 Quick Deployment

### Option 1: Automated Deployment (Recommended)

#### Windows:
```bash
# Run the deployment script
.\deploy.bat
```

#### Linux/Mac:
```bash
# Make script executable and run
chmod +x deploy.sh
./deploy.sh
```

### Option 2: Manual Deployment

#### Step 1: Deploy Backend to Railway

1. **Install Railway CLI**:
   ```bash
   npm install -g @railway/cli
   ```

2. **Login to Railway**:
   ```bash
   railway login
   ```

3. **Create new project**:
   ```bash
   railway project new
   ```

4. **Set environment variables**:
   ```bash
   railway env set FLASK_ENV=production
   railway env set PORT=5000
   railway env set CORS_ORIGINS="*"
   railway env set USE_FALLBACK_AI=true
   ```

5. **Deploy**:
   ```bash
   railway up
   ```

#### Step 2: Deploy Frontend to Netlify

1. **Build the frontend**:
   ```bash
   cd frontend
   npm run build
   ```

2. **Deploy to Netlify**:
   - Go to [netlify.com](https://netlify.com)
   - Drag and drop the `build` folder
   - Or connect your GitHub repository

#### Step 3: Connect Frontend to Backend

1. Get your Railway backend URL from the Railway dashboard
2. Update `frontend/.env.production`:
   ```
   REACT_APP_API_BASE_URL=https://your-backend-url.railway.app
   ```
3. Rebuild and redeploy frontend

## 🌐 Access Your Cloud Application

After deployment, you'll have:
- **Backend API**: `https://your-app-name.railway.app`
- **Frontend**: `https://your-app-name.netlify.app`

## 🔧 Configuration Files Created

- `Dockerfile` - Container configuration
- `requirements.txt` - Python dependencies
- `railway.json` - Railway deployment config
- `.env.production` - Production environment variables
- `deploy.sh` / `deploy.bat` - Deployment scripts

## 📊 Monitoring & Logs

- **Railway Dashboard**: View logs, metrics, and manage environment variables
- **Netlify Dashboard**: Monitor frontend deployments and performance

## 🔄 Updating Your Deployment

### Backend Updates:
```bash
railway up
```

### Frontend Updates:
```bash
cd frontend
npm run build
# Redeploy build folder to Netlify
```

## 🛠️ Advanced Configuration

### Database Upgrade
To use PostgreSQL instead of SQLite:
1. Add PostgreSQL plugin in Railway dashboard
2. Update `DATABASE_URL` environment variable

### File Storage Upgrade
For production file storage:
1. Use AWS S3, Cloudinary, or similar service
2. Update image handling in `backend/app.py`

## 🚨 Troubleshooting

### Common Issues:
1. **Build fails**: Check Python/Node versions
2. **CORS errors**: Verify CORS_ORIGINS environment variable
3. **Database errors**: Check DATABASE_URL configuration
4. **File upload fails**: Verify MAX_CONTENT_LENGTH setting

### Getting Help:
- Railway Discord: [discord.gg/railway](https://discord.gg/railway)
- Netlify Support: [docs.netlify.com](https://docs.netlify.com)

## 🎉 Success!

Your Artisans Hub marketplace is now running in the cloud! 

**Next Steps:**
1. Set up a custom domain
2. Configure SSL certificates (automatic on both platforms)
3. Set up monitoring and analytics
4. Consider upgrading to PostgreSQL for production

---

**🌟 Features Available in Cloud Deployment:**
- ✅ AI-powered image classification
- ✅ Real-time product listings
- ✅ Secure file uploads
- ✅ Mobile-responsive design
- ✅ Privacy settings
- ✅ Editable AI content
- ✅ Indian localization (INR pricing)
- ✅ Auto-scaling and high availability

Happy selling! 🎨✨