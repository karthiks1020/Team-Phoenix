# 🎨 AI-Powered Marketplace Assistant for Local Artisans

A revolutionary platform that empowers local artisans with AI-driven tools to showcase, sell, and promote their unique handicrafts.

## 🌟 Key Features

### 🤖 AI-Powered Core Features
- **Smart Handicraft Classification**: CNN model recognizing pottery, wooden dolls, basket weaving, handlooms
- **Intelligent Product Recommendations**: Personalized suggestions based on user preferences
- **AI Artisan Storytelling**: Automated generation of compelling product narratives
- **Dynamic Pricing Assistant**: AI-suggested pricing based on market trends

### 🎯 Innovative Marketplace Features  
- **Virtual Try-On with AR**: Experience products before purchase
- **Cultural Heritage Preservation**: Digital archive of traditional techniques
- **Accessibility Tools**: Voice navigation and screen reader support
- **Smart Search & Filtering**: Multi-modal search (text, image, voice)
- **Artisan Mentorship Network**: Connect experienced and new artisans

### 📱 User Experience Excellence
- **Real-time Chat Support**: Instant communication with artisans
- **Multi-language Support**: Breaking language barriers
- **Sustainable Impact Tracking**: Environmental and social impact metrics
- **Mobile-First Design**: Optimized for all devices

## 🏗️ Project Structure

```
artisan-marketplace/
├── ai_models/                  # ML models and training
│   ├── cnn_classifier/        # Handicraft classification
│   ├── recommendation/        # Product recommendation system
│   └── data_augmentation/     # Dataset enhancement
├── backend/                   # Flask API server
│   ├── app.py                # Main application
│   ├── models/               # Database models
│   ├── routes/               # API endpoints
│   └── utils/                # Helper functions
├── frontend/                  # React web application
│   ├── src/
│   │   ├── components/       # Reusable components
│   │   ├── pages/           # Main pages
│   │   └── utils/           # Frontend utilities
├── data/                     # Training datasets
└── tests/                    # Test suites
```

## 🚀 Quick Start

1. **Setup Environment**:
   ```bash
   pip install -r requirements.txt
   cd frontend && npm install
   ```

2. **Train CNN Model**:
   ```bash
   python ai_models/train_classifier.py
   ```

3. **Start Backend**:
   ```bash
   python backend/app.py
   ```

4. **Start Frontend**:
   ```bash
   cd frontend && npm start
   ```

## 🔬 CNN Training Improvements for Small Datasets

### Transfer Learning Strategy
- Using pre-trained EfficientNet-B0 as base model
- Fine-tuning on handicraft dataset
- Freezing early layers, training final layers

### Data Augmentation Techniques
- Rotation, scaling, flipping, color jittering
- Advanced techniques: Mixup, CutMix, AutoAugment
- Synthetic data generation using StyleGAN

### Model Architecture Optimizations
- Dropout layers for regularization
- Batch normalization for stability
- Learning rate scheduling
- Early stopping with validation monitoring

## 🎯 Hackathon Success Factors

- **Innovation**: Unique AI-powered features not found in existing marketplaces
- **Social Impact**: Empowering local artisans and preserving cultural heritage
- **Technical Excellence**: Advanced ML techniques with limited data
- **User Experience**: Intuitive, accessible, and engaging interface
- **Scalability**: Architecture designed for growth

## 📊 Expected Performance Improvements

With our enhanced training approach:
- **Baseline accuracy**: ~60% (small dataset, basic CNN)
- **With transfer learning**: ~85-90%
- **With data augmentation**: ~90-95%
- **With synthetic data**: ~95%+

## 🏆 Competitive Advantages

1. **AI-First Approach**: Every feature enhanced with machine learning
2. **Cultural Preservation**: Digital heritage documentation
3. **Accessibility**: Inclusive design for all users
4. **Sustainability**: Environmental impact tracking
5. **Community Building**: Artisan mentorship network