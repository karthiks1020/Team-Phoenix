import React, { useState, useRef } from 'react';
import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import Webcam from 'react-webcam';
import axios from 'axios';
import { formatINRPrice } from '../utils/indianLocalization';
import Logo from '../components/Logo';

const SellPage = () => {
  const [step, setStep] = useState(1);
  const [selectedImage, setSelectedImage] = useState(null);
  const [previewUrl, setPreviewUrl] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [aiResult, setAiResult] = useState(null);
  const [editableDescription, setEditableDescription] = useState('');
  const [editablePrice, setEditablePrice] = useState('');
  const [isEditingDescription, setIsEditingDescription] = useState(false);
  const [isEditingPrice, setIsEditingPrice] = useState(false);
  const [showCamera, setShowCamera] = useState(false);
  const [sellerDetails, setSellerDetails] = useState({
    name: '',
    mobile: '',
    location: '',
    price: ''
  });
  const [isSubmitting, setIsSubmitting] = useState(false);
  
  const webcamRef = useRef(null);
  const fileInputRef = useRef(null);
  const navigate = useNavigate();

  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedImage(file);
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
      setStep(2);
    }
  };

  const captureImage = () => {
    const imageSrc = webcamRef.current.getScreenshot();
    if (imageSrc) {
      setPreviewUrl(imageSrc);
      setSelectedImage(imageSrc);
      setShowCamera(false);
      setStep(2);
    }
  };

  const generateAIDescription = async () => {
    setIsProcessing(true);
    try {
      let imageData;
      
      if (typeof selectedImage === 'string') {
        imageData = selectedImage;
      } else {
        imageData = await fileToBase64(selectedImage);
      }

      const response = await axios.post('/api/upload-analyze', {
        image: imageData
      });

      if (response.data.success) {
        setAiResult(response.data);
        setEditableDescription(response.data.ai_description);
        setEditablePrice(response.data.pricing_suggestion.suggested_price);
        setSellerDetails(prev => ({
          ...prev,
          price: response.data.pricing_suggestion.suggested_price
        }));
        setStep(3);
      } else {
        alert(response.data.message || 'Failed to analyze image.');
      }
    } catch (error) {
      alert('An error occurred while analyzing the image.');
    } finally {
      setIsProcessing(false);
    }
  };

  const fileToBase64 = (file) => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.readAsDataURL(file);
      reader.onload = () => resolve(reader.result);
      reader.onerror = error => reject(error);
    });
  };

  const handleDescriptionEdit = () => {
    setIsEditingDescription(!isEditingDescription);
  };

  const handlePriceEdit = () => {
    if (isEditingPrice) {
      setSellerDetails(prev => ({ ...prev, price: editablePrice }));
    }
    setIsEditingPrice(!isEditingPrice);
  };

  const handleSubmitListing = async () => {
    if (!sellerDetails.name || !sellerDetails.mobile || !sellerDetails.location || !sellerDetails.price) {
      alert('Please fill in all required fields.');
      return;
    }

    setIsSubmitting(true);
    try {
      const response = await axios.post('/api/create-listing', {
        seller_name: sellerDetails.name,
        seller_mobile: sellerDetails.mobile,
        seller_location: sellerDetails.location,
        category: aiResult.analysis.predicted_category,
        description: editableDescription,
        price: parseFloat(sellerDetails.price),
        image_filename: aiResult.image_filename,
        ai_generated: true
      });

      if (response.data.success) {
        alert('Your listing has been created successfully!');
        navigate('/');
      } else {
        alert(response.data.message || 'Failed to create listing.');
      }
    } catch (error) {
      alert('An error occurred while creating the listing.');
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-green-50 to-teal-50">
      <header className="bg-white/80 backdrop-blur-md shadow-lg">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <button
              onClick={() => navigate(-1)}
              className="flex items-center space-x-2 text-gray-600 hover:text-gray-800 transition-colors"
            >
              <span className="text-xl">←</span>
              <span>Back</span>
            </button>
            <h1 className="text-2xl font-bold bg-gradient-to-r from-green-600 to-teal-600 bg-clip-text text-transparent">
              <Logo width="140" height="45" variant="white" className="inline-block" />
            </h1>
            <div></div>
          </div>
        </div>
      </header>

      <main className="max-w-4xl mx-auto px-4 py-8">
        <div className="mb-8">
          <div className="flex items-center justify-center space-x-8">
            {[1, 2, 3].map((stepNum) => (
              <div key={stepNum} className="flex items-center">
                <div className={`w-10 h-10 rounded-full flex items-center justify-center font-semibold ${
                  step >= stepNum 
                    ? 'bg-green-500 text-white' 
                    : 'bg-gray-200 text-gray-500'
                }`}>
                  {stepNum}
                </div>
                {stepNum < 3 && (
                  <div className={`w-16 h-1 mx-2 ${
                    step > stepNum ? 'bg-green-500' : 'bg-gray-200'
                  }`} />
                )}
              </div>
            ))}
          </div>
          <div className="flex justify-center mt-4 space-x-16 text-sm text-gray-600">
            <span>Upload Image</span>
            <span>AI Analysis</span>
            <span>Seller Details</span>
          </div>
        </div>

        {step === 1 && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-center"
          >
            <h2 className="text-3xl font-bold mb-8 text-gray-800">Upload Your Artwork</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8 max-w-2xl mx-auto">
              <motion.div
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                className="p-8 rounded-2xl bg-white shadow-lg cursor-pointer border-2 border-dashed border-gray-300 hover:border-green-500 transition-all duration-300"
                onClick={() => setShowCamera(true)}
              >
                <div className="text-6xl mb-4">📷</div>
                <h3 className="text-xl font-semibold mb-2">Use Camera</h3>
                <p className="text-gray-600">Take a photo of your artwork</p>
              </motion.div>
              <motion.div
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                className="p-8 rounded-2xl bg-white shadow-lg cursor-pointer border-2 border-dashed border-gray-300 hover:border-green-500 transition-all duration-300"
                onClick={() => fileInputRef.current?.click()}
              >
                <div className="text-6xl mb-4">🖼️</div>
                <h3 className="text-xl font-semibold mb-2">From Gallery</h3>
                <p className="text-gray-600">Choose from your photos</p>
              </motion.div>
            </div>
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              onChange={handleFileUpload}
              className="hidden"
            />
          </motion.div>
        )}

        {showCamera && (
          <div className="fixed inset-0 bg-black/75 flex items-center justify-center z-50">
            <div className="bg-white p-6 rounded-2xl max-w-md mx-4">
              <Webcam
                ref={webcamRef}
                audio={false}
                screenshotFormat="image/jpeg"
                width={400}
                height={300}
                className="rounded-lg mb-4"
              />
              <div className="flex space-x-4">
                <button
                  onClick={captureImage}
                  className="flex-1 bg-green-500 text-white py-3 rounded-lg font-semibold hover:bg-green-600 transition-colors"
                >
                  📸 Capture
                </button>
                <button
                  onClick={() => setShowCamera(false)}
                  className="flex-1 bg-gray-500 text-white py-3 rounded-lg font-semibold hover:bg-gray-600 transition-colors"
                >
                  Cancel
                </button>
              </div>
            </div>
          </div>
        )}

        {step === 2 && (
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            className="text-center"
          >
            <h2 className="text-3xl font-bold mb-8 text-gray-800">Review Your Image</h2>
            <div className="max-w-md mx-auto mb-8">
              <img
                src={previewUrl}
                alt="Uploaded artwork"
                className="w-full h-64 object-cover rounded-2xl shadow-lg"
              />
            </div>
            <div className="flex justify-center space-x-4">
              <button
                onClick={() => setStep(1)}
                className="px-8 py-4 rounded-2xl font-semibold text-gray-700 bg-gray-200 hover:bg-gray-300 transition-all duration-300"
              >
                Back
              </button>
              <button
                onClick={generateAIDescription}
                disabled={isProcessing}
                className={`px-8 py-4 rounded-2xl font-semibold text-white transition-all duration-300 transform hover:scale-105 ${
                  isProcessing
                    ? 'bg-gray-400 cursor-not-allowed'
                    : 'bg-gradient-to-r from-purple-500 to-blue-500 hover:from-purple-600 hover:to-blue-600 shadow-lg'
                }`}
              >
                {isProcessing ? (
                  <div className="flex items-center space-x-2">
                    <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                    <span>Analyzing with AI...</span>
                  </div>
                ) : (
                  <span>🤖 Generate AI Description & Price</span>
                )}
              </button>
            </div>
          </motion.div>
        )}

        {step === 3 && aiResult && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="max-w-2xl mx-auto"
          >
            <h2 className="text-3xl font-bold mb-8 text-center text-gray-800">Complete Your Listing</h2>
            
            <div className="bg-white rounded-2xl shadow-lg p-6 mb-8">
              <h3 className="text-xl font-semibold mb-4 text-gray-800">🤖 AI Analysis Results</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <img
                    src={previewUrl}
                    alt="Your artwork"
                    className="w-full h-48 object-cover rounded-lg"
                  />
                </div>
                <div className="space-y-4">
                  <div>
                    <label className="text-sm font-medium text-gray-600">Detected Category</label>
                    <p className="text-lg font-semibold text-green-600">{aiResult.analysis.predicted_category}</p>
                  </div>
                  <div>
                    <label className="text-sm font-medium text-gray-600">Suggested Price</label>
                    <div className="flex items-center space-x-2">
                      {isEditingPrice ? (
                        <input
                          type="number"
                          value={editablePrice}
                          onChange={(e) => setEditablePrice(e.target.value)}
                          className="text-xl font-bold text-green-600 border border-gray-300 rounded px-2 py-1"
                          min="1"
                        />
                      ) : (
                        <p className="text-xl font-bold text-green-600">{formatINRPrice(editablePrice, false)}</p>
                      )}
                      <button
                        onClick={handlePriceEdit}
                        className="text-blue-600 hover:text-blue-800 transition-colors"
                      >
                        {isEditingPrice ? '✓' : '✎️'}
                      </button>
                    </div>
                  </div>
                </div>
              </div>
              <div className="mt-4">
                <div className="flex items-center justify-between mb-2">
                  <label className="text-sm font-medium text-gray-600">AI Generated Description</label>
                  <button
                    onClick={handleDescriptionEdit}
                    className="text-blue-600 hover:text-blue-800 transition-colors px-2 py-1 rounded"
                  >
                    {isEditingDescription ? '✓ Save' : '✎️ Edit'}
                  </button>
                </div>
                {isEditingDescription ? (
                  <textarea
                    value={editableDescription}
                    onChange={(e) => setEditableDescription(e.target.value)}
                    className="w-full p-4 border border-gray-300 rounded-lg"
                    rows="4"
                  />
                ) : (
                  <p className="mt-2 p-4 bg-gray-50 rounded-lg text-gray-800">{editableDescription}</p>
                )}
              </div>
            </div>

            <div className="bg-white rounded-2xl shadow-lg p-6">
              <h3 className="text-xl font-semibold mb-6 text-gray-800">📝 Your Details</h3>
              <div className="space-y-6">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Full Name *</label>
                  <input
                    type="text"
                    value={sellerDetails.name}
                    onChange={(e) => setSellerDetails(prev => ({ ...prev, name: e.target.value }))}
                    className="w-full px-4 py-3 border border-gray-300 rounded-lg"
                    placeholder="Enter your full name"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Mobile Number *</label>
                  <input
                    type="tel"
                    value={sellerDetails.mobile}
                    onChange={(e) => setSellerDetails(prev => ({ ...prev, mobile: e.target.value }))}
                    className="w-full px-4 py-3 border border-gray-300 rounded-lg"
                    placeholder="Enter your mobile number"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Location *</label>
                  <input
                    type="text"
                    value={sellerDetails.location}
                    onChange={(e) => setSellerDetails(prev => ({ ...prev, location: e.target.value }))}
                    className="w-full px-4 py-3 border border-gray-300 rounded-lg"
                    placeholder="Enter your city, state"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Price (INR) *</label>
                  <input
                    type="number"
                    value={sellerDetails.price}
                    onChange={(e) => setSellerDetails(prev => ({ ...prev, price: e.target.value }))}
                    className="w-full px-4 py-3 border border-gray-300 rounded-lg"
                    placeholder="Enter your price"
                    min="1"
                  />
                  <p className="mt-1 text-sm text-gray-500">
                    AI suggested: {formatINRPrice(aiResult.pricing_suggestion.suggested_price, false)} 
                    (Range: {formatINRPrice(aiResult.pricing_suggestion.min_price, false)} - {formatINRPrice(aiResult.pricing_suggestion.max_price, false)})
                  </p>
                </div>
              </div>
              
              <div className="flex justify-center space-x-4 mt-8">
                <button
                  onClick={() => setStep(2)}
                  className="px-8 py-4 rounded-2xl font-semibold text-gray-700 bg-gray-200 hover:bg-gray-300"
                >
                  Back
                </button>
                <button
                  onClick={handleSubmitListing}
                  disabled={isSubmitting}
                  className={`w-full py-4 rounded-2xl font-semibold text-white transition-all duration-300 ${
                    isSubmitting
                      ? 'bg-gray-400 cursor-not-allowed'
                      : 'bg-gradient-to-r from-green-500 to-teal-500 hover:from-green-600 hover:to-teal-600 shadow-lg'
                  }`}
                >
                  {isSubmitting ? (
                    <div className="flex items-center justify-center space-x-2">
                      <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                      <span>Creating Listing...</span>
                    </div>
                  ) : (
                    <span>🚀 Create Listing</span>
                  )}
                </button>
              </div>
            </div>
          </motion.div>
        )}
      </main>
    </div>
  );
};

export default SellPage;