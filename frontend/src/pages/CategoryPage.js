import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { useNavigate, useParams } from 'react-router-dom';
import axios from 'axios';
import { formatINRPrice } from '../utils/indianLocalization';

const CategoryPage = () => {
  const navigate = useNavigate();
  const { categoryName } = useParams();
  const [products, setProducts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Category information
  const categoryInfo = {
    'wooden-dolls': {
      name: 'Wooden Dolls',
      icon: '🪆',
      description: 'Hand-carved wooden figurines showcasing traditional craftsmanship',
      color: 'from-amber-500 to-orange-600'
    },
    'handlooms': {
      name: 'Handlooms',
      icon: '🧵',
      description: 'Traditional textiles woven with heritage techniques',
      color: 'from-purple-500 to-pink-600'
    },
    'basket-weaving': {
      name: 'Basket Weaving',
      icon: '🧺',
      description: 'Natural fiber crafts combining function and beauty',
      color: 'from-green-500 to-teal-600'
    },
    'pottery': {
      name: 'Pottery',
      icon: '🏺',
      description: 'Ceramic masterpieces shaped by skilled hands',
      color: 'from-red-500 to-rose-600'
    }
  };

  const currentCategory = categoryInfo[categoryName] || {
    name: 'Category',
    icon: '🎨',
    description: 'Artisan creations',
    color: 'from-blue-500 to-indigo-600'
  };

  useEffect(() => {
    fetchProducts();
  }, [categoryName]);

  const fetchProducts = async () => {
    try {
      setLoading(true);
      const response = await axios.get(`http://localhost:5000/api/products?category=${currentCategory.name}`);
      
      if (response.data.success) {
        setProducts(response.data.products);
      } else {
        setError('Failed to fetch products');
      }
    } catch (err) {
      setError('Error connecting to server');
      console.error('Error fetching products:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleProductClick = (product) => {
    // For now, just show an alert with product details
    alert(`Product: ${product.description.substring(0, 50)}...\nPrice: ${formatINRPrice(product.price, false)}\nArtist: ${product.seller?.name}\nLocation: ${product.seller?.location}`);
  };

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1
      }
    }
  };

  const itemVariants = {
    hidden: { y: 20, opacity: 0 },
    visible: {
      y: 0,
      opacity: 1,
      transition: {
        duration: 0.5
      }
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center">
        <motion.div
          className="text-center"
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
        >
          <div className="w-16 h-16 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
          <p className="text-xl text-gray-600">Loading amazing artworks...</p>
        </motion.div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center">
        <motion.div
          className="text-center bg-white p-8 rounded-2xl shadow-xl"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <div className="text-6xl mb-4">😔</div>
          <h2 className="text-2xl font-bold text-gray-800 mb-4">Oops! Something went wrong</h2>
          <p className="text-gray-600 mb-6">{error}</p>
          <button
            onClick={() => navigate('/')}
            className="px-6 py-3 bg-gradient-to-r from-blue-500 to-purple-500 text-white rounded-full hover:from-blue-600 hover:to-purple-600 transition-all duration-200"
          >
            Back to Home
          </button>
        </motion.div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      {/* Header */}
      <div className="bg-white shadow-lg">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <motion.button
              onClick={() => navigate('/')}
              className="flex items-center space-x-2 text-blue-600 hover:text-blue-700 transition-colors"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              <span className="text-lg">←</span>
              <span className="font-medium">Back to Home</span>
            </motion.button>
            
            <div className="flex items-center space-x-3">
              <span className="text-3xl">{currentCategory.icon}</span>
              <motion.button
                onClick={() => navigate('/')}
                className={`text-2xl font-bold bg-gradient-to-r ${currentCategory.color} bg-clip-text text-transparent cursor-pointer`}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                {currentCategory.name}
              </motion.button>
            </div>
            
            <div></div> {/* Spacer */}
          </div>
        </div>
      </div>

      {/* Category Header */}
      <motion.div
        className="py-12 px-4 text-center"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
      >
        <div className="max-w-4xl mx-auto">
          <motion.div
            className="text-8xl mb-6"
            whileHover={{ scale: 1.1, rotate: 5 }}
            transition={{ duration: 0.3 }}
          >
            {currentCategory.icon}
          </motion.div>
          <h2 className={`text-5xl font-bold mb-4 bg-gradient-to-r ${currentCategory.color} bg-clip-text text-transparent`}>
            {currentCategory.name}
          </h2>
          <p className="text-xl text-gray-600 mb-8">
            {currentCategory.description}
          </p>
          <div className="inline-block bg-white px-6 py-3 rounded-full shadow-lg">
            <span className="text-lg font-semibold text-gray-700">
              {products.length} {products.length === 1 ? 'Product' : 'Products'} Available
            </span>
          </div>
        </div>
      </motion.div>

      {/* Products Grid */}
      {products.length > 0 ? (
        <motion.div
          className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pb-12"
          variants={containerVariants}
          initial="hidden"
          animate="visible"
        >
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            {products.map((product) => (
              <motion.div
                key={product.id}
                variants={itemVariants}
                whileHover={{ y: -8, scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                onClick={() => handleProductClick(product)}
                className="cursor-pointer"
              >
                <div className="bg-white rounded-2xl shadow-xl overflow-hidden hover:shadow-2xl transition-all duration-300">
                  {/* Product Image Placeholder */}
                  <div className={`h-64 bg-gradient-to-br ${currentCategory.color} flex items-center justify-center relative`}>
                    <div className="text-6xl text-white opacity-80">
                      {currentCategory.icon}
                    </div>
                    <div className="absolute top-4 right-4 bg-white/20 backdrop-blur-sm rounded-full px-3 py-1">
                      <span className="text-white font-semibold">{formatINRPrice(product.price, false)}</span>
                    </div>
                  </div>
                  
                  {/* Product Info */}
                  <div className="p-6">
                    <div className="mb-4">
                      <div className="flex items-center justify-between mb-2">
                        <span className={`px-3 py-1 rounded-full text-xs font-semibold bg-gradient-to-r ${currentCategory.color} text-white`}>
                          {product.category}
                        </span>
                        {product.ai_generated && (
                          <span className="px-2 py-1 bg-purple-100 text-purple-600 rounded-full text-xs font-medium">
                            🤖 AI Enhanced
                          </span>
                        )}
                      </div>
                      <p className="text-gray-700 leading-relaxed line-clamp-3">
                        {product.description}
                      </p>
                    </div>
                    
                    {/* Artist Info */}
                    <div className="border-t pt-4">
                      <div className="flex items-center justify-between">
                        <div>
                          <p className="font-semibold text-gray-800">{product.seller?.name}</p>
                          <p className="text-sm text-gray-500">📍 {product.seller?.location}</p>
                        </div>
                        <div className="text-right">
                          <p className="text-2xl font-bold text-gray-800">{formatINRPrice(product.price, false)}</p>
                          <p className="text-xs text-gray-500">Click to learn more</p>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        </motion.div>
      ) : (
        <motion.div
          className="max-w-4xl mx-auto px-4 text-center py-16"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
        >
          <div className="bg-white rounded-2xl shadow-xl p-12">
            <div className="text-6xl mb-6">🎨</div>
            <h3 className="text-2xl font-bold text-gray-800 mb-4">No Products Available Yet</h3>
            <p className="text-gray-600 mb-8">
              This category is ready for amazing {currentCategory.name.toLowerCase()} creations! 
              Be the first artisan to showcase your beautiful handcrafted pieces in this category.
            </p>
            <div className="bg-blue-50 p-6 rounded-xl mb-8">
              <h4 className="font-semibold text-blue-800 mb-2">✨ Perfect for {currentCategory.name}:</h4>
              <div className="text-blue-700 space-y-1">
                {currentCategory.name === 'Wooden Dolls' && (
                  <>
                    <p>• Traditional carved figurines</p>
                    <p>• Decorative wooden art</p>
                    <p>• Cultural heritage pieces</p>
                  </>
                )}
                {currentCategory.name === 'Handlooms' && (
                  <>
                    <p>• Traditional textiles</p>
                    <p>• Handwoven fabrics</p>
                    <p>• Regional patterns</p>
                  </>
                )}
                {currentCategory.name === 'Basket Weaving' && (
                  <>
                    <p>• Natural fiber crafts</p>
                    <p>• Functional baskets</p>
                    <p>• Traditional techniques</p>
                  </>
                )}
                {currentCategory.name === 'Pottery' && (
                  <>
                    <p>• Ceramic artworks</p>
                    <p>• Traditional pottery</p>
                    <p>• Functional pieces</p>
                  </>
                )}
              </div>
            </div>
            <div className="space-y-4">
              <button
                onClick={() => navigate('/sell')}
                className={`px-8 py-3 bg-gradient-to-r ${currentCategory.color} text-white rounded-full hover:scale-105 transition-all duration-200 font-semibold mr-4`}
              >
                🎨 List Your Product
              </button>
              <button
                onClick={() => navigate('/')}
                className="px-8 py-3 bg-gray-200 text-gray-700 rounded-full hover:bg-gray-300 transition-all duration-200 font-semibold"
              >
                🏠 Back to Home
              </button>
            </div>
          </div>
        </motion.div>
      )}
    </div>
  );
};

export default CategoryPage;