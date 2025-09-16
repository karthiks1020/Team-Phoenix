import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import { INDIAN_COLORS, INDIAN_ICONS, getIndianGreeting } from '../utils/indianLocalization';
import Logo from '../components/Logo';

const HomePage = () => {
  const [searchTerm, setSearchTerm] = useState('');
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [showTitle, setShowTitle] = useState(false);
  const [greeting, setGreeting] = useState(getIndianGreeting());
  const navigate = useNavigate();

  useEffect(() => {
    const timer = setTimeout(() => {
      setShowTitle(true);
    }, 500);
    return () => clearTimeout(timer);
  }, []);

  const categories = [
    {
      id: 1,
      name: 'Wooden Dolls',
      icon: '🪆',
      description: 'Hand-carved figurines',
      color: INDIAN_COLORS.heritage
    },
    {
      id: 2,
      name: 'Handlooms',
      icon: '🧵',
      description: 'Traditional textiles',
      color: INDIAN_COLORS.royal
    },
    {
      id: 3,
      name: 'Basket Weaving',
      icon: '🧺',
      description: 'Natural fiber crafts',
      color: INDIAN_COLORS.earth
    },
    {
      id: 4,
      name: 'Pottery',
      icon: '🏺',
      description: 'Ceramic masterpieces',
      color: INDIAN_COLORS.peacock
    }
  ];

  const handleSearch = (e) => {
    e.preventDefault();
    if (searchTerm.trim()) {
      // Navigate to search results page
      navigate(`/search?q=${encodeURIComponent(searchTerm.trim())}`);
    }
  };

  const handleCategoryClick = (category) => {
    console.log('Selected category:', category.name);
    // Convert category name to URL-friendly format
    const categoryUrl = category.name.toLowerCase().replace(/\s+/g, '-');
    navigate(`/category/${categoryUrl}`);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50">
      {/* Header with Navigation */}
      <header className="relative z-50">
        <nav className="bg-white/80 backdrop-blur-md shadow-lg">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex justify-between items-center h-16">
              {/* Logo */}
              <div className="flex-shrink-0">
                <motion.button
                  onClick={() => navigate('/')}
                  className="cursor-pointer"
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  <Logo width="140" height="45" className="hover:opacity-90 transition-opacity" />
                </motion.button>
              </div>

              {/* Right side navigation */}
              <div className="flex items-center space-x-4">

                {/* Profile Icon */}
                <motion.button
                  onClick={() => navigate('/profile')}
                  className="p-2 rounded-full bg-gray-100 hover:bg-gray-200 transition-all duration-200"
                  whileHover={{ scale: 1.1 }}
                  whileTap={{ scale: 0.9 }}
                >
                  <div className="w-8 h-8 bg-gradient-to-r from-purple-500 to-blue-500 rounded-full flex items-center justify-center text-white font-semibold">
                    👤
                  </div>
                </motion.button>

                {/* Hamburger Menu */}
                <motion.button
                  onClick={() => setIsMenuOpen(!isMenuOpen)}
                  className="p-2 rounded-lg bg-gray-100 hover:bg-gray-200 transition-all duration-200"
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  <div className="w-6 h-6 flex flex-col justify-center items-center">
                    <motion.span 
                      className="w-5 h-0.5 bg-gray-700 mb-1 transition-all duration-300"
                      animate={isMenuOpen ? { rotate: 45, y: 6 } : { rotate: 0, y: 0 }}
                    />
                    <motion.span 
                      className="w-5 h-0.5 bg-gray-700 mb-1 transition-all duration-300"
                      animate={isMenuOpen ? { opacity: 0 } : { opacity: 1 }}
                    />
                    <motion.span 
                      className="w-5 h-0.5 bg-gray-700 transition-all duration-300"
                      animate={isMenuOpen ? { rotate: -45, y: -6 } : { rotate: 0, y: 0 }}
                    />
                  </div>
                </motion.button>
              </div>
            </div>
          </div>

          {/* Slide-out menu */}
          <motion.div
            className="absolute top-full right-0 w-80 bg-white shadow-xl rounded-bl-2xl z-40"
            initial={{ x: '100%', opacity: 0 }}
            animate={isMenuOpen ? { x: 0, opacity: 1 } : { x: '100%', opacity: 0 }}
            transition={{ duration: 0.3, ease: 'easeInOut' }}
          >
            <div className="p-6 space-y-4">
              <motion.button
                onClick={() => navigate('/sell')}
                className="w-full text-left p-4 rounded-lg bg-gradient-to-r from-green-500 to-teal-500 text-white font-semibold hover:from-green-600 hover:to-teal-600 transition-all duration-200"
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                🛍️ Sell
              </motion.button>
              
              <motion.button
                onClick={() => navigate('/cart')}
                className="w-full text-left p-4 rounded-lg bg-gradient-to-r from-blue-500 to-purple-500 text-white font-semibold hover:from-blue-600 hover:to-purple-600 transition-all duration-200"
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                🛒 Add to Cart
              </motion.button>
              
              <motion.button
                onClick={() => navigate('/chatbot')}
                className="w-full text-left p-4 rounded-lg bg-gradient-to-r from-pink-500 to-rose-500 text-white font-semibold hover:from-pink-600 hover:to-rose-600 transition-all duration-200"
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                🤖 My Artist Friend
              </motion.button>
              
              <motion.button
                onClick={() => navigate('/about')}
                className="w-full text-left p-4 rounded-lg bg-gradient-to-r from-amber-500 to-orange-500 text-white font-semibold hover:from-amber-600 hover:to-orange-600 transition-all duration-200"
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                ℹ️ About
              </motion.button>
              
              <motion.button
                onClick={() => navigate('/settings')}
                className="w-full text-left p-4 rounded-lg bg-gradient-to-r from-gray-500 to-slate-500 text-white font-semibold hover:from-gray-600 hover:to-slate-600 transition-all duration-200"
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                ⚙️ Settings
              </motion.button>
            </div>
          </motion.div>
        </nav>
      </header>

      {/* Main Content */}
      <main className="relative">
        {/* Animated Title Section */}
        <section className="py-20 px-4">
          <div className="max-w-4xl mx-auto text-center">
            <motion.div
              initial={{ opacity: 0, y: 50 }}
              animate={showTitle ? { opacity: 1, y: 0 } : { opacity: 0, y: 50 }}
              transition={{ duration: 0.8, delay: 0.2 }}
            >
              <h1 className="text-6xl md:text-8xl font-bold mb-6">
                <span className="bg-gradient-to-r from-purple-600 via-blue-600 to-teal-600 bg-clip-text text-transparent animate-pulse-slow">
                  Artisans Hub
                </span>
              </h1>
              <motion.p 
                className="text-xl md:text-2xl text-gray-600 mb-12"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 0.8, delay: 0.8 }}
              >
                {greeting.emoji} {greeting.en} - Where Traditional Crafts Meet Modern Technology ✨
              </motion.p>
            </motion.div>

            {/* Search Bar */}
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 1.0 }}
              className="max-w-2xl mx-auto mb-16"
            >
              <form onSubmit={handleSearch} className="relative">
                <input
                  type="text"
                  placeholder="Search for handcrafted treasures..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="w-full px-6 py-4 text-lg rounded-full border-2 border-gray-200 focus:border-purple-500 focus:outline-none shadow-lg bg-white/80 backdrop-blur-sm"
                />
                <button
                  type="submit"
                  className="absolute right-2 top-2 px-6 py-2 bg-gradient-to-r from-purple-500 to-blue-500 text-white rounded-full hover:from-purple-600 hover:to-blue-600 transition-all duration-200 font-semibold"
                >
                  🔍 Search
                </button>
              </form>
            </motion.div>
          </div>
        </section>

        {/* Categories Grid */}
        <section className="py-16 px-4">
          <div className="max-w-6xl mx-auto">
            <motion.h2 
              className="text-4xl font-bold text-center mb-12 text-gray-800"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 1.2 }}
            >
              Explore Our Categories
            </motion.h2>
            
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
              {categories.map((category, index) => (
                <motion.div
                  key={category.id}
                  initial={{ opacity: 0, y: 50, scale: 0.9 }}
                  animate={{ opacity: 1, y: 0, scale: 1 }}
                  transition={{ duration: 0.5, delay: 1.4 + index * 0.1 }}
                  whileHover={{ y: -10, scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={() => handleCategoryClick(category)}
                  className="cursor-pointer"
                >
                  <div className={`p-8 rounded-2xl bg-gradient-to-br ${category.color} text-white shadow-xl hover:shadow-2xl transition-all duration-300 text-center`}>
                    <motion.div 
                      className="text-6xl mb-4"
                      whileHover={{ rotate: 360 }}
                      transition={{ duration: 0.6 }}
                    >
                      {category.icon}
                    </motion.div>
                    <h3 className="text-xl font-bold mb-2">{category.name}</h3>
                    <p className="text-white/90">{category.description}</p>
                  </div>
                </motion.div>
              ))}
            </div>
          </div>
        </section>

        {/* Features Section */}
        <section className="py-20 px-4 bg-white/50">
          <div className="max-w-6xl mx-auto">
            <motion.h2 
              className="text-4xl font-bold text-center mb-12 text-gray-800"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.6, delay: 2.0 }}
            >
              Why Choose Artisans Hub?
            </motion.h2>
            
            <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
              {[
                {
                  icon: '🤖',
                  title: 'AI-Powered',
                  description: 'Smart image recognition and automated descriptions'
                },
                {
                  icon: '🌍',
                  title: 'Global Reach',
                  description: 'Connect local artisans with worldwide customers'
                },
                {
                  icon: '🎨',
                  title: 'Cultural Heritage',
                  description: 'Preserving traditional crafts for future generations'
                }
              ].map((feature, index) => (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, y: 30 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5, delay: 2.2 + index * 0.2 }}
                  className="text-center p-8 rounded-xl bg-white shadow-lg hover:shadow-xl transition-all duration-300"
                >
                  <div className="text-5xl mb-4">{feature.icon}</div>
                  <h3 className="text-xl font-bold mb-3 text-gray-800">{feature.title}</h3>
                  <p className="text-gray-600">{feature.description}</p>
                </motion.div>
              ))}
            </div>
          </div>
        </section>
      </main>

      {/* Overlay for menu */}
      {isMenuOpen && (
        <motion.div
          className="fixed inset-0 bg-black/20 z-30"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          onClick={() => setIsMenuOpen(false)}
        />
      )}
    </div>
  );
};

export default HomePage;