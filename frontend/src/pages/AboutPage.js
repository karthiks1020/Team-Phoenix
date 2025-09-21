import React from 'react';
import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';

const AboutPage = () => {
  const navigate = useNavigate();

  const containerVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: { 
      opacity: 1, 
      y: 0,
      transition: {
        duration: 0.6,
        staggerChildren: 0.1
      }
    }
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: { opacity: 1, y: 0 }
  };

  const missionItems = [
    {
      title: "Support Artisans",
      description: "by giving them fair pricing, visibility, and tools to sell easily.",
      icon: "👥",
      color: "from-blue-500 to-purple-600"
    },
    {
      title: "Preserve Culture",
      description: "by promoting handcrafted art forms that reflect heritage.",
      icon: "🌍",
      color: "from-green-500 to-teal-600"
    },
    {
      title: "Enable Buyers",
      description: "by offering authentic, AI-assisted recommendations and product discovery.",
      icon: "❤️",
      color: "from-pink-500 to-red-600"
    }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-amber-50 via-orange-50 to-red-50">
      {/* Header */}
      <div className="bg-white shadow-lg">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <motion.button
              onClick={() => navigate('/')}
              className="flex items-center space-x-2 text-amber-600 hover:text-amber-700 transition-colors"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              <span className="text-lg">←</span>
              <span className="font-medium">Back to Home</span>
            </motion.button>
            
            <motion.button
              onClick={() => navigate('/')}
              className="text-2xl font-bold bg-gradient-to-r from-amber-600 to-orange-600 bg-clip-text text-transparent cursor-pointer"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              About Artisans Hub
            </motion.button>
            
            <div></div> {/* Spacer for center alignment */}
          </div>
        </div>
      </div>

      {/* Main Content */}
      <motion.div 
        className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-12"
        variants={containerVariants}
        initial="hidden"
        animate="visible"
      >
        {/* Hero Section */}
        <motion.div 
          className="text-center mb-16"
          variants={itemVariants}
        >
          <motion.div
            className="inline-block mb-6"
            whileHover={{ rotate: 5, scale: 1.05 }}
          >
            <div className="w-24 h-24 mx-auto bg-gradient-to-br from-amber-500 to-orange-600 rounded-full flex items-center justify-center shadow-xl">
              <svg className="w-12 h-12 text-white" fill="currentColor" viewBox="0 0 24 24">
                <path d="M12 2L2 7v10c0 5.55 3.84 9.74 9 10.66C16.16 26.74 20 22.55 20 17V7l-8-5z"/>
              </svg>
            </div>
          </motion.div>
          
          <h2 className="text-4xl md:text-5xl font-bold text-gray-800 mb-6 leading-tight">
            Empowering Local{' '}
            <span className="bg-gradient-to-r from-amber-600 to-orange-600 bg-clip-text text-transparent">
              Artisans
            </span>
          </h2>
          
          <p className="text-xl text-gray-600 max-w-3xl mx-auto leading-relaxed">
            Artisans Hub is an AI-powered marketplace built to empower local artisans and craftsmen by giving them a digital space to showcase, sell, and grow their unique creations.
          </p>
        </motion.div>

        {/* Description Section */}
        <motion.div 
          className="bg-white rounded-2xl shadow-xl p-8 md:p-12 mb-12"
          variants={itemVariants}
          whileHover={{ y: -5 }}
        >
          <p className="text-lg text-gray-700 leading-relaxed mb-6">
            From wooden dolls and handlooms to basket weaving and pottery, our platform connects traditional skills with modern buyers.
          </p>
          
          <div className="bg-gradient-to-r from-amber-100 to-orange-100 rounded-xl p-6">
            <h3 className="text-2xl font-bold text-gray-800 mb-4">Our Mission is Simple:</h3>
            
            <div className="grid md:grid-cols-3 gap-6">
              {missionItems.map((item, index) => (
                <motion.div
                  key={index}
                  className="text-center"
                  variants={itemVariants}
                  whileHover={{ scale: 1.05 }}
                >
                  <div className={`w-16 h-16 mx-auto mb-4 bg-gradient-to-r ${item.color} rounded-full flex items-center justify-center text-white shadow-lg text-2xl`}>
                    {item.icon}
                  </div>
                  <h4 className="text-lg font-bold text-gray-800 mb-2">{item.title}</h4>
                  <p className="text-gray-600 text-sm leading-relaxed">{item.description}</p>
                </motion.div>
              ))}
            </div>
          </div>
        </motion.div>

        {/* Features Section */}
        <motion.div 
          className="bg-white rounded-2xl shadow-xl p-8 md:p-12"
          variants={itemVariants}
          whileHover={{ y: -5 }}
        >
          <h3 className="text-2xl font-bold text-gray-800 mb-6 text-center">Beyond Just a Marketplace</h3>
          
          <p className="text-lg text-gray-700 leading-relaxed mb-8 text-center">
            With features like AI-generated descriptions, fair price suggestions, and "My Artist Friend" – an AI guide chatbot, Artisans Hub goes beyond being just a marketplace.
          </p>
          
          <div className="grid md:grid-cols-2 gap-8">
            <div className="space-y-4">
              <div className="flex items-start space-x-4">
                <div className="w-3 h-3 bg-gradient-to-r from-amber-500 to-orange-600 rounded-full mt-2 flex-shrink-0"></div>
                <div>
                  <h4 className="font-semibold text-gray-800">AI-Generated Descriptions</h4>
                  <p className="text-gray-600 text-sm">Smart product descriptions that highlight craftsmanship</p>
                </div>
              </div>
              
              <div className="flex items-start space-x-4">
                <div className="w-3 h-3 bg-gradient-to-r from-amber-500 to-orange-600 rounded-full mt-2 flex-shrink-0"></div>
                <div>
                  <h4 className="font-semibold text-gray-800">Fair Price Suggestions</h4>
                  <p className="text-gray-600 text-sm">AI-powered pricing that ensures fair value</p>
                </div>
              </div>
            </div>
            
            <div className="space-y-4">
              <div className="flex items-start space-x-4">
                <div className="w-3 h-3 bg-gradient-to-r from-amber-500 to-orange-600 rounded-full mt-2 flex-shrink-0"></div>
                <div>
                  <h4 className="font-semibold text-gray-800">"My Artist Friend" Chatbot</h4>
                  <p className="text-gray-600 text-sm">AI guide to help navigate the platform</p>
                </div>
              </div>
              
              <div className="flex items-start space-x-4">
                <div className="w-3 h-3 bg-gradient-to-r from-amber-500 to-orange-600 rounded-full mt-2 flex-shrink-0"></div>
                <div>
                  <h4 className="font-semibold text-gray-800">Cultural Preservation</h4>
                  <p className="text-gray-600 text-sm">Documenting and promoting traditional crafts</p>
                </div>
              </div>
            </div>
          </div>
          
          <div className="text-center mt-10">
            <div className="bg-gradient-to-r from-amber-600 to-orange-600 rounded-2xl p-8 text-white">
              <h4 className="text-xl font-bold mb-4">Our Vision</h4>
              <p className="text-lg leading-relaxed">
                We aim to become a community that celebrates creativity, culture, and craftsmanship.
              </p>
            </div>
          </div>
        </motion.div>
      </motion.div>
    </div>
  );
};

export default AboutPage;