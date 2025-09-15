import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import { getCurrentUser, logout, getUserGreeting } from '../utils/auth';
import Logo from '../components/Logo';

const ProfilePage = () => {
  const [activeTab, setActiveTab] = useState('profile');
  const [currentUser, setCurrentUser] = useState(null);
  const [showPrivacyModal, setShowPrivacyModal] = useState(false);
  const [privacySettings, setPrivacySettings] = useState({
    profileVisibility: 'public', // 'public', 'friends', 'private'
    showEmail: true,
    showPhone: false,
    showLocation: true,
    allowMessages: true,
    showActivity: true
  });
  const [userInfo, setUserInfo] = useState({
    name: '',
    email: '',
    phone: '+91 (555) 123-4567',
    location: 'India',
    memberSince: 'January 2024'
  });
  
  const navigate = useNavigate();

  const handlePrivacyUpdate = (key, value) => {
    setPrivacySettings(prev => ({
      ...prev,
      [key]: value
    }));
  };

  const savePrivacySettings = () => {
    // In a real app, this would save to backend
    localStorage.setItem('privacySettings', JSON.stringify(privacySettings));
    alert('Privacy settings saved successfully!');
    setShowPrivacyModal(false);
  };

  useEffect(() => {
    // Load privacy settings from localStorage
    const savedPrivacy = localStorage.getItem('privacySettings');
    if (savedPrivacy) {
      setPrivacySettings(JSON.parse(savedPrivacy));
    }
  }, []);

  useEffect(() => {
    const user = getCurrentUser();
    if (user) {
      setCurrentUser(user);
      setUserInfo(prev => ({
        ...prev,
        name: user.name,
        email: user.email
      }));
    }
  }, []);

  const handleLogout = () => {
    logout();
    navigate('/auth');
  };

  const stats = [
    { label: 'Products Listed', value: '0', icon: '🎨' },
    { label: 'Total Sales', value: '₹0', icon: '💰' },
    { label: 'Reviews', value: '0', icon: '⭐' },
    { label: 'Wishlist Items', value: '0', icon: '❤️' }
  ];

  const recentActivity = [
    // Empty array - no activity yet as this is a new platform
  ];

  const achievements = [
    { title: 'Welcome', description: 'Joined Artisans Hub platform', earned: true, icon: '🎉' },
    { title: 'First Sale', description: 'Complete your first transaction', earned: false, icon: '💰' },
    { title: 'Trusted Seller', description: 'Maintain 4.8+ rating with 10+ reviews', earned: false, icon: '🌟' },
    { title: 'Cultural Ambassador', description: 'Share 10+ cultural stories', earned: false, icon: '🏛️' },
    { title: 'Eco Warrior', description: 'List 20+ eco-friendly items', earned: false, icon: '🌱' }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-purple-50">
      {/* Header */}
      <header className="bg-white/80 backdrop-blur-md shadow-lg">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <button
              onClick={() => navigate('/')}
              className="flex items-center space-x-2 text-gray-600 hover:text-gray-800 transition-colors"
            >
              <span className="text-xl">←</span>
              <span>Back to Home</span>
            </button>
            <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
              My Profile
            </h1>
            <div></div>
          </div>
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-4 py-8">
        {/* Profile Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-white rounded-2xl shadow-lg p-8 mb-8"
        >
          <div className="flex flex-col md:flex-row items-center space-y-4 md:space-y-0 md:space-x-6">
            <div className="w-24 h-24 bg-gradient-to-r from-blue-500 to-purple-500 rounded-full flex items-center justify-center text-white text-4xl font-bold">
              {userInfo.name.split(' ').map(n => n[0]).join('')}
            </div>
            <div className="text-center md:text-left flex-1">
              <h2 className="text-3xl font-bold text-gray-800">{userInfo.name}</h2>
              <p className="text-gray-600">{userInfo.email}</p>
              <p className="text-gray-500">Member since {userInfo.memberSince}</p>
            </div>
            <div className="flex space-x-3">
              <button 
                onClick={handleLogout}
                className="px-6 py-3 bg-gray-200 text-gray-700 rounded-lg font-semibold hover:bg-gray-300 transition-all duration-200"
              >
                Logout
              </button>
              <button className="px-6 py-3 bg-gradient-to-r from-blue-500 to-purple-500 text-white rounded-lg font-semibold hover:from-blue-600 hover:to-purple-600 transition-all duration-200">
                Edit Profile
              </button>
            </div>
          </div>
        </motion.div>

        {/* Stats Cards */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="grid grid-cols-2 md:grid-cols-4 gap-6 mb-8"
        >
          {stats.map((stat, index) => (
            <div key={index} className="bg-white rounded-xl shadow-lg p-6 text-center">
              <div className="text-3xl mb-2">{stat.icon}</div>
              <div className="text-2xl font-bold text-gray-800">{stat.value}</div>
              <div className="text-sm text-gray-600">{stat.label}</div>
            </div>
          ))}
        </motion.div>

        {/* Tabs */}
        <div className="mb-8">
          <div className="flex space-x-1 bg-gray-100 p-1 rounded-lg">
            {[
              { id: 'profile', label: 'Profile Info', icon: '👤' },
              { id: 'activity', label: 'Recent Activity', icon: '📊' },
              { id: 'achievements', label: 'Achievements', icon: '🏆' },
              { id: 'settings', label: 'Settings', icon: '⚙️' }
            ].map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex-1 flex items-center justify-center space-x-2 py-3 px-4 rounded-lg font-medium transition-all duration-200 ${
                  activeTab === tab.id
                    ? 'bg-white text-blue-600 shadow-md'
                    : 'text-gray-600 hover:text-gray-800'
                }`}
              >
                <span>{tab.icon}</span>
                <span className="hidden sm:inline">{tab.label}</span>
              </button>
            ))}
          </div>
        </div>

        {/* Tab Content */}
        <motion.div
          key={activeTab}
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.3 }}
        >
          {/* Profile Info Tab */}
          {activeTab === 'profile' && (
            <div className="bg-white rounded-2xl shadow-lg p-8">
              <h3 className="text-2xl font-bold text-gray-800 mb-6">Profile Information</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Full Name</label>
                  <input
                    type="text"
                    value={userInfo.name}
                    onChange={(e) => setUserInfo(prev => ({ ...prev, name: e.target.value }))}
                    className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Email</label>
                  <input
                    type="email"
                    value={userInfo.email}
                    onChange={(e) => setUserInfo(prev => ({ ...prev, email: e.target.value }))}
                    className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Phone</label>
                  <input
                    type="tel"
                    value={userInfo.phone}
                    onChange={(e) => setUserInfo(prev => ({ ...prev, phone: e.target.value }))}
                    className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Location</label>
                  <input
                    type="text"
                    value={userInfo.location}
                    onChange={(e) => setUserInfo(prev => ({ ...prev, location: e.target.value }))}
                    className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                </div>
              </div>
              <button className="mt-6 px-6 py-3 bg-gradient-to-r from-blue-500 to-purple-500 text-white rounded-lg font-semibold hover:from-blue-600 hover:to-purple-600 transition-all duration-200">
                Save Changes
              </button>
            </div>
          )}

          {/* Recent Activity Tab */}
          {activeTab === 'activity' && (
            <div className="bg-white rounded-2xl shadow-lg p-8">
              <h3 className="text-2xl font-bold text-gray-800 mb-6">Recent Activity</h3>
              {recentActivity.length > 0 ? (
                <div className="space-y-4">
                  {recentActivity.map((activity) => (
                    <div key={activity.id} className="flex items-center space-x-4 p-4 bg-gray-50 rounded-lg">
                      <div className="text-2xl">{activity.icon}</div>
                      <div className="flex-1">
                        <p className="font-medium text-gray-800">{activity.action}</p>
                        <p className="text-sm text-gray-500">{activity.time}</p>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-12">
                  <div className="text-6xl mb-4">🎆</div>
                  <h4 className="text-xl font-semibold text-gray-800 mb-2">No Activity Yet!</h4>
                  <p className="text-gray-600 mb-6">
                    Welcome to Artisans Hub! Start your journey by uploading your first product or exploring amazing artisan creations.
                  </p>
                  <div className="space-x-4">
                    <button
                      onClick={() => navigate('/sell')}
                      className="px-6 py-3 bg-gradient-to-r from-blue-500 to-purple-500 text-white rounded-lg font-semibold hover:from-blue-600 hover:to-purple-600 transition-all duration-200"
                    >
                      🎨 Start Selling
                    </button>
                    <button
                      onClick={() => navigate('/')}
                      className="px-6 py-3 bg-gray-200 text-gray-700 rounded-lg font-semibold hover:bg-gray-300 transition-all duration-200"
                    >
                      🔍 Explore Marketplace
                    </button>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Achievements Tab */}
          {activeTab === 'achievements' && (
            <div className="bg-white rounded-2xl shadow-lg p-8">
              <h3 className="text-2xl font-bold text-gray-800 mb-6">Achievements</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {achievements.map((achievement, index) => (
                  <div
                    key={index}
                    className={`p-6 rounded-xl border-2 ${
                      achievement.earned
                        ? 'border-green-300 bg-green-50'
                        : 'border-gray-200 bg-gray-50'
                    }`}
                  >
                    <div className="flex items-center space-x-3 mb-3">
                      <div className={`text-3xl ${achievement.earned ? '' : 'grayscale'}`}>
                        {achievement.icon}
                      </div>
                      <div>
                        <h4 className={`font-semibold ${
                          achievement.earned ? 'text-green-800' : 'text-gray-600'
                        }`}>
                          {achievement.title}
                        </h4>
                        {achievement.earned && (
                          <span className="text-xs bg-green-200 text-green-800 px-2 py-1 rounded-full">
                            Earned
                          </span>
                        )}
                      </div>
                    </div>
                    <p className={achievement.earned ? 'text-green-700' : 'text-gray-500'}>
                      {achievement.description}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Settings Tab */}
          {activeTab === 'settings' && (
            <div className="bg-white rounded-2xl shadow-lg p-8">
              <h3 className="text-2xl font-bold text-gray-800 mb-6">Settings</h3>
              <div className="space-y-6">
                <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                  <div>
                    <h4 className="font-medium text-gray-800">Email Notifications</h4>
                    <p className="text-sm text-gray-600">Receive updates about your listings and sales</p>
                  </div>
                  <label className="relative inline-flex items-center cursor-pointer">
                    <input type="checkbox" className="sr-only peer" defaultChecked />
                    <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
                  </label>
                </div>
                
                <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                  <div>
                    <h4 className="font-medium text-gray-800">Push Notifications</h4>
                    <p className="text-sm text-gray-600">Get notified about new messages and updates</p>
                  </div>
                  <label className="relative inline-flex items-center cursor-pointer">
                    <input type="checkbox" className="sr-only peer" />
                    <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
                  </label>
                </div>

                <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                  <div>
                    <h4 className="font-medium text-gray-800">Privacy Settings</h4>
                    <p className="text-sm text-gray-600">Control who can see your profile information</p>
                  </div>
                  <button 
                    onClick={() => setShowPrivacyModal(true)}
                    className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors"
                  >
                    Manage
                  </button>
                </div>
              </div>
            </div>
          )}
        </motion.div>
      </main>

      {/* Privacy Settings Modal */}
      {showPrivacyModal && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            className="bg-white rounded-2xl shadow-xl max-w-2xl w-full max-h-[90vh] overflow-y-auto"
          >
            <div className="p-6">
              <div className="flex justify-between items-center mb-6">
                <h2 className="text-2xl font-bold text-gray-800">🔒 Privacy Settings</h2>
                <button
                  onClick={() => setShowPrivacyModal(false)}
                  className="p-2 hover:bg-gray-100 rounded-full transition-colors"
                >
                  ✕
                </button>
              </div>

              <div className="space-y-6">
                {/* Profile Visibility */}
                <div className="border-b border-gray-200 pb-4">
                  <h3 className="text-lg font-semibold text-gray-800 mb-3">Profile Visibility</h3>
                  <div className="space-y-3">
                    {[
                      { value: 'public', label: 'Public', desc: 'Anyone can see your profile' },
                      { value: 'friends', label: 'Friends Only', desc: 'Only your connections can see your profile' },
                      { value: 'private', label: 'Private', desc: 'Only you can see your profile' }
                    ].map((option) => (
                      <label key={option.value} className="flex items-start space-x-3 cursor-pointer">
                        <input
                          type="radio"
                          name="profileVisibility"
                          value={option.value}
                          checked={privacySettings.profileVisibility === option.value}
                          onChange={(e) => handlePrivacyUpdate('profileVisibility', e.target.value)}
                          className="mt-1 text-blue-600"
                        />
                        <div>
                          <div className="font-medium text-gray-800">{option.label}</div>
                          <div className="text-sm text-gray-600">{option.desc}</div>
                        </div>
                      </label>
                    ))}
                  </div>
                </div>

                {/* Information Visibility */}
                <div className="border-b border-gray-200 pb-4">
                  <h3 className="text-lg font-semibold text-gray-800 mb-3">Information Visibility</h3>
                  <div className="space-y-4">
                    {[
                      { key: 'showEmail', label: 'Show Email Address', desc: 'Let others see your email' },
                      { key: 'showPhone', label: 'Show Phone Number', desc: 'Let others see your phone number' },
                      { key: 'showLocation', label: 'Show Location', desc: 'Let others see your city/location' },
                      { key: 'showActivity', label: 'Show Recent Activity', desc: 'Let others see your recent marketplace activity' }
                    ].map((setting) => (
                      <div key={setting.key} className="flex items-center justify-between">
                        <div>
                          <div className="font-medium text-gray-800">{setting.label}</div>
                          <div className="text-sm text-gray-600">{setting.desc}</div>
                        </div>
                        <label className="relative inline-flex items-center cursor-pointer">
                          <input
                            type="checkbox"
                            checked={privacySettings[setting.key]}
                            onChange={(e) => handlePrivacyUpdate(setting.key, e.target.checked)}
                            className="sr-only peer"
                          />
                          <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
                        </label>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Communication Settings */}
                <div className="pb-4">
                  <h3 className="text-lg font-semibold text-gray-800 mb-3">Communication</h3>
                  <div className="flex items-center justify-between">
                    <div>
                      <div className="font-medium text-gray-800">Allow Messages</div>
                      <div className="text-sm text-gray-600">Let other users send you messages</div>
                    </div>
                    <label className="relative inline-flex items-center cursor-pointer">
                      <input
                        type="checkbox"
                        checked={privacySettings.allowMessages}
                        onChange={(e) => handlePrivacyUpdate('allowMessages', e.target.checked)}
                        className="sr-only peer"
                      />
                      <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
                    </label>
                  </div>
                </div>
              </div>

              {/* Action Buttons */}
              <div className="flex space-x-4 mt-8">
                <button
                  onClick={() => setShowPrivacyModal(false)}
                  className="flex-1 px-6 py-3 bg-gray-200 text-gray-700 rounded-lg font-semibold hover:bg-gray-300 transition-colors"
                >
                  Cancel
                </button>
                <button
                  onClick={savePrivacySettings}
                  className="flex-1 px-6 py-3 bg-blue-500 text-white rounded-lg font-semibold hover:bg-blue-600 transition-colors"
                >
                  Save Settings
                </button>
              </div>
            </div>
          </motion.div>
        </div>
      )}
    </div>
  );
};

export default ProfilePage;