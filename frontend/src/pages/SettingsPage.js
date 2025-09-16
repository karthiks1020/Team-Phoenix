import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import Logo from '../components/Logo';

const SettingsPage = () => {
  const navigate = useNavigate();
  const [showPrivacyModal, setShowPrivacyModal] = useState(false);
  const [privacySettings, setPrivacySettings] = useState({
    profileVisibility: 'public', // 'public', 'friends', 'private'
    showEmail: true,
    showPhone: false,
    showLocation: true,
    allowMessages: true,
    showActivity: true
  });

  const [notifications, setNotifications] = useState({
    emailNotifications: true,
    pushNotifications: true,
    orderUpdates: true,
    marketingEmails: false
  });

  const [accountSettings, setAccountSettings] = useState({
    language: 'english',
    currency: 'INR',
    theme: 'light'
  });

  useEffect(() => {
    // Load settings from localStorage
    const savedPrivacy = localStorage.getItem('privacySettings');
    const savedNotifications = localStorage.getItem('notificationSettings');
    const savedAccount = localStorage.getItem('accountSettings');
    
    if (savedPrivacy) {
      setPrivacySettings(JSON.parse(savedPrivacy));
    }
    if (savedNotifications) {
      setNotifications(JSON.parse(savedNotifications));
    }
    if (savedAccount) {
      setAccountSettings(JSON.parse(savedAccount));
    }
  }, []);

  const handlePrivacyUpdate = (key, value) => {
    setPrivacySettings(prev => ({
      ...prev,
      [key]: value
    }));
  };

  const handleNotificationUpdate = (key, value) => {
    setNotifications(prev => ({
      ...prev,
      [key]: value
    }));
  };

  const handleAccountUpdate = (key, value) => {
    setAccountSettings(prev => ({
      ...prev,
      [key]: value
    }));
  };

  const savePrivacySettings = () => {
    localStorage.setItem('privacySettings', JSON.stringify(privacySettings));
    alert('Privacy settings saved successfully!');
    setShowPrivacyModal(false);
  };

  const saveAllSettings = () => {
    localStorage.setItem('privacySettings', JSON.stringify(privacySettings));
    localStorage.setItem('notificationSettings', JSON.stringify(notifications));
    localStorage.setItem('accountSettings', JSON.stringify(accountSettings));
    alert('All settings saved successfully!');
  };

  const privacyOptions = [
    { value: 'public', label: 'Public', description: 'Anyone can see your profile' },
    { value: 'friends', label: 'Friends Only', description: 'Only your connections can see' },
    { value: 'private', label: 'Private', description: 'Only you can see your profile' }
  ];

  const infoSettings = [
    { key: 'showEmail', label: 'Show Email', description: 'Display email on profile' },
    { key: 'showPhone', label: 'Show Phone', description: 'Display phone number on profile' },
    { key: 'showLocation', label: 'Show Location', description: 'Display location on profile' },
    { key: 'showActivity', label: 'Show Recent Activity', description: 'Display recent activity to others' }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-purple-50">
      {/* Header */}
      <header className="bg-white/80 backdrop-blur-md shadow-lg">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <button
              onClick={() => navigate('/profile')}
              className="flex items-center space-x-2 text-gray-600 hover:text-gray-800 transition-colors"
            >
              <span className="text-xl">←</span>
              <span>Back to Profile</span>
            </button>
            
            <motion.button
              onClick={() => navigate('/')}
              className="cursor-pointer"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              <Logo width="140" height="45" className="hover:opacity-90 transition-opacity" />
            </motion.button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <h1 className="text-4xl font-bold text-gray-800 mb-2">⚙️ Settings</h1>
          <p className="text-gray-600">Manage your account preferences and privacy settings</p>
        </motion.div>

        <div className="grid gap-8">
          {/* Account Settings */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="bg-white rounded-2xl shadow-lg p-8"
          >
            <h2 className="text-2xl font-bold text-gray-800 mb-6">🔧 Account Settings</h2>
            <div className="space-y-6">
              <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                <div>
                  <h4 className="font-medium text-gray-800">Language</h4>
                  <p className="text-sm text-gray-600">Choose your preferred language</p>
                </div>
                <select
                  value={accountSettings.language}
                  onChange={(e) => handleAccountUpdate('language', e.target.value)}
                  className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                >
                  <option value="english">🇺🇸 English</option>
                  <option value="hindi">🇮🇳 हिन्दी</option>
                </select>
              </div>

              <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                <div>
                  <h4 className="font-medium text-gray-800">Currency</h4>
                  <p className="text-sm text-gray-600">Display prices in your preferred currency</p>
                </div>
                <select
                  value={accountSettings.currency}
                  onChange={(e) => handleAccountUpdate('currency', e.target.value)}
                  className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                >
                  <option value="INR">₹ Indian Rupee</option>
                  <option value="USD">$ US Dollar</option>
                  <option value="EUR">€ Euro</option>
                </select>
              </div>

              <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                <div>
                  <h4 className="font-medium text-gray-800">Theme</h4>
                  <p className="text-sm text-gray-600">Choose your display theme</p>
                </div>
                <select
                  value={accountSettings.theme}
                  onChange={(e) => handleAccountUpdate('theme', e.target.value)}
                  className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                >
                  <option value="light">☀️ Light</option>
                  <option value="dark">🌙 Dark</option>
                  <option value="auto">⚡ Auto</option>
                </select>
              </div>
            </div>
          </motion.div>

          {/* Notification Settings */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="bg-white rounded-2xl shadow-lg p-8"
          >
            <h2 className="text-2xl font-bold text-gray-800 mb-6">🔔 Notifications</h2>
            <div className="space-y-4">
              {[
                { key: 'emailNotifications', label: 'Email Notifications', description: 'Receive updates via email' },
                { key: 'pushNotifications', label: 'Push Notifications', description: 'Get browser notifications' },
                { key: 'orderUpdates', label: 'Order Updates', description: 'Notifications about your orders' },
                { key: 'marketingEmails', label: 'Marketing Emails', description: 'Promotional content and offers' }
              ].map((setting) => (
                <div key={setting.key} className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                  <div>
                    <h4 className="font-medium text-gray-800">{setting.label}</h4>
                    <p className="text-sm text-gray-600">{setting.description}</p>
                  </div>
                  <label className="relative inline-flex items-center cursor-pointer">
                    <input
                      type="checkbox"
                      checked={notifications[setting.key]}
                      onChange={(e) => handleNotificationUpdate(setting.key, e.target.checked)}
                      className="sr-only peer"
                    />
                    <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
                  </label>
                </div>
              ))}
            </div>
          </motion.div>

          {/* Privacy Settings */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
            className="bg-white rounded-2xl shadow-lg p-8"
          >
            <h2 className="text-2xl font-bold text-gray-800 mb-6">🔒 Privacy & Security</h2>
            <div className="space-y-4">
              <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                <div>
                  <h4 className="font-medium text-gray-800">Profile Visibility</h4>
                  <p className="text-sm text-gray-600">Control who can see your profile</p>
                </div>
                <button 
                  onClick={() => setShowPrivacyModal(true)}
                  className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors"
                >
                  Manage
                </button>
              </div>

              <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                <div>
                  <h4 className="font-medium text-gray-800">Two-Factor Authentication</h4>
                  <p className="text-sm text-gray-600">Add extra security to your account</p>
                </div>
                <button className="px-4 py-2 bg-green-500 text-white rounded-lg hover:bg-green-600 transition-colors">
                  Enable
                </button>
              </div>

              <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                <div>
                  <h4 className="font-medium text-gray-800">Data Export</h4>
                  <p className="text-sm text-gray-600">Download your account data</p>
                </div>
                <button className="px-4 py-2 bg-gray-500 text-white rounded-lg hover:bg-gray-600 transition-colors">
                  Download
                </button>
              </div>
            </div>
          </motion.div>

          {/* Save Button */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4 }}
            className="text-center"
          >
            <button
              onClick={saveAllSettings}
              className="px-8 py-4 bg-gradient-to-r from-blue-500 to-purple-500 text-white rounded-xl font-semibold text-lg hover:from-blue-600 hover:to-purple-600 transition-all duration-200 shadow-lg hover:shadow-xl"
            >
              💾 Save All Settings
            </button>
          </motion.div>
        </div>
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

              {/* Profile Visibility */}
              <div className="mb-6">
                <h3 className="text-lg font-semibold text-gray-800 mb-3">Profile Visibility</h3>
                <div className="space-y-3">
                  {privacyOptions.map((option) => (
                    <label key={option.value} className="flex items-start space-x-3 p-3 border rounded-lg cursor-pointer hover:bg-gray-50">
                      <input
                        type="radio"
                        name="profileVisibility"
                        value={option.value}
                        checked={privacySettings.profileVisibility === option.value}
                        onChange={() => handlePrivacyUpdate('profileVisibility', option.value)}
                        className="mt-1"
                      />
                      <div>
                        <div className="font-medium text-gray-800">{option.label}</div>
                        <div className="text-sm text-gray-600">{option.description}</div>
                      </div>
                    </label>
                  ))}
                </div>
              </div>

              {/* Information Control */}
              <div className="mb-6">
                <h3 className="text-lg font-semibold text-gray-800 mb-3">Information Control</h3>
                <div className="space-y-3">
                  {infoSettings.map((setting) => (
                    <div key={setting.key} className="flex items-center justify-between p-3 border rounded-lg">
                      <div>
                        <div className="font-medium text-gray-800">{setting.label}</div>
                        <div className="text-sm text-gray-600">{setting.description}</div>
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
                    <div className="text-sm text-gray-600">Who can send you messages</div>
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

export default SettingsPage;