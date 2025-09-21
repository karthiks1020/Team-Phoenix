const getLanguage = () => {
  try {
    const attr = document.documentElement.getAttribute('data-language');
    if (attr) return attr;
    const raw = localStorage.getItem('accountSettings');
    if (raw) {
      const { language } = JSON.parse(raw);
      return language || 'english';
    }
  } catch {}
  return 'english';
};

const dict = {
  english: {
    home: {
      sell: 'Sell',
      addToCart: 'Add to Cart',
      chatbot: 'My Artist Friend',
      about: 'About',
      settings: 'Settings',
      searchPlaceholder: 'Search for handcrafted treasures...'
    },
    common: {
      backToHome: 'Back to Home',
      save: 'Save',
      saveAll: 'Save All Settings',
      manage: 'Manage',
      enable: 'Enable',
      download: 'Download'
    },
    settings: {
      title: 'Settings',
      subtitle: 'Manage your account preferences and privacy settings',
      language: 'Language',
      chooseLanguage: 'Choose your preferred language',
      currency: 'Currency',
      currencyDesc: 'Display prices in your preferred currency',
      theme: 'Theme',
      themeDesc: 'Choose your display theme',
      notifications: 'Notifications',
      emailNotifications: 'Email Notifications',
      pushNotifications: 'Push Notifications',
      orderUpdates: 'Order Updates',
      marketingEmails: 'Marketing Emails',
      privacyTitle: 'Privacy & Security',
      profileVisibility: 'Profile Visibility',
      twoFactor: 'Two-Factor Authentication',
      addSecurity: 'Add extra security to your account',
      dataExport: 'Data Export',
      downloadData: 'Download your account data'
    },
    profile: {
      myProfile: 'My Profile',
      backToHome: 'Back to Home',
      logout: 'Logout',
      editProfile: 'Edit Profile',
      profileInformation: 'Profile Information',
      fullName: 'Full Name',
      email: 'Email',
      phone: 'Phone',
      location: 'Location',
      saveChanges: 'Save Changes'
    }
  },
  hindi: {
    home: {
      sell: 'बेचें',
      addToCart: 'कार्ट में जोड़ें',
      chatbot: 'मेरा कलाकार दोस्त',
      about: 'परिचय',
      settings: 'सेटिंग्स',
      searchPlaceholder: 'हस्तनिर्मित खजाने खोजें...'
    },
    common: {
      backToHome: 'होम पर वापस',
      save: 'सहेजें',
      saveAll: 'सभी सेटिंग्स सहेजें',
      manage: 'प्रबंधित करें',
      enable: 'सक्रिय करें',
      download: 'डाउनलोड'
    },
    settings: {
      title: 'सेटिंग्स',
      subtitle: 'अपनी खाता प्राथमिकताएँ और गोपनीयता सेटिंग्स प्रबंधित करें',
      language: 'भाषा',
      chooseLanguage: 'अपनी पसंदीदा भाषा चुनें',
      currency: 'मुद्रा',
      currencyDesc: 'अपनी पसंदीदा मुद्रा में मूल्य दिखाएँ',
      theme: 'थीम',
      themeDesc: 'अपनी प्रदर्शन थीम चुनें',
      notifications: 'सूचनाएँ',
      emailNotifications: 'ईमेल सूचनाएँ',
      pushNotifications: 'पुश सूचनाएँ',
      orderUpdates: 'ऑर्डर अपडेट',
      marketingEmails: 'मार्केटिंग ईमेल',
      privacyTitle: 'गोपनीयता और सुरक्षा',
      profileVisibility: 'प्रोफ़ाइल दृश्यता',
      twoFactor: 'दो-कारक प्रमाणीकरण',
      addSecurity: 'अपने खाते में अतिरिक्त सुरक्षा जोड़ें',
      dataExport: 'डेटा निर्यात',
      downloadData: 'अपना खाता डेटा डाउनलोड करें'
    },
    profile: {
      myProfile: 'मेरा प्रोफ़ाइल',
      backToHome: 'होम पर वापस',
      logout: 'लॉगआउट',
      editProfile: 'प्रोफ़ाइल संपादित करें',
      profileInformation: 'प्रोफ़ाइल जानकारी',
      fullName: 'पूरा नाम',
      email: 'ईमेल',
      phone: 'फ़ोन',
      location: 'स्थान',
      saveChanges: 'परिवर्तन सहेजें'
    }
  }
};

export const t = (path) => {
  const lang = getLanguage();
  const parts = path.split('.');
  let node = dict[lang] || dict.english;
  for (const p of parts) {
    if (node && p in node) node = node[p];
    else return path; // fallback: key string
  }
  return typeof node === 'string' ? node : path;
};

const api = { t, getLanguage };
export default api;
