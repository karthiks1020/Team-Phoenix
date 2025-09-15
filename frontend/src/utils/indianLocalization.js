// Indian Localization Utilities
// Currency conversion and formatting for Indian users

export const CURRENCY_RATE_USD_TO_INR = 83.5; // Approximate rate as of 2025

/**
 * Convert USD to Indian Rupees
 * @param {number} usdAmount - Amount in USD
 * @returns {number} - Amount in INR
 */
export const convertToINR = (usdAmount) => {
  return Math.round(usdAmount * CURRENCY_RATE_USD_TO_INR);
};

/**
 * Format price in Indian Rupees with proper Indian formatting
 * @param {number} amount - Amount to format (can be USD or INR)
 * @param {boolean} convertFromUSD - Whether to convert from USD first
 * @returns {string} - Formatted price string
 */
export const formatINRPrice = (amount, convertFromUSD = true) => {
  const inrAmount = convertFromUSD ? convertToINR(amount) : amount;
  
  // Indian number formatting with proper comma placement
  const formatter = new Intl.NumberFormat('en-IN', {
    style: 'currency',
    currency: 'INR',
    minimumFractionDigits: 0,
    maximumFractionDigits: 0
  });
  
  // Alternative format with "Rs" suffix for better Indian UX
  const formattedNumber = new Intl.NumberFormat('en-IN').format(inrAmount);
  return `₹${formattedNumber}`;
};

/**
 * Format price in simple INR format (e.g., "299Rs")
 * @param {number} amount - Amount to format
 * @param {boolean} convertFromUSD - Whether to convert from USD first
 * @returns {string} - Simple formatted price
 */
export const formatSimpleINR = (amount, convertFromUSD = true) => {
  const inrAmount = convertFromUSD ? convertToINR(amount) : amount;
  return `${inrAmount}Rs`;
};

/**
 * Indian design color schemes and gradients
 */
export const INDIAN_COLORS = {
  saffron: 'from-orange-500 to-red-500',
  green: 'from-green-600 to-emerald-600', 
  navy: 'from-blue-800 to-indigo-900',
  heritage: 'from-amber-600 to-orange-700',
  royal: 'from-purple-700 to-pink-600',
  earth: 'from-yellow-700 to-red-800',
  peacock: 'from-teal-600 to-blue-700'
};

/**
 * Indian cultural icons and emojis
 */
export const INDIAN_ICONS = {
  namaste: '🙏',
  lotus: '🪷',
  om: 'ॐ',
  diya: '🪔',
  temple: '🛕',
  elephant: '🐘',
  peacock: '🦚',
  tabla: '🥁',
  sitar: '🪕'
};

/**
 * Get greeting based on time of day
 */
export const getIndianGreeting = () => {
  const hour = new Date().getHours();
  if (hour < 12) {
    return { en: 'Good Morning', emoji: '🌅' };
  } else if (hour < 17) {
    return { en: 'Good Afternoon', emoji: '☀️' };
  } else {
    return { en: 'Good Evening', emoji: '🌆' };
  }
};

/**
 * Indian state names and their craft specialties
 */
export const INDIAN_CRAFT_REGIONS = {
  'Rajasthan': ['Wooden Dolls', 'Pottery', 'Handlooms'],
  'Gujarat': ['Handlooms', 'Basket Weaving'],
  'Uttar Pradesh': ['Wooden Dolls', 'Pottery'],
  'Tamil Nadu': ['Handlooms', 'Pottery'],
  'Punjab': ['Handlooms', 'Basket Weaving'],
  'Maharashtra': ['Pottery', 'Handlooms'],
  'West Bengal': ['Handlooms', 'Pottery'],
  'Odisha': ['Handlooms', 'Wooden Dolls'],
  'Karnataka': ['Pottery', 'Basket Weaving']
};

export default {
  convertToINR,
  formatINRPrice,
  formatSimpleINR,
  INDIAN_COLORS,
  INDIAN_ICONS,
  getIndianGreeting,
  INDIAN_CRAFT_REGIONS
};