// Authentication utilities for Artisans Hub

/**
 * Check if user is authenticated
 * @returns {boolean} - True if user is authenticated
 */
export const isAuthenticated = () => {
  try {
    const authData = localStorage.getItem('artisansHubAuth');
    if (!authData) return false;
    
    const auth = JSON.parse(authData);
    const now = new Date().getTime();
    const authTime = auth.timestamp;
    
    // Session expires after 24 hours (86400000 ms)
    const sessionDuration = 24 * 60 * 60 * 1000;
    
    if (now - authTime > sessionDuration) {
      localStorage.removeItem('artisansHubAuth');
      return false;
    }
    
    return auth.isAuthenticated === true;
  } catch (error) {
    console.error('Error checking authentication:', error);
    return false;
  }
};

/**
 * Get current user information
 * @returns {object|null} - User object or null if not authenticated
 */
export const getCurrentUser = () => {
  try {
    if (!isAuthenticated()) return null;
    
    const authData = localStorage.getItem('artisansHubAuth');
    const auth = JSON.parse(authData);
    return auth.user;
  } catch (error) {
    console.error('Error getting current user:', error);
    return null;
  }
};

/**
 * Log out the current user
 */
export const logout = () => {
  localStorage.removeItem('artisansHubAuth');
  localStorage.removeItem('intendedPage');
};

/**
 * Redirect to auth page if not authenticated
 * @param {string} intendedPage - Page user was trying to access
 * @returns {boolean} - True if user is authenticated, false if redirected
 */
export const requireAuth = (intendedPage = '/') => {
  if (!isAuthenticated()) {
    // Store the intended page for redirect after login
    localStorage.setItem('intendedPage', intendedPage);
    return false;
  }
  return true;
};

/**
 * Get user greeting with name
 * @returns {string} - Personalized greeting
 */
export const getUserGreeting = () => {
  const user = getCurrentUser();
  if (!user) return 'Welcome, Guest!';
  
  const hour = new Date().getHours();
  let timeGreeting = 'Hello';
  
  if (hour < 12) {
    timeGreeting = 'Good Morning';
  } else if (hour < 17) {
    timeGreeting = 'Good Afternoon';
  } else {
    timeGreeting = 'Good Evening';
  }
  
  const firstName = user.name.split(' ')[0];
  return `${timeGreeting}, ${firstName}!`;
};

export default {
  isAuthenticated,
  getCurrentUser,
  logout,
  requireAuth,
  getUserGreeting
};