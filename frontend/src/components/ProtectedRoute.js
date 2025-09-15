import React, { useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { isAuthenticated } from '../utils/auth';

const ProtectedRoute = ({ children }) => {
  const navigate = useNavigate();
  const location = useLocation();

  useEffect(() => {
    if (!isAuthenticated()) {
      // Store the current page as the intended destination
      localStorage.setItem('intendedPage', location.pathname + location.search);
      // Redirect to auth page
      navigate('/auth');
    }
  }, [navigate, location]);

  // Only render children if authenticated
  if (!isAuthenticated()) {
    return null;
  }

  return children;
};

export default ProtectedRoute;