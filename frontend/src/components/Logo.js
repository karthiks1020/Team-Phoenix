import React, { useState } from 'react';

const Logo = ({ 
  width = "120", 
  height = "40", 
  className = "",
  variant = "default" // "default", "white", "dark"
}) => {
  const [imageError, setImageError] = useState(false);

  const getStyles = () => {
    const baseStyle = {
      width: width + 'px',
      height: height + 'px',
      objectFit: 'contain'
    };

    switch (variant) {
      case 'white':
        return {
          ...baseStyle,
          filter: 'brightness(0) saturate(100%) invert(27%) sepia(51%) saturate(2878%) hue-rotate(346deg) brightness(104%) contrast(97%)'
        };
      case 'dark':
        return {
          ...baseStyle,
          filter: 'brightness(0) saturate(100%) invert(0%) sepia(0%) saturate(0%) hue-rotate(0deg) brightness(100%) contrast(100%)'
        };
      default:
        return baseStyle;
    }
  };

  const handleImageError = () => {
    console.warn('Logo image not found, using text fallback');
    setImageError(true);
  };

  // If image failed to load, show text logo as fallback
  if (imageError) {
    return (
      <div className={`logo-fallback ${className}`} style={{
        width: width + 'px',
        height: height + 'px',
        display: 'flex',
        alignItems: 'center',
        fontSize: '18px',
        fontWeight: 'bold',
        background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
        color: 'white',
        padding: '8px 12px',
        borderRadius: '8px',
        boxShadow: '0 2px 10px rgba(102, 126, 234, 0.3)'
      }}>
        🎨 Artisans Hub
      </div>
    );
  }

  return (
    <img 
      src="/logo.svg" 
      alt="Artisans Hub Logo" 
      style={getStyles()}
      className={`logo ${className}`}
      onError={handleImageError}
    />
  );
};

export default Logo;