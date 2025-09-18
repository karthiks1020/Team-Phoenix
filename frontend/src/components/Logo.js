
import React, { useState, useEffect } from 'react';

const Logo = ({ 
  width = "120", 
  height = "40", 
  className = "",
  variant = "default"
}) => {
  const [imageError, setImageError] = useState(false);

  // FIX: Use process.env.PUBLIC_URL for reliable path to assets in the public folder
  const logoJpg = `${process.env.PUBLIC_URL}/logo.jpg`;
  const logoPng = `${process.env.PUBLIC_URL}/logo.png`;

  const [logoSrc, setLogoSrc] = useState(logoJpg);

  useEffect(() => {
    // Reset to JPG when component re-renders, e.g., on page navigation
    setLogoSrc(logoJpg);
    setImageError(false);
  }, [logoJpg]);

  const handleImageError = () => {
    if (logoSrc.includes('logo.jpg')) {
      // If JPG fails, try PNG as a fallback
      console.warn('Logo JPG not found, trying PNG fallback');
      setLogoSrc(logoPng);
    } else {
      // If PNG also fails, show the text fallback
      console.error('All logo images not found, using text fallback.');
      setImageError(true);
    }
  };

  if (imageError) {
    return (
      <div className={`logo-fallback ${className}`} style={{
        width: width + 'px',
        height: height + 'px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
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
      src={logoSrc}
      alt="Artisans Hub Logo" 
      style={{ width: `${width}px`, height: `${height}px`, objectFit: 'contain' }}
      className={`logo ${className}`}
      onError={handleImageError}
    />
  );
};

export default Logo;
