import React, { useState, useEffect, useRef } from 'react';
import {
  Box,
  Container,
  Typography,
  Button,
  Paper,
  IconButton,
  Slider,
  Switch,
  FormControlLabel,
  Alert,
  Fab,
  Dialog,
  DialogTitle,
  DialogContent,
  Grid,
  Card,
  CardContent,
  Chip
} from '@mui/material';
import {
  CameraAlt as CameraIcon,
  Flip as FlipIcon,
  RotateRight as RotateIcon,
  ZoomIn as ZoomInIcon,
  ZoomOut as ZoomOutIcon,
  Share as ShareIcon,
  Save as SaveIcon,
  Settings as SettingsIcon,
  ViewInAr as ArIcon,
  Close as CloseIcon,
  PhotoCamera as PhotoIcon,
  VideoCall as VideoIcon,
  ThreeDRotation as ThreeDIcon
} from '@mui/icons-material';
import { motion, AnimatePresence } from 'framer-motion';
import { useParams, useNavigate } from 'react-router-dom';
import { formatINRPrice } from '../utils/indianLocalization';

const ARViewerPage = () => {
  const { productId } = useParams();
  const navigate = useNavigate();
  
  // State management
  const [product, setProduct] = useState(null);
  const [arSupported, setArSupported] = useState(false);
  const [arActive, setArActive] = useState(false);
  const [cameraPermission, setCameraPermission] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [settingsOpen, setSettingsOpen] = useState(false);
  
  // AR Controls
  const [scale, setScale] = useState(1);
  const [rotation, setRotation] = useState(0);
  const [position, setPosition] = useState({ x: 0, y: 0 });
  const [lightingEnabled, setLightingEnabled] = useState(true);
  const [shadowsEnabled, setShadowsEnabled] = useState(true);
  
  // Refs
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);

  useEffect(() => {
    initializeAR();
    loadProductData();
    
    return () => {
      // Cleanup camera stream
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
    };
  }, [productId]);

  const initializeAR = async () => {
    try {
      // Check for WebXR support
      if ('xr' in navigator) {
        const isSupported = await navigator.xr.isSessionSupported('immersive-ar');
        setArSupported(isSupported);
      }
      
      // Fallback to camera for AR simulation
      if ('mediaDevices' in navigator && 'getUserMedia' in navigator.mediaDevices) {
        setArSupported(true);
      }
      
      setLoading(false);
    } catch (err) {
      console.error('AR initialization error:', err);
      setError('AR functionality is not supported on this device');
      setLoading(false);
    }
  };

  const loadProductData = async () => {
    try {
      const response = await fetch(`/api/products/${productId}`);
      if (response.ok) {
        const data = await response.json();
        setProduct(data.product);
      } else {
        setError('Product not found');
      }
    } catch (err) {
      setError('Failed to load product data');
    }
  };

  const startARSession = async () => {
    try {
      // Request camera permission
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { 
          facingMode: 'environment', // Use back camera
          width: { ideal: 1280 },
          height: { ideal: 720 }
        } 
      });
      
      streamRef.current = stream;
      setCameraPermission(true);
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
      }
      
      setArActive(true);
    } catch (err) {
      console.error('Camera access error:', err);
      setError('Camera access is required for AR functionality');
    }
  };

  const stopARSession = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
    }
    setArActive(false);
    setCameraPermission(false);
  };

  const capturePhoto = () => {
    if (videoRef.current && canvasRef.current) {
      const canvas = canvasRef.current;
      const video = videoRef.current;
      
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      
      const ctx = canvas.getContext('2d');
      
      // Draw video frame
      ctx.drawImage(video, 0, 0);
      
      // Draw AR overlay (product visualization)
      drawAROverlay(ctx);
      
      // Convert to blob and save
      canvas.toBlob(blob => {
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `ar-preview-${product?.title || 'product'}.jpg`;
        a.click();
      }, 'image/jpeg', 0.9);
    }
  };

  const drawAROverlay = (ctx) => {
    // Simple AR simulation - draw product representation
    const centerX = ctx.canvas.width / 2 + position.x;
    const centerY = ctx.canvas.height / 2 + position.y;
    const size = 100 * scale;
    
    ctx.save();
    ctx.translate(centerX, centerY);
    ctx.rotate((rotation * Math.PI) / 180);
    
    // Draw product placeholder (simplified)
    if (product?.category === 'pottery') {
      drawPotteryAR(ctx, size);
    } else if (product?.category === 'wooden_dolls') {
      drawWoodenDollAR(ctx, size);
    } else if (product?.category === 'basket_weaving') {
      drawBasketAR(ctx, size);
    } else if (product?.category === 'handlooms') {
      drawTextileAR(ctx, size);
    }
    
    ctx.restore();
  };

  const drawPotteryAR = (ctx, size) => {
    // Draw pottery silhouette
    ctx.fillStyle = shadowsEnabled ? 'rgba(139, 69, 19, 0.8)' : 'rgba(139, 69, 19, 1)';
    ctx.strokeStyle = '#8B4513';
    ctx.lineWidth = 3;
    
    // Draw pottery shape
    ctx.beginPath();
    ctx.ellipse(0, size/4, size/2, size/3, 0, 0, 2 * Math.PI);
    ctx.fill();
    ctx.stroke();
    
    // Draw pottery neck
    ctx.fillRect(-size/6, -size/2, size/3, size/2);
    ctx.strokeRect(-size/6, -size/2, size/3, size/2);
  };

  const drawWoodenDollAR = (ctx, size) => {
    // Draw wooden doll silhouette
    ctx.fillStyle = shadowsEnabled ? 'rgba(160, 82, 45, 0.8)' : 'rgba(160, 82, 45, 1)';
    ctx.strokeStyle = '#A0522D';
    ctx.lineWidth = 3;
    
    // Head
    ctx.beginPath();
    ctx.arc(0, -size/3, size/4, 0, 2 * Math.PI);
    ctx.fill();
    ctx.stroke();
    
    // Body
    ctx.fillRect(-size/3, -size/6, size/1.5, size);
    ctx.strokeRect(-size/3, -size/6, size/1.5, size);
  };

  const drawBasketAR = (ctx, size) => {
    // Draw basket silhouette
    ctx.fillStyle = shadowsEnabled ? 'rgba(218, 165, 32, 0.8)' : 'rgba(218, 165, 32, 1)';
    ctx.strokeStyle = '#DAA520';
    ctx.lineWidth = 3;
    
    // Draw basket shape (trapezoid)
    ctx.beginPath();
    ctx.moveTo(-size/2, size/2);
    ctx.lineTo(-size/3, -size/2);
    ctx.lineTo(size/3, -size/2);
    ctx.lineTo(size/2, size/2);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
    
    // Draw weave pattern
    for (let i = -size/2; i < size/2; i += 10) {
      ctx.beginPath();
      ctx.moveTo(i, -size/2);
      ctx.lineTo(i, size/2);
      ctx.stroke();
    }
  };

  const drawTextileAR = (ctx, size) => {
    // Draw textile silhouette
    ctx.fillStyle = shadowsEnabled ? 'rgba(220, 20, 60, 0.8)' : 'rgba(220, 20, 60, 1)';
    ctx.strokeStyle = '#DC143C';
    ctx.lineWidth = 3;
    
    // Draw fabric rectangle
    ctx.fillRect(-size/2, -size/3, size, size/1.5);
    ctx.strokeRect(-size/2, -size/3, size, size/1.5);
    
    // Draw pattern
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.5)';
    for (let i = 0; i < size; i += 15) {
      ctx.beginPath();
      ctx.moveTo(-size/2, -size/3 + i);
      ctx.lineTo(size/2, -size/3 + i);
      ctx.stroke();
    }
  };

  const shareARExperience = () => {
    if (navigator.share) {
      navigator.share({
        title: `AR Preview: ${product?.title}`,
        text: `Check out this ${product?.category} in AR!`,
        url: window.location.href
      });
    } else {
      // Fallback - copy link
      navigator.clipboard.writeText(window.location.href);
      // Show notification
    }
  };

  if (loading) {
    return (
      <Container maxWidth="lg" sx={{ py: 4, textAlign: 'center' }}>
        <Typography variant="h6">Loading AR Viewer...</Typography>
      </Container>
    );
  }

  if (error) {
    return (
      <Container maxWidth="lg" sx={{ py: 4 }}>
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
        <Button onClick={() => navigate(-1)}>Go Back</Button>
      </Container>
    );
  }

  return (
    <Box sx={{ height: '100vh', bgcolor: 'black', position: 'relative', overflow: 'hidden' }}>
      {/* Header */}
      <Box
        sx={{
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          zIndex: 10,
          background: 'linear-gradient(180deg, rgba(0,0,0,0.8) 0%, transparent 100%)',
          p: 2
        }}
      >
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', color: 'white' }}>
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            <IconButton onClick={() => navigate(-1)} sx={{ color: 'white', mr: 2 }}>
              <CloseIcon />
            </IconButton>
            <Box>
              <Typography variant="h6">{product?.title}</Typography>
              <Typography variant="body2" sx={{ opacity: 0.8 }}>
                AR Preview Mode
              </Typography>
            </Box>
          </Box>
          <IconButton onClick={() => setSettingsOpen(true)} sx={{ color: 'white' }}>
            <SettingsIcon />
          </IconButton>
        </Box>
      </Box>

      {/* AR Camera View */}
      <Box sx={{ width: '100%', height: '100%', position: 'relative' }}>
        {arActive ? (
          <>
            <video
              ref={videoRef}
              style={{
                width: '100%',
                height: '100%',
                objectFit: 'cover'
              }}
              playsInline
              muted
            />
            
            {/* AR Controls Overlay */}
            <Box
              sx={{
                position: 'absolute',
                bottom: 20,
                left: '50%',
                transform: 'translateX(-50%)',
                display: 'flex',
                gap: 2,
                alignItems: 'center',
                bgcolor: 'rgba(0,0,0,0.7)',
                borderRadius: 3,
                p: 2
              }}
            >
              <IconButton
                onClick={capturePhoto}
                sx={{
                  bgcolor: 'white',
                  color: 'black',
                  '&:hover': { bgcolor: 'grey.200' }
                }}
              >
                <PhotoIcon />
              </IconButton>
              
              <IconButton onClick={() => setRotation(r => r + 90)} sx={{ color: 'white' }}>
                <RotateIcon />
              </IconButton>
              
              <IconButton onClick={() => setScale(s => Math.max(0.5, s - 0.1))} sx={{ color: 'white' }}>
                <ZoomOutIcon />
              </IconButton>
              
              <IconButton onClick={() => setScale(s => Math.min(2, s + 0.1))} sx={{ color: 'white' }}>
                <ZoomInIcon />
              </IconButton>
              
              <IconButton onClick={shareARExperience} sx={{ color: 'white' }}>
                <ShareIcon />
              </IconButton>
            </Box>

            {/* Position Controls */}
            <Box
              sx={{
                position: 'absolute',
                right: 20,
                top: '50%',
                transform: 'translateY(-50%)',
                display: 'flex',
                flexDirection: 'column',
                gap: 1,
                bgcolor: 'rgba(0,0,0,0.7)',
                borderRadius: 2,
                p: 1
              }}
            >
              <Typography variant="caption" sx={{ color: 'white', textAlign: 'center' }}>
                Position
              </Typography>
              <Box
                sx={{
                  width: 80,
                  height: 80,
                  bgcolor: 'rgba(255,255,255,0.2)',
                  borderRadius: 1,
                  position: 'relative',
                  cursor: 'pointer'
                }}
                onClick={(e) => {
                  const rect = e.currentTarget.getBoundingClientRect();
                  const x = ((e.clientX - rect.left - rect.width / 2) / rect.width) * 200;
                  const y = ((e.clientY - rect.top - rect.height / 2) / rect.height) * 200;
                  setPosition({ x, y });
                }}
              >
                <Box
                  sx={{
                    width: 8,
                    height: 8,
                    bgcolor: 'white',
                    borderRadius: '50%',
                    position: 'absolute',
                    left: '50%',
                    top: '50%',
                    transform: `translate(calc(-50% + ${position.x/5}px), calc(-50% + ${position.y/5}px))`
                  }}
                />
              </Box>
            </Box>
          </>
        ) : (
          <Box
            sx={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              height: '100%',
              textAlign: 'center',
              color: 'white',
              p: 4
            }}
          >
            <motion.div
              initial={{ scale: 0.8, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{ duration: 0.6 }}
            >
              <ArIcon sx={{ fontSize: 100, mb: 3 }} />
              <Typography variant="h4" sx={{ mb: 2, fontWeight: 600 }}>
                AR Preview
              </Typography>
              <Typography variant="h6" sx={{ mb: 4, opacity: 0.8 }}>
                See how {product?.title} looks in your space
              </Typography>
              
              {arSupported ? (
                <Button
                  variant="contained"
                  size="large"
                  startIcon={<CameraIcon />}
                  onClick={startARSession}
                  sx={{
                    py: 2,
                    px: 4,
                    fontSize: '1.1rem',
                    borderRadius: 3
                  }}
                >
                  Start AR Experience
                </Button>
              ) : (
                <Alert severity="warning" sx={{ mt: 2 }}>
                  AR is not supported on this device or browser
                </Alert>
              )}
            </motion.div>
          </Box>
        )}
      </Box>

      {/* Hidden Canvas for Screenshots */}
      <canvas ref={canvasRef} style={{ display: 'none' }} />

      {/* Settings Dialog */}
      <Dialog open={settingsOpen} onClose={() => setSettingsOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>AR Settings</DialogTitle>
        <DialogContent>
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <Typography gutterBottom>Scale</Typography>
              <Slider
                value={scale}
                onChange={(e, value) => setScale(value)}
                min={0.5}
                max={2}
                step={0.1}
                marks
                valueLabelDisplay="auto"
              />
            </Grid>
            
            <Grid item xs={12}>
              <Typography gutterBottom>Rotation</Typography>
              <Slider
                value={rotation}
                onChange={(e, value) => setRotation(value)}
                min={0}
                max={360}
                step={15}
                marks
                valueLabelDisplay="auto"
              />
            </Grid>
            
            <Grid item xs={6}>
              <FormControlLabel
                control={
                  <Switch
                    checked={lightingEnabled}
                    onChange={(e) => setLightingEnabled(e.target.checked)}
                  />
                }
                label="Lighting Effects"
              />
            </Grid>
            
            <Grid item xs={6}>
              <FormControlLabel
                control={
                  <Switch
                    checked={shadowsEnabled}
                    onChange={(e) => setShadowsEnabled(e.target.checked)}
                  />
                }
                label="Shadows"
              />
            </Grid>

            <Grid item xs={12}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Product Information
                  </Typography>
                  <Typography variant="body2" paragraph>
                    {product?.description}
                  </Typography>
                  <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                    <Chip label={product?.category} size="small" />
                    <Chip label={formatINRPrice(product?.price, false)} size="small" color="primary" />
                    {product?.ai_tags?.map((tag, index) => (
                      <Chip key={index} label={tag} size="small" variant="outlined" />
                    ))}
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </DialogContent>
      </Dialog>

      {/* Feature Info Fab */}
      <Fab
        sx={{
          position: 'absolute',
          bottom: 100,
          left: 20,
          bgcolor: 'primary.main',
          color: 'white',
          '&:hover': { bgcolor: 'primary.dark' }
        }}
        onClick={() => {
          // Show AR feature info
          alert('AR Features:\n• Rotate and scale products\n• Capture screenshots\n• Realistic lighting\n• Share AR experiences');
        }}
      >
        <ThreeDIcon />
      </Fab>
    </Box>
  );
};

export default ARViewerPage;