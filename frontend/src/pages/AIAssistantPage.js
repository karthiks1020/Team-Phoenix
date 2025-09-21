import React, { useState, useEffect, useRef } from 'react';
import {
  Box,
  Container,
  Typography,
  Paper,
  TextField,
  Button,
  Grid,
  Card,
  CardContent,
  CardMedia,
  Avatar,
  Chip,
  Divider,
  IconButton,
  LinearProgress,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Alert,
  Fab,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Tabs,
  Tab
} from '@mui/material';
import {
  Send as SendIcon,
  PhotoCamera as PhotoIcon,
  Mic as MicIcon,
  ExpandMore as ExpandMoreIcon,
  Psychology as AIIcon,
  CameraAlt as CameraIcon,
  Upload as UploadIcon,
  SmartToy as BotIcon,
  Translate as TranslateIcon,
  VolumeUp as VolumeIcon,
  Close as CloseIcon,
  Help as HelpIcon,
  AutoFixHigh as MagicIcon
} from '@mui/icons-material';
import { motion, AnimatePresence } from 'framer-motion';
import { useDropzone } from 'react-dropzone';
import { formatINRPrice } from '../utils/indianLocalization';

const AIAssistantPage = () => {
  // State management
  const [activeTab, setActiveTab] = useState(0);
  const [chatMessages, setChatMessages] = useState([]);
  const [messageInput, setMessageInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [imageAnalysis, setImageAnalysis] = useState(null);
  const [voiceInput, setVoiceInput] = useState('');
  const [isListening, setIsListening] = useState(false);
  const [analysisResults, setAnalysisResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [helpDialogOpen, setHelpDialogOpen] = useState(false);
  const [language, setLanguage] = useState('en');

  // Refs
  const messagesEndRef = useRef(null);
  const recognitionRef = useRef(null);

  // Initialize AI Assistant
  useEffect(() => {
    // Add welcome message
    setChatMessages([
      {
        id: 1,
        type: 'bot',
        message: "👋 Hello! I'm your AI-powered marketplace assistant. I can help you:",
        timestamp: new Date(),
        suggestions: [
          "Identify handicrafts from photos",
          "Find authentic artisan products", 
          "Get cultural heritage information",
          "Translate product descriptions",
          "Provide sustainability insights"
        ]
      }
    ]);

    // Initialize speech recognition
    if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
      const SpeechRecognition = window.webkitSpeechRecognition || window.SpeechRecognition;
      recognitionRef.current = new SpeechRecognition();
      recognitionRef.current.continuous = false;
      recognitionRef.current.interimResults = false;
      recognitionRef.current.lang = 'en-US';
    }
  }, []);

  // Auto-scroll to bottom of messages
  useEffect(() => {
    scrollToBottom();
  }, [chatMessages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  // Handle message sending
  const sendMessage = async (message = messageInput) => {
    if (!message.trim()) return;

    const userMessage = {
      id: Date.now(),
      type: 'user',
      message: message,
      timestamp: new Date()
    };

    setChatMessages(prev => [...prev, userMessage]);
    setMessageInput('');
    setIsTyping(true);

    try {
      // Simulate AI processing
      await new Promise(resolve => setTimeout(resolve, 1500));
      
      const botResponse = await generateAIResponse(message);
      
      setChatMessages(prev => [...prev, {
        id: Date.now() + 1,
        type: 'bot',
        message: botResponse.message,
        timestamp: new Date(),
        suggestions: botResponse.suggestions,
        relatedProducts: botResponse.relatedProducts,
        culturalInfo: botResponse.culturalInfo
      }]);

    } catch (error) {
      console.error('Error sending message:', error);
      setChatMessages(prev => [...prev, {
        id: Date.now() + 1,
        type: 'bot',
        message: "I apologize, but I'm experiencing some technical difficulties. Please try again.",
        timestamp: new Date(),
        isError: true
      }]);
    } finally {
      setIsTyping(false);
    }
  };

  // AI Response Generator
  const generateAIResponse = async (userMessage) => {
    const message = userMessage.toLowerCase();
    
    // Pottery-related queries
    if (message.includes('pottery') || message.includes('ceramic') || message.includes('clay')) {
      return {
        message: "🏺 Pottery is one of humanity's oldest crafts! I can help you find authentic ceramic pieces. Traditional pottery often shows unique glazing techniques, firing methods, and regional styles that reflect the artisan's cultural heritage.",
        suggestions: [
          "Show me pottery from specific regions",
          "Explain pottery techniques",
          "Find pottery-making workshops",
          "Sustainable pottery practices"
        ],
        relatedProducts: await mockProductSearch('pottery'),
        culturalInfo: {
          title: "Pottery Cultural Heritage",
          description: "Pottery traditions span thousands of years, with each culture developing unique techniques and styles.",
          techniques: ["Wheel throwing", "Hand building", "Glazing", "Firing methods"]
        }
      };
    }

    // Wooden crafts queries
    if (message.includes('wood') || message.includes('carved') || message.includes('doll')) {
      return {
        message: "🪆 Wooden crafts represent incredible skill and patience! Traditional wood carving often uses local wood types and techniques passed down through generations. Each piece tells a story of cultural identity.",
        suggestions: [
          "Types of wood used in crafts",
          "Regional carving styles",
          "Wood finishing techniques",
          "Caring for wooden items"
        ],
        relatedProducts: await mockProductSearch('wooden_dolls'),
        culturalInfo: {
          title: "Wood Carving Traditions",
          description: "From Russian matryoshkas to African masks, wood carving is deeply rooted in cultural expression.",
          techniques: ["Relief carving", "Chip carving", "Whittling", "Turning"]
        }
      };
    }

    // Sustainability queries
    if (message.includes('sustainable') || message.includes('eco') || message.includes('environment')) {
      return {
        message: "🌱 Sustainability is at the heart of traditional handicrafts! Most artisan products use natural, biodegradable materials and time-honored techniques that have minimal environmental impact.",
        suggestions: [
          "Eco-friendly materials used",
          "Carbon footprint of handmade vs mass-produced",
          "Supporting sustainable artisans",
          "Recycling and upcycling crafts"
        ],
        culturalInfo: {
          title: "Sustainable Craft Practices",
          description: "Traditional crafts often embody sustainable practices developed over centuries.",
          benefits: ["Natural materials", "Low carbon footprint", "Biodegradable", "Supporting local economies"]
        }
      };
    }

    // Default response
    return {
      message: "I'd be happy to help you explore the world of authentic handicrafts! Could you tell me more about what you're looking for?",
      suggestions: [
        "Tell me about pottery",
        "Show me wooden crafts",
        "Sustainability in handicrafts",
        "Cultural significance of textiles",
        "How to verify authenticity"
      ]
    };
  };

  // Mock product search
  const mockProductSearch = async (category) => {
    return [
      {
        id: 1,
        title: `Handcrafted ${category} piece`,
        price: 75,
        image: '/images/sample-product.jpg',
        artisan: 'Local Artisan'
      },
      {
        id: 2,
        title: `Traditional ${category} art`,
        price: 120,
        image: '/images/sample-product2.jpg',
        artisan: 'Heritage Crafts'
      }
    ];
  };

  // Handle image upload and analysis
  const onDrop = async (acceptedFiles) => {
    const file = acceptedFiles[0];
    if (!file) return;

    setLoading(true);
    
    try {
      // Convert to base64
      const reader = new FileReader();
      reader.onload = async (e) => {
        const imageData = e.target.result;
        
        // Simulate AI image analysis
        await new Promise(resolve => setTimeout(resolve, 2000));
        
        const analysis = await analyzeImage(imageData);
        setImageAnalysis(analysis);
        
        // Add analysis to chat
        setChatMessages(prev => [...prev, {
          id: Date.now(),
          type: 'user',
          message: "📸 Image uploaded for analysis",
          image: imageData,
          timestamp: new Date()
        }, {
          id: Date.now() + 1,
          type: 'bot',
          message: `🎯 I've analyzed your image! This appears to be ${analysis.category} with ${Math.round(analysis.confidence * 100)}% confidence.`,
          timestamp: new Date(),
          analysis: analysis,
          suggestions: [
            "Find similar products",
            "Learn about this craft type",
            "Get pricing estimates",
            "Connect with artisans"
          ]
        }]);
      };
      reader.readAsDataURL(file);
      
    } catch (error) {
      console.error('Image analysis error:', error);
    } finally {
      setLoading(false);
    }
  };

  // AI Image Analysis
  const analyzeImage = async (imageData) => {
    // Simulate AI classification
    const categories = ['pottery', 'wooden_dolls', 'basket_weaving', 'handlooms'];
    const randomCategory = categories[Math.floor(Math.random() * categories.length)];
    
    return {
      category: randomCategory,
      confidence: Math.random() * 0.3 + 0.7, // 70-100%
      description: `This appears to be a traditional ${randomCategory} piece. The craftsmanship shows authentic handmade characteristics.`,
      techniques: [`Traditional ${randomCategory} making`, 'Hand-finished details', 'Natural materials'],
      culturalOrigin: 'Regional traditional craft',
      estimatedValue: Math.floor(Math.random() * 100) + 50,
      authenticity: 'High - Shows traditional crafting methods'
    };
  };

  // Voice input handling
  const startListening = () => {
    if (recognitionRef.current) {
      setIsListening(true);
      recognitionRef.current.start();
      
      recognitionRef.current.onresult = (event) => {
        const transcript = event.results[0][0].transcript;
        setVoiceInput(transcript);
        sendMessage(transcript);
        setIsListening(false);
      };
      
      recognitionRef.current.onerror = () => {
        setIsListening(false);
      };
    }
  };

  // Text-to-speech
  const speakMessage = (text) => {
    if ('speechSynthesis' in window) {
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.lang = language;
      window.speechSynthesis.speak(utterance);
    }
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png', '.gif']
    },
    multiple: false
  });

  const tabItems = [
    { label: 'AI Chat', icon: <BotIcon /> },
    { label: 'Image Analysis', icon: <CameraIcon /> },
    { label: 'Cultural Insights', icon: <MagicIcon /> }
  ];

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      {/* Header */}
      <Box sx={{ textAlign: 'center', mb: 4 }}>
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <Avatar
            sx={{
              width: 80,
              height: 80,
              bgcolor: 'primary.main',
              margin: '0 auto 16px auto'
            }}
          >
            <AIIcon sx={{ fontSize: 40 }} />
          </Avatar>
          <Typography variant="h3" sx={{ fontWeight: 600, mb: 2 }}>
            AI Marketplace Assistant
          </Typography>
          <Typography variant="h6" color="text.secondary">
            Your intelligent guide to authentic handicrafts and cultural heritage
          </Typography>
        </motion.div>
      </Box>

      {/* Main Interface */}
      <Paper elevation={4} sx={{ borderRadius: 3, overflow: 'hidden' }}>
        {/* Tabs */}
        <Tabs
          value={activeTab}
          onChange={(e, newValue) => setActiveTab(newValue)}
          variant="fullWidth"
          sx={{ borderBottom: 1, borderColor: 'divider' }}
        >
          {tabItems.map((tab, index) => (
            <Tab
              key={index}
              icon={tab.icon}
              label={tab.label}
              sx={{ py: 2 }}
            />
          ))}
        </Tabs>

        {/* Chat Interface */}
        {activeTab === 0 && (
          <Box sx={{ height: 600, display: 'flex', flexDirection: 'column' }}>
            {/* Messages Area */}
            <Box sx={{ flex: 1, p: 2, overflow: 'auto', maxHeight: 500 }}>
              <AnimatePresence>
                {chatMessages.map((msg, index) => (
                  <motion.div
                    key={msg.id}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.3 }}
                  >
                    <MessageBubble 
                      message={msg} 
                      onSpeak={speakMessage}
                      onSuggestionClick={sendMessage}
                    />
                  </motion.div>
                ))}
              </AnimatePresence>
              
              {/* Typing Indicator */}
              {isTyping && (
                <Box sx={{ display: 'flex', alignItems: 'center', mt: 2 }}>
                  <Avatar sx={{ width: 32, height: 32, bgcolor: 'primary.main', mr: 1 }}>
                    <BotIcon sx={{ fontSize: 16 }} />
                  </Avatar>
                  <Paper sx={{ p: 2, borderRadius: 2 }}>
                    <Typography variant="body2" color="text.secondary">
                      AI is thinking...
                    </Typography>
                    <LinearProgress sx={{ mt: 1, width: 100 }} />
                  </Paper>
                </Box>
              )}
              <div ref={messagesEndRef} />
            </Box>

            {/* Input Area */}
            <Divider />
            <Box sx={{ p: 2, display: 'flex', alignItems: 'center', gap: 1 }}>
              <TextField
                fullWidth
                placeholder="Ask me about handicrafts, cultural heritage, or anything else..."
                value={messageInput}
                onChange={(e) => setMessageInput(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && !e.shiftKey && sendMessage()}
                multiline
                maxRows={3}
              />
              <IconButton 
                color="primary"
                onClick={startListening}
                disabled={isListening}
              >
                <MicIcon />
              </IconButton>
              <Button
                variant="contained"
                onClick={() => sendMessage()}
                disabled={!messageInput.trim()}
                sx={{ minWidth: 'auto', p: 1.5 }}
              >
                <SendIcon />
              </Button>
            </Box>
          </Box>
        )}

        {/* Image Analysis Tab */}
        {activeTab === 1 && (
          <Box sx={{ p: 4 }}>
            <Typography variant="h5" sx={{ mb: 3, textAlign: 'center' }}>
              🎨 AI-Powered Image Analysis
            </Typography>
            
            {/* Dropzone */}
            <Box
              {...getRootProps()}
              sx={{
                border: '2px dashed',
                borderColor: isDragActive ? 'primary.main' : 'divider',
                borderRadius: 2,
                p: 4,
                textAlign: 'center',
                cursor: 'pointer',
                bgcolor: isDragActive ? 'action.hover' : 'transparent',
                transition: 'all 0.3s ease',
                mb: 3
              }}
            >
              <input {...getInputProps()} />
              <PhotoIcon sx={{ fontSize: 48, color: 'text.secondary', mb: 2 }} />
              <Typography variant="h6" sx={{ mb: 1 }}>
                {isDragActive ? 'Drop your image here!' : 'Upload or drag an image'}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Our AI will identify the handicraft type, authenticity, and cultural significance
              </Typography>
            </Box>

            {loading && (
              <Box sx={{ textAlign: 'center', mb: 3 }}>
                <LinearProgress sx={{ mb: 2 }} />
                <Typography>Analyzing image with AI...</Typography>
              </Box>
            )}

            {/* Analysis Results */}
            {imageAnalysis && (
              <motion.div
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.5 }}
              >
                <ImageAnalysisResults analysis={imageAnalysis} />
              </motion.div>
            )}
          </Box>
        )}

        {/* Cultural Insights Tab */}
        {activeTab === 2 && (
          <Box sx={{ p: 4 }}>
            <Typography variant="h5" sx={{ mb: 3, textAlign: 'center' }}>
              🌍 Cultural Heritage Explorer
            </Typography>
            <CulturalInsightsPanel />
          </Box>
        )}
      </Paper>

      {/* Help FAB */}
      <Fab
        color="secondary"
        sx={{ position: 'fixed', bottom: 24, right: 24 }}
        onClick={() => setHelpDialogOpen(true)}
      >
        <HelpIcon />
      </Fab>

      {/* Help Dialog */}
      <Dialog open={helpDialogOpen} onClose={() => setHelpDialogOpen(false)}>
        <DialogTitle>AI Assistant Help</DialogTitle>
        <DialogContent>
          <Typography paragraph>
            Our AI Assistant can help you with:
          </Typography>
          <ul>
            <li>Identifying handicrafts from photos</li>
            <li>Finding authentic artisan products</li>
            <li>Learning about cultural heritage</li>
            <li>Getting pricing and authenticity insights</li>
            <li>Connecting with artisan communities</li>
          </ul>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setHelpDialogOpen(false)}>Close</Button>
        </DialogActions>
      </Dialog>
    </Container>
  );
};

// Message Bubble Component
const MessageBubble = ({ message, onSpeak, onSuggestionClick }) => {
  const isBot = message.type === 'bot';
  
  return (
    <Box sx={{ mb: 2, display: 'flex', justifyContent: isBot ? 'flex-start' : 'flex-end' }}>
      {isBot && (
        <Avatar sx={{ width: 32, height: 32, bgcolor: 'primary.main', mr: 1 }}>
          <BotIcon sx={{ fontSize: 16 }} />
        </Avatar>
      )}
      
      <Box sx={{ maxWidth: '70%' }}>
        <Paper
          sx={{
            p: 2,
            bgcolor: isBot ? 'background.paper' : 'primary.main',
            color: isBot ? 'text.primary' : 'primary.contrastText',
            borderRadius: 2,
            borderBottomLeftRadius: isBot ? 0 : 2,
            borderBottomRightRadius: isBot ? 2 : 0
          }}
        >
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <Typography variant="body1">{message.message}</Typography>
            {isBot && (
              <IconButton size="small" onClick={() => onSpeak(message.message)}>
                <VolumeIcon sx={{ fontSize: 16 }} />
              </IconButton>
            )}
          </Box>

          {message.image && (
            <Box sx={{ mt: 1 }}>
              <img
                src={message.image}
                alt="Uploaded"
                style={{
                  maxWidth: '200px',
                  maxHeight: '200px',
                  borderRadius: '8px',
                  objectFit: 'cover'
                }}
              />
            </Box>
          )}

          {message.analysis && (
            <Box sx={{ mt: 2 }}>
              <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 1 }}>
                Analysis Results:
              </Typography>
              <Chip label={`${message.analysis.category} (${Math.round(message.analysis.confidence * 100)}%)`} size="small" color="success" />
            </Box>
          )}

          {message.culturalInfo && (
            <Box sx={{ mt: 2 }}>
              <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
                {message.culturalInfo.title}
              </Typography>
              <Typography variant="body2">
                {message.culturalInfo.description}
              </Typography>
            </Box>
          )}
        </Paper>

        {/* Suggestions */}
        {message.suggestions && (
          <Box sx={{ mt: 1, display: 'flex', gap: 0.5, flexWrap: 'wrap' }}>
            {message.suggestions.map((suggestion, index) => (
              <Chip
                key={index}
                label={suggestion}
                size="small"
                variant="outlined"
                clickable
                onClick={() => onSuggestionClick(suggestion)}
                sx={{ fontSize: '0.75rem' }}
              />
            ))}
          </Box>
        )}

        <Typography variant="caption" color="text.secondary" sx={{ mt: 0.5, display: 'block' }}>
          {message.timestamp.toLocaleTimeString()}
        </Typography>
      </Box>
    </Box>
  );
};

// Image Analysis Results Component
const ImageAnalysisResults = ({ analysis }) => (
  <Card>
    <CardContent>
      <Typography variant="h6" sx={{ mb: 2, display: 'flex', alignItems: 'center' }}>
        🎯 Analysis Results
        <Chip 
          label={`${Math.round(analysis.confidence * 100)}% confidence`} 
          size="small" 
          color="success" 
          sx={{ ml: 2 }}
        />
      </Typography>
      
      <Grid container spacing={2}>
        <Grid item xs={12} md={6}>
          <Typography variant="subtitle2" gutterBottom>Category</Typography>
          <Typography variant="body1" sx={{ mb: 2 }}>{analysis.category}</Typography>
          
          <Typography variant="subtitle2" gutterBottom>Description</Typography>
          <Typography variant="body2" sx={{ mb: 2 }}>{analysis.description}</Typography>
          
          <Typography variant="subtitle2" gutterBottom>Authenticity</Typography>
          <Typography variant="body2">{analysis.authenticity}</Typography>
        </Grid>
        
        <Grid item xs={12} md={6}>
          <Typography variant="subtitle2" gutterBottom>Techniques Identified</Typography>
          {analysis.techniques.map((technique, index) => (
            <Chip key={index} label={technique} size="small" sx={{ mr: 0.5, mb: 0.5 }} />
          ))}
          
          <Typography variant="subtitle2" sx={{ mt: 2 }} gutterBottom>Estimated Value</Typography>
          <Typography variant="h6" color="primary">{formatINRPrice(parseFloat(analysis.estimatedValue), false)}</Typography>
        </Grid>
      </Grid>
    </CardContent>
  </Card>
);

// Cultural Insights Panel Component
const CulturalInsightsPanel = () => {
  const insights = [
    {
      title: "Pottery Traditions Worldwide",
      description: "Explore the rich history of ceramic arts across different cultures",
      techniques: ["Japanese Raku", "Greek Amphorae", "Native American Pueblo", "Chinese Porcelain"]
    },
    {
      title: "Textile Heritage",
      description: "Traditional weaving and fabric arts that preserve cultural identity", 
      techniques: ["Indian Khadi", "Peruvian Alpaca", "Scottish Tartan", "Kente Cloth"]
    },
    {
      title: "Wood Carving Mastery",
      description: "Sculptural traditions passed down through generations",
      techniques: ["Balinese Wood Carving", "Nordic Woodwork", "African Masks", "Totem Poles"]
    }
  ];

  return (
    <Box>
      {insights.map((insight, index) => (
        <Accordion key={index} sx={{ mb: 1 }}>
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Typography variant="h6">{insight.title}</Typography>
          </AccordionSummary>
          <AccordionDetails>
            <Typography paragraph>{insight.description}</Typography>
            <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap' }}>
              {insight.techniques.map((technique, i) => (
                <Chip key={i} label={technique} size="small" variant="outlined" />
              ))}
            </Box>
          </AccordionDetails>
        </Accordion>
      ))}
    </Box>
  );
};

export default AIAssistantPage;