const express = require('express');
const bcrypt = require('bcrypt');
const session = require('express-session');
const { v4: uuidv4 } = require('uuid');
const nodemailer = require('nodemailer');
const crypto = require('crypto');
const cors = require('cors');
const fetch = require('node-fetch');
require('dotenv').config();
const pool = require('./db-config');

const app = express();

const isProduction = process.env.NODE_ENV === 'production';
if (isProduction) {
    // Needed when behind a proxy (e.g. Render, Railway, Vercel)
    app.set('trust proxy', 1);
}

// CORS configuration for deployment
const corsOptions = {
    origin: process.env.FRONTEND_URL || 'http://localhost:3000',
    credentials: true,
    methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
    allowedHeaders: ['Content-Type', 'Authorization']
};

app.use(cors(corsOptions));
app.use(express.json());
app.use(express.static(__dirname));

const transporter = nodemailer.createTransport({
    host: 'smtp.ethereal.email',
    port: 587,
    secure: false,
    auth: {
        user: process.env.ETHEREAL_EMAIL_USER || 'madisen.zulauf@ethereal.email',
        pass: process.env.ETHEREAL_EMAIL_PASS || 'zS2vXfQcEaKx4c3P54'
    }
});

app.use(session({
    genid: (req) => uuidv4(),
    secret: process.env.SESSION_SECRET || 'a-very-secret-key',
    resave: false,
    saveUninitialized: false,
    cookie: {
        secure: isProduction, // set true when served over HTTPS in production
        sameSite: isProduction ? 'none' : 'lax',
        httpOnly: true,
        maxAge: 60 * 60 * 1000
    }
}));

const geminiApiKey = process.env.GEMINI_API_KEY;
const geminiApiUrl = `https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key=${geminiApiKey}`;

const systemPrompt = `You are a friendly and helpful AI-powered chatbot assistant for a local artisan marketplace called 'Artisan Hub.' Your primary purpose is to assist local artisans and customers with questions about the platform. You must respond in a friendly and encouraging tone.

You have the following knowledge base:

- **Selling Artwork:** The platform helps local artisans market their craft, tell their stories, and expand their reach. To get started, artisans can create a profile and list their handmade items. The AI can assist with tasks like generating product descriptions.
- **Product Categories:** The platform features several art categories including Wooden Dolls, Handlooms, Basket Weaving, and Pottery.
- **AI Features:** The AI can help with tasks like generating product descriptions, suggesting ideal pricing, and translating product descriptions.
- **Artisan Success Stories:** The marketplace plans to showcase success stories to inspire other sellers, but this feature is currently in development. You can mention that this is a great feature to add in the future.
- **Platform Navigation:** The website has a user profile, a login page, a help center, and a search bar on the homepage.`;

app.post('/api/chatbot', async (req, res) => {
    const { message, history } = req.body;

    try {
        if (!geminiApiKey) {
            throw new Error('GEMINI_API_KEY is not set.');
        }

        const payload = {
            systemInstruction: { parts: [{ text: systemPrompt }] },
            contents: [...history, { role: 'user', parts: [{ text: message }] }],
        };

        const response = await fetch(geminiApiUrl, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`API call failed with status: ${response.status}. Message: ${errorText}`);
        }

        const result = await response.json();

        if (!result.candidates || result.candidates.length === 0) {
            throw new Error('No response candidates found from API.');
        }

        const botResponse = result.candidates[0].content.parts[0].text;

        res.json({ reply: botResponse });
    } catch (error) {
        console.error('Error during API call:', error);
        res.status(500).json({ error: 'Failed to get a response from the chatbot.' });
    }
});
// --- Auth and Other Routes ---
app.get('/api/artisans', async (req, res) => {
    try {
        const [rows] = await pool.query('SELECT * FROM artisans');
        res.status(200).json(rows);
    } catch (error) {
        console.error('Failed to retrieve artisans:', error);
        res.status(500).json({ error: 'Internal Server Error' });
    }
});

app.post('/signup', async (req, res) => {
    try {
        const { email, password } = req.body;
        const [existingUsers] = await pool.query('SELECT * FROM users WHERE email = ?', [email]);
        if (existingUsers.length > 0) {
            return res.status(409).json({ error: 'User already exists' });
        }
        const hashedPassword = await bcrypt.hash(password, 10);
        await pool.query('INSERT INTO users (email, password) VALUES (?, ?)', [email, hashedPassword]);
        res.status(201).json({ message: 'User created successfully' });
    } catch (error) {
        console.error('Signup Error:', error);
        res.status(500).json({ error: 'Internal server error' });
    }
});

app.post('/login', async (req, res) => {
    try {
        const { email, password } = req.body;
        const [users] = await pool.query('SELECT * FROM users WHERE email = ?', [email]);
        if (users.length === 0) {
            return res.status(401).json({ error: 'Invalid credentials' });
        }
        const user = users[0];
        const isMatch = await bcrypt.compare(password, user.password);
        if (!isMatch) {
            return res.status(401).json({ error: 'Invalid credentials' });
        }
        req.session.user = { id: user.id, email: user.email };
        res.status(200).json({ message: 'Login successful' });
    } catch (error) {
        console.error('Login Error:', error);
        res.status(500).json({ error: 'Internal server error' });
    }
});

app.get('/profile', async (req, res) => {
    if (!req.session.user) {
        return res.status(401).json({ error: 'Unauthorized' });
    }
    try {
        const [users] = await pool.query('SELECT email, phone FROM users WHERE id = ?', [req.session.user.id]);
        if (users.length === 0) {
            return res.status(404).json({ error: 'User not found' });
        }
        res.status(200).json(users[0]);
    } catch (error) {
        console.error('Profile Error:', error);
        res.status(500).json({ error: 'Internal server error' });
    }
});

app.post('/logout', (req, res) => {
    req.session.destroy(err => {
        if (err) {
            return res.status(500).json({ error: 'Could not log out, please try again' });
        }
        res.clearCookie('connect.sid');
        res.status(200).json({ message: 'Logout successful' });
    });
});

app.post('/forgot-password', async (req, res) => {
    try {
        const { email } = req.body;
        const [users] = await pool.query('SELECT * FROM users WHERE email = ?', [email]);
        if (users.length === 0) {
            return res.status(200).json({ message: 'If your email is in our database, you will receive a password reset link.' });
        }
        const user = users[0];
        const token = crypto.randomBytes(20).toString('hex');
        const expires = new Date(Date.now() + 3600000);

        await pool.query('UPDATE users SET resetPasswordToken = ?, resetPasswordExpires = ? WHERE id = ?', [token, expires, user.id]);

const resetLink = `${process.env.FRONTEND_URL || 'http://localhost:3000'}/reset-password.html?token=${token}`;

        const mailOptions = {
            to: user.email,
            from: 'password-reset@artisan-hub.com',
            subject: 'Password Reset',
            text: `...`
        };

        await transporter.sendMail(mailOptions);

        res.status(200).json({ message: 'Password reset link sent.' });
    } catch (error) {
        console.error('Forgot Password Error:', error);
        res.status(500).json({ error: 'Internal server error' });
    }
});

app.post('/reset-password/:token', async (req, res) => {
    try {
        const { token } = req.params;
        const { password } = req.body;
        const [users] = await pool.query('SELECT * FROM users WHERE resetPasswordToken = ? AND resetPasswordExpires > NOW()', [token]);

        if (users.length === 0) {
            return res.status(400).json({ error: 'Password reset token is invalid or has expired.' });
        }
        const user = users[0];
        const hashedPassword = await bcrypt.hash(password, 10);

        await pool.query('UPDATE users SET password = ?, resetPasswordToken = NULL, resetPasswordExpires = NULL WHERE id = ?', [hashedPassword, user.id]);

        res.status(200).json({ message: 'Password has been reset.' });
    } catch (error) {
        console.error('Reset Password Error:', error);
        res.status(500).json({ error: 'Internal server error' });
    }
});

// The routes for updating and verifying phone numbers are commented out as they use the Twilio client
/*
app.post('/profile/update-phone', (req, res) => {
    if (!req.session.user) {
        return res.status(401).json({ error: 'Unauthorized' });
    }
    const { newPhoneNumber } = req.body;
    const otp = Math.floor(100000 + Math.random() * 900000).toString();
    req.session.phoneUpdate = { newPhoneNumber, otp, timestamp: Date.now() };
    console.log(`Sending OTP ${otp} to ${newPhoneNumber}`);

    res.status(200).json({ message: 'OTP sent to the new phone number.' });
});

app.post('/profile/verify-phone', async (req, res) => {
    if (!req.session.user || !req.session.phoneUpdate) {
        return res.status(401).json({ error: 'Unauthorized or no pending phone update.' });
    }

    const { otp } = req.body;
    const { newPhoneNumber, otp: sessionOtp, timestamp } = req.session.phoneUpdate;

    if (Date.now() - timestamp > 600000) { // 10-minute expiry
        return res.status(400).json({ error: 'OTP has expired.' });
    }
    if (otp !== sessionOtp) {
        return res.status(400).json({ error: 'Invalid OTP.' });
    }

    try {
        await pool.query('UPDATE users SET phone = ? WHERE id = ?', [newPhoneNumber, req.session.user.id]);
        req.session.phoneUpdate = null; // Clear session data
        res.status(200).json({ message: 'Phone number updated successfully.' });
    } catch (error) {
        console.error('Verify Phone Error:', error);
        res.status(500).json({ error: 'Internal server error' });
    }
});
*/

const PORT = process.env.PORT || 3001;
app.listen(PORT, () => {
    console.log(`Server is running on port ${PORT}`);
});
