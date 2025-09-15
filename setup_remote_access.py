"""
Setup Remote Access for Artisans Hub
This script helps you create public URLs for your team to access from office
"""

import subprocess
import sys
import os
import time
import requests
import json

def check_ngrok_installed():
    """Check if ngrok is installed"""
    try:
        result = subprocess.run(['ngrok', 'version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ ngrok is installed")
            return True
        else:
            print("❌ ngrok not found")
            return False
    except FileNotFoundError:
        print("❌ ngrok not installed")
        return False

def install_ngrok_instructions():
    """Provide instructions to install ngrok"""
    print("\n📦 HOW TO INSTALL NGROK (FREE TUNNELING SERVICE)")
    print("=" * 60)
    print("1. 🌐 Go to: https://ngrok.com/")
    print("2. 📝 Sign up for free account")
    print("3. 💾 Download ngrok for Windows")
    print("4. 📁 Extract ngrok.exe to a folder")
    print("5. ➕ Add folder to Windows PATH or copy to System32")
    print("6. 🔑 Run: ngrok config add-authtoken YOUR_TOKEN")
    print("7. ▶️ Run this script again")
    
    print("\n🚀 QUICK INSTALL OPTION (PowerShell as Admin):")
    print("choco install ngrok")
    print("OR")
    print("winget install ngrok")

def start_ngrok_tunnel(port, service_name):
    """Start ngrok tunnel for a specific port"""
    print(f"\n🌐 Starting {service_name} tunnel on port {port}...")
    
    try:
        # Start ngrok tunnel in background
        process = subprocess.Popen(
            ['ngrok', 'http', str(port)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait a moment for ngrok to start
        time.sleep(3)
        
        # Get tunnel URL from ngrok API
        try:
            response = requests.get('http://localhost:4040/api/tunnels')
            if response.status_code == 200:
                tunnels = response.json()['tunnels']
                for tunnel in tunnels:
                    if str(port) in tunnel['config']['addr']:
                        public_url = tunnel['public_url']
                        print(f"✅ {service_name} tunnel created!")
                        print(f"🔗 Public URL: {public_url}")
                        return public_url, process
            
            print(f"⚠️ Could not get tunnel URL for {service_name}")
            return None, process
            
        except Exception as e:
            print(f"⚠️ Could not fetch tunnel info: {e}")
            print(f"💡 Check manually at: http://localhost:4040")
            return None, process
            
    except Exception as e:
        print(f"❌ Failed to start {service_name} tunnel: {e}")
        return None, None

def create_team_links_file(frontend_url, backend_url):
    """Create a file with public links for the team"""
    
    content = f"""🌐 ARTISANS HUB - REMOTE ACCESS LINKS
==========================================

📅 Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}
🌍 Access Type: REMOTE (Internet Access)
🏆 Status: ✅ LIVE AND ACCESSIBLE

🎯 MAIN APPLICATION LINKS
==========================================

🖥️ FRONTEND APPLICATION:
{frontend_url if frontend_url else 'Starting...'}

🔧 BACKEND API:
{backend_url if backend_url else 'Starting...'}

📱 TEAM ACCESS INSTRUCTIONS
==========================================

✅ SHARE WITH TEAM MEMBERS:
1. Send them the Frontend Application link above
2. They can access from anywhere with internet
3. No VPN or special setup required
4. Works on mobile, tablet, and desktop

🧪 FOR TESTING:
• Main App: Use Frontend Application link
• API Testing: Use Backend API link + /api/health
• Mobile: Same links work on mobile devices

🎮 APPLICATION FEATURES TO TEST
==========================================

🤖 AI Features:
• Upload handicraft images for classification
• Get AI-generated prices in Indian Rupees
• Test with different categories

🏪 Marketplace:
• Browse Wooden Dolls, Handlooms, Basket Weaving, Pottery
• View sample Indian artisan products
• Check responsive mobile design

📱 User Experience:
• Test hamburger menu navigation
• Try image upload functionality
• Check smooth animations and UI

⚠️ IMPORTANT NOTES
==========================================

🔒 SECURITY: These are public URLs - don't share sensitive data
⏰ DURATION: Links stay active while your computer is running
🔄 REFRESH: If you restart, run the remote access script again
🌐 SPEED: Slight delay due to tunneling (normal for remote access)

🛠️ TROUBLESHOOTING
==========================================

❌ If links don't work:
1. Check if both frontend and backend servers are running
2. Restart the remote access script
3. Verify ngrok is properly installed

🔧 API Endpoints for Developers:
• Health: {backend_url}/api/health
• Products: {backend_url}/api/products
• Upload: {backend_url}/api/upload-analyze

📞 TEAM SUPPORT
==========================================

If team members have issues:
1. Try the links in incognito/private mode
2. Clear browser cache
3. Try different browsers (Chrome, Firefox, Safari)
4. Check internet connection

🎉 Ready for Remote Team Demo!
==========================================

Share the Frontend Application link with your team.
They can now access Artisans Hub from the office! 🚀

Last Updated: {time.strftime('%Y-%m-%d %H:%M:%S')}
"""

    with open('REMOTE_TEAM_ACCESS.txt', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"📄 Team access file created: REMOTE_TEAM_ACCESS.txt")

def main():
    """Main function to setup remote access"""
    print("🌍 ARTISANS HUB - REMOTE ACCESS SETUP")
    print("=" * 50)
    print("Setting up public URLs for your office team...")
    
    # Check if ngrok is installed
    if not check_ngrok_installed():
        install_ngrok_instructions()
        input("\n⏳ Press Enter after installing ngrok...")
        
        if not check_ngrok_installed():
            print("❌ Please install ngrok first!")
            return
    
    print("\n🚀 Starting remote access tunnels...")
    
    # Start tunnels for frontend and backend
    frontend_url, frontend_process = start_ngrok_tunnel(3001, "Frontend")
    time.sleep(2)  # Wait before starting second tunnel
    backend_url, backend_process = start_ngrok_tunnel(5000, "Backend")
    
    # Create team access file
    create_team_links_file(frontend_url, backend_url)
    
    print(f"\n🎉 REMOTE ACCESS SETUP COMPLETE!")
    print("=" * 40)
    
    if frontend_url:
        print(f"📱 SHARE THIS LINK WITH YOUR TEAM:")
        print(f"🔗 {frontend_url}")
        print(f"\n📋 Complete details in: REMOTE_TEAM_ACCESS.txt")
    else:
        print("⚠️ Check ngrok dashboard at: http://localhost:4040")
    
    print(f"\n⚡ IMPORTANT:")
    print("• Keep this script running to maintain access")
    print("• Don't close this terminal window")
    print("• Team can access from anywhere with internet")
    
    # Keep script running
    try:
        print(f"\n🔄 Tunnels active... Press Ctrl+C to stop")
        while True:
            time.sleep(30)
            print("🟢 Tunnels still active...")
    except KeyboardInterrupt:
        print(f"\n🛑 Stopping tunnels...")
        if frontend_process:
            frontend_process.terminate()
        if backend_process:
            backend_process.terminate()
        print("✅ Remote access stopped.")

if __name__ == "__main__":
    main()