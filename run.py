import os
import sys
import subprocess
import webbrowser
import time

def check_dependencies():
    """Check if required packages are installed"""
    try:
        import flask
        import cv2
        import numpy
        import PIL
        import sklearn
        print("✅ All dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        return False

def install_dependencies():
    """Install required packages"""
    print("Installing dependencies...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

def train_model():
    """Train the ML model if not exists"""
    if not os.path.exists("color_classifier.pkl"):
        print("Training ML model...")
        from ml_classifier import ColorClassifier
        classifier = ColorClassifier()
        classifier.train()
        classifier.save_model("color_classifier.pkl")
        print("✅ Model trained and saved")
    else:
        print("✅ Model already exists")

def main():
    print("=" * 50)
    print("Water Quality Test Analyzer")
    print("=" * 50)
    
    # Check and install dependencies
    if not check_dependencies():
        print("\nInstalling missing dependencies...")
        install_dependencies()
    
    # Train model if needed
    train_model()
    
    # Start Flask app
    print("\nStarting web server...")
    port = 5000
    url = f"http://127.0.0.1:{port}"
    
    # Open browser after a short delay
    def open_browser():
        time.sleep(2)
        webbrowser.open(url)
        print(f"\n✅ Application started!")
        print(f"📱 Open your browser at: {url}")
        print("⚠️  Press Ctrl+C to stop the server")
    
    import threading
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    # Run Flask app
    from web_app import app
    app.run(debug=True, port=port)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down...")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)