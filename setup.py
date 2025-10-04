"""
Setup and Installation Script for Student Engagement Monitoring System
Automatically installs dependencies and sets up the project
"""

import subprocess
import sys
import os
import time

def print_banner():
    """Print setup banner"""
    print("=" * 70)
    print("🎓 STUDENT ENGAGEMENT MONITORING SYSTEM - SETUP")
    print("=" * 70)
    print("This script will set up your engagement monitoring system")
    print("=" * 70)

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    print(f"🐍 Python Version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Error: Python 3.8 or higher is required!")
        print("   Please upgrade Python and try again.")
        return False
    
    print("✅ Python version is compatible")
    return True

def install_dependencies():
    """Install required dependencies"""
    print("\n📦 Installing Dependencies...")
    print("=" * 40)
    
    # Core dependencies that should work on most systems
    core_deps = [
        "opencv-python==4.8.1.78",
        "numpy==1.24.3",
        "matplotlib==3.7.2", 
        "requests==2.31.0",
        "Pillow==10.0.0"
    ]
    
    # Optional dependencies
    optional_deps = [
        "pandas==2.0.3",
        "scikit-learn==1.3.0",
        "seaborn==0.13.0",
        "tqdm==4.66.1"
    ]
    
    # Install core dependencies
    print("🔧 Installing core dependencies...")
    for dep in core_deps:
        try:
            print(f"   Installing {dep}...")
            result = subprocess.run([sys.executable, "-m", "pip", "install", dep], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print(f"   ✅ {dep} installed successfully")
            else:
                print(f"   ⚠️  Warning: Failed to install {dep}")
                print(f"      Error: {result.stderr}")
        except Exception as e:
            print(f"   ❌ Error installing {dep}: {e}")
    
    # Install optional dependencies
    print("\n🔧 Installing optional dependencies...")
    for dep in optional_deps:
        try:
            print(f"   Installing {dep}...")
            result = subprocess.run([sys.executable, "-m", "pip", "install", dep], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print(f"   ✅ {dep} installed successfully")
            else:
                print(f"   ⚠️  Skipping {dep} (not critical)")
        except Exception as e:
            print(f"   ⚠️  Skipping {dep}: {e}")

def install_tensorflow():
    """Install TensorFlow with compatibility handling"""
    print("\n🧠 Installing TensorFlow...")
    print("=" * 30)
    
    try:
        # Try to install TensorFlow
        print("   Installing TensorFlow...")
        result = subprocess.run([sys.executable, "-m", "pip", "install", "tensorflow"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("   ✅ TensorFlow installed successfully")
            return True
        else:
            print("   ⚠️  TensorFlow installation failed, trying CPU version...")
            result = subprocess.run([sys.executable, "-m", "pip", "install", "tensorflow-cpu"], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print("   ✅ TensorFlow CPU version installed successfully")
                return True
            else:
                print("   ❌ Failed to install TensorFlow")
                print("   💡 Note: TensorFlow is only needed for custom model training")
                return False
    except Exception as e:
        print(f"   ❌ Error installing TensorFlow: {e}")
        return False

def install_mediapipe():
    """Install MediaPipe with protobuf compatibility"""
    print("\n🎥 Installing MediaPipe (optional)...")
    print("=" * 40)
    
    try:
        # First try to install compatible protobuf
        print("   Installing compatible protobuf...")
        subprocess.run([sys.executable, "-m", "pip", "install", "protobuf==3.20.3"], 
                      capture_output=True, text=True)
        
        # Then install MediaPipe
        print("   Installing MediaPipe...")
        result = subprocess.run([sys.executable, "-m", "pip", "install", "mediapipe"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("   ✅ MediaPipe installed successfully")
            return True
        else:
            print("   ⚠️  MediaPipe installation failed")
            print("   💡 Note: MediaPipe is optional for advanced features")
            return False
    except Exception as e:
        print(f"   ⚠️  MediaPipe installation failed: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating Project Directories...")
    print("=" * 40)
    
    directories = [
        "data",
        "data/processed", 
        "models",
        "models/saved_models",
        "logs",
        "screenshots",
        "sessions",
        "auto_saves"
    ]
    
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"   ✅ Created: {directory}")
        except Exception as e:
            print(f"   ❌ Failed to create {directory}: {e}")

def test_installation():
    """Test if key components work"""
    print("\n🧪 Testing Installation...")
    print("=" * 30)
    
    tests = [
        ("OpenCV", "import cv2; print(f'OpenCV {cv2.__version__}')"),
        ("NumPy", "import numpy as np; print(f'NumPy {np.__version__}')"),
        ("Matplotlib", "import matplotlib; print(f'Matplotlib {matplotlib.__version__}')"),
        ("Tkinter", "import tkinter; print('Tkinter available')"),
    ]
    
    results = []
    
    for name, test_code in tests:
        try:
            exec(test_code)
            print(f"   ✅ {name}: Working")
            results.append(True)
        except Exception as e:
            print(f"   ❌ {name}: Failed - {e}")
            results.append(False)
    
    # Test camera access
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("   ✅ Camera: Accessible")
            cap.release()
            results.append(True)
        else:
            print("   ⚠️  Camera: Not accessible")
            results.append(False)
    except:
        print("   ❌ Camera: Test failed")
        results.append(False)
    
    return all(results)

def show_next_steps():
    """Show what to do next"""
    print("\n🎯 Next Steps:")
    print("=" * 20)
    print("1. Run the launcher: python launcher.py")
    print("2. Start with option 1 (Quick Test) to verify everything works")
    print("3. For full features, use option 2 (Comprehensive App)")
    print("4. Read README.md for detailed instructions")
    print("\n💡 Tips:")
    print("• Ensure your webcam is connected and working")
    print("• Start with the simple detector to test your setup")
    print("• Check the comprehensive app for advanced features")

def main():
    """Main setup function"""
    print_banner()
    
    # Check Python version
    if not check_python_version():
        input("Press Enter to exit...")
        return
    
    print(f"\n📍 Current Directory: {os.getcwd()}")
    
    # Ask user if they want to proceed
    response = input("\n🤔 Do you want to install dependencies? (y/n): ").strip().lower()
    if response != 'y':
        print("❌ Setup cancelled.")
        return
    
    start_time = time.time()
    
    # Install dependencies
    install_dependencies()
    
    # Try to install TensorFlow (optional)
    tensorflow_ok = install_tensorflow()
    
    # Try to install MediaPipe (optional)
    mediapipe_ok = install_mediapipe()
    
    # Create directories
    create_directories()
    
    # Test installation
    print("\n" + "=" * 70)
    all_good = test_installation()
    
    end_time = time.time()
    duration = end_time - start_time
    
    # Show results
    print("\n" + "=" * 70)
    print("🎉 SETUP COMPLETE!")
    print("=" * 70)
    print(f"⏱️  Setup time: {duration:.1f} seconds")
    
    if all_good:
        print("✅ All core components are working!")
        print("🚀 Your system is ready to use!")
    else:
        print("⚠️  Some components may not be working optimally")
        print("💡 You can still use the basic features")
    
    if not tensorflow_ok:
        print("\n📝 Note: TensorFlow not installed")
        print("   • Basic engagement detection will work fine")
        print("   • Custom model training won't be available")
    
    if not mediapipe_ok:
        print("\n📝 Note: MediaPipe not installed")
        print("   • Basic OpenCV detection will work")
        print("   • Advanced facial analysis won't be available")
    
    show_next_steps()
    
    input("\nPress Enter to exit setup...")

if __name__ == "__main__":
    main()