"""
Script installer untuk Face Detection Program
"""
import subprocess
import sys
import os

def install_packages(packages):
    """Install packages menggunakan pip"""
    for package in packages:
        try:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✓ {package} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}")
            return False
    return True

def test_imports(modules):
    """Test import modules"""
    for module in modules:
        try:
            __import__(module)
            print(f"✓ {module} imported successfully")
        except ImportError as e:
            print(f"❌ Failed to import {module}: {e}")
            return False
    return True

def install_opencv_version():
    """Install OpenCV version"""
    print("=" * 50)
    print("🔧 INSTALLING OPENCV VERSION")
    print("=" * 50)
    
    packages = [
        "opencv-python==4.5.5.64",
        "numpy==1.21.6"
    ]
    
    if install_packages(packages):
        print("\n🧪 Testing imports...")
        if test_imports(["cv2", "numpy"]):
            print("\n✅ OpenCV version installed successfully!")
            print("You can now run: python face_detection_opencv.py")
            return True
    
    print("\n❌ OpenCV version installation failed!")
    return False

def install_mediapipe_version():
    """Install MediaPipe version"""
    print("=" * 50)
    print("🔧 INSTALLING MEDIAPIPE VERSION")
    print("=" * 50)
    
    packages = [
        "opencv-python==4.5.5.64",
        "numpy==1.21.6",
        "mediapipe==0.8.11"
    ]
    
    if install_packages(packages):
        print("\n🧪 Testing imports...")
        if test_imports(["cv2", "numpy", "mediapipe"]):
            print("\n✅ MediaPipe version installed successfully!")
            print("You can now run: python main.py")
            return True
    
    print("\n❌ MediaPipe version installation failed!")
    return False

def main():
    print("=" * 60)
    print("🎭 FACE DETECTION PROGRAM INSTALLER")
    print("=" * 60)
    
    print("Choose installation type:")
    print("1. OpenCV Version (Recommended - More Stable)")
    print("2. MediaPipe Version (Advanced - More Accurate)")
    print("3. Both Versions")
    print("=" * 60)
    
    choice = input("Enter your choice (1/2/3): ")
    
    if choice == "1":
        install_opencv_version()
    elif choice == "2":
        install_mediapipe_version()
    elif choice == "3":
        print("Installing both versions...")
        success1 = install_opencv_version()
        print("\n" + "="*50)
        success2 = install_mediapipe_version()
        
        if success1 and success2:
            print("\n🎉 Both versions installed successfully!")
        elif success1:
            print("\n⚠️ Only OpenCV version installed successfully")
        elif success2:
            print("\n⚠️ Only MediaPipe version installed successfully")
        else:
            print("\n❌ Both installations failed")
    else:
        print("❌ Invalid choice")
    
    print("\n" + "="*60)
    print("Installation completed!")
    input("Press Enter to exit...")

if __name__ == "__main__":
    main()
