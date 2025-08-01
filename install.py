#!/usr/bin/env python3
"""
Installation Script for PP-PicoDet-L Object Detection System
==========================================================

This script helps install and set up the object detection system.
"""

import subprocess
import sys
import os
import platform

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    else:
        print(f"✓ Python version {version.major}.{version.minor}.{version.micro} is compatible")
        return True

def install_requirements():
    """Install required packages"""
    print("\nInstalling required packages...")
    
    try:
        # Install packages from requirements.txt
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ All packages installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install packages: {e}")
        return False

def test_imports():
    """Test if all required modules can be imported"""
    print("\nTesting imports...")
    
    required_modules = [
        'cv2',
        'numpy',
        'torch',
        'torchvision',
        'PIL',
        'ultralytics'
    ]
    
    failed_imports = []
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"✓ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\n❌ Failed to import: {', '.join(failed_imports)}")
        return False
    else:
        print("\n✓ All imports successful")
        return True

def test_gpu_availability():
    """Test GPU availability"""
    print("\nChecking GPU availability...")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✓ GPU available: {gpu_name}")
            print(f"✓ GPU count: {gpu_count}")
        else:
            print("⚠️  No GPU detected, will use CPU")
        return True
    except Exception as e:
        print(f"❌ Error checking GPU: {e}")
        return False

def download_test_model():
    """Download a test model if needed"""
    print("\nChecking model availability...")
    
    try:
        from ultralytics import YOLO
        
        # Try to load a small model for testing
        model = YOLO('yolov8n.pt')
        print("✓ Model loaded successfully")
        return True
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        print("This might be due to network issues. You can try again later.")
        return False

def create_sample_script():
    """Create a sample usage script"""
    sample_script = '''#!/usr/bin/env python3
"""
Sample usage of PP-PicoDet-L Object Detection System
"""

from object_detector import PP_PicoDet_Detector
import cv2

def main():
    # Initialize detector
    detector = PP_PicoDet_Detector(
        confidence_threshold=0.5,
        device='auto'
    )
    
    # Start webcam detection
    print("Starting webcam detection...")
    print("Press 'q' to quit")
    
    detector.process_video(video_source=0)

if __name__ == "__main__":
    main()
'''
    
    with open("sample_usage.py", "w") as f:
        f.write(sample_script)
    
    print("✓ Sample usage script created: sample_usage.py")

def main():
    """Main installation function"""
    print("PP-PicoDet-L Object Detection System Installation")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Install requirements
    if not install_requirements():
        return False
    
    # Test imports
    if not test_imports():
        return False
    
    # Test GPU
    test_gpu_availability()
    
    # Test model download
    if not download_test_model():
        print("⚠️  Model download failed, but installation can continue")
    
    # Create sample script
    create_sample_script()
    
    print("\n" + "=" * 50)
    print("🎉 Installation completed successfully!")
    print("=" * 50)
    print("\nNext steps:")
    print("1. Run tests: python test_detector.py")
    print("2. Try demo: python demo.py")
    print("3. Use webcam: python main.py --webcam")
    print("4. Check sample: python sample_usage.py")
    print("\nFor more information, see README.md")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        if not success:
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n❌ Installation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Installation failed: {e}")
        sys.exit(1) 