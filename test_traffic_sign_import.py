#!/usr/bin/env python3
"""
Test script to verify traffic sign detection imports
"""

import sys
import os

def test_traffic_sign_imports():
    """Test if traffic sign detection modules can be imported"""
    print("🔍 Testing Traffic Sign Detection Imports")
    print("=" * 50)
    
    try:
        # Add the traffic sign detection directory to the path
        traffic_sign_path = os.path.join(os.path.dirname(__file__), 'Traffic-Sign-Detection', 'Traffic-Sign-Detection')
        if traffic_sign_path not in sys.path:
            sys.path.append(traffic_sign_path)
        
        print(f"📁 Added path: {traffic_sign_path}")
        
        # Test imports
        print("\n📦 Testing imports...")
        
        from classification import training, getLabel
        print("✅ classification module imported")
        
        from improved_classification import improved_training, improved_getLabel
        print("✅ improved_classification module imported")
        
        from main import localization
        print("✅ main module imported")
        
        from common import clock, mosaic
        print("✅ common module imported")
        
        print("\n🎉 All traffic sign detection modules imported successfully!")
        print("✅ Traffic sign detection is ready to use")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = test_traffic_sign_imports()
    if success:
        print("\n🚀 Traffic sign detection is working!")
    else:
        print("\n⚠️ Traffic sign detection needs attention") 