"""
Test script untuk memverifikasi instalasi dependencies
"""
import sys

def test_imports():
    try:
        import cv2
        print("✓ OpenCV berhasil diimport")
        print(f"  Version: {cv2.__version__}")
        
        import mediapipe as mp
        print("✓ MediaPipe berhasil diimport")
        print(f"  Version: {mp.__version__}")
        
        import numpy as np
        print("✓ NumPy berhasil diimport")
        print(f"  Version: {np.__version__}")
        
        import sklearn
        print("✓ Scikit-learn berhasil diimport")
        print(f"  Version: {sklearn.__version__}")
        
        import pandas as pd
        print("✓ Pandas berhasil diimport")
        print(f"  Version: {pd.__version__}")
        
        print("\n🎉 Semua dependencies berhasil diimport!")
        print("Program siap dijalankan!")
        
        return True
        
    except ImportError as e:
        print(f"❌ Error importing: {e}")
        return False

if __name__ == "__main__":
    test_imports()
