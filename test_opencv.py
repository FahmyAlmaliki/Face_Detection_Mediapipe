import cv2
import numpy as np
import time

def test_opencv():
    """Test OpenCV dengan webcam"""
    print("Testing OpenCV...")
    
    # Test webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Tidak dapat mengakses webcam")
        return False
    
    print("OpenCV berhasil mengakses webcam!")
    print("Tekan 'q' untuk keluar")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Tidak dapat membaca frame")
            break
        
        # Tambahkan text
        cv2.putText(frame, "OpenCV Test - Tekan 'q' untuk keluar", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Tampilkan frame
        cv2.imshow('OpenCV Test', frame)
        
        # Keluar jika 'q' ditekan
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("OpenCV test selesai!")
    return True

def main():
    print("=" * 40)
    print("🔧 OPENCV TEST")
    print("=" * 40)
    
    try:
        import cv2
        print(f"✓ OpenCV version: {cv2.__version__}")
        
        import numpy as np
        print(f"✓ NumPy version: {np.__version__}")
        
        print("\nMemulai test webcam...")
        test_opencv()
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Silahkan install dependencies terlebih dahulu:")
        print("pip install opencv-python numpy")

if __name__ == "__main__":
    main()
