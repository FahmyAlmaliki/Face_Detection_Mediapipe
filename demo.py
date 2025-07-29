"""
Contoh penggunaan Face Expression Detector
"""
from main import FaceExpressionDetector
import cv2

def demo_image_processing():
    """Demo pemrosesan gambar"""
    print("=== Demo Pemrosesan Gambar ===")
    
    detector = FaceExpressionDetector()
    
    # Contoh membuat gambar test dengan webcam
    print("Ambil foto test dengan webcam...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Webcam tidak dapat diakses")
        return
    
    print("Tekan SPACE untuk ambil foto, ESC untuk keluar")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Tampilkan preview
        cv2.putText(frame, "Tekan SPACE untuk foto, ESC untuk keluar", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow('Ambil Foto Test', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 32:  # SPACE
            # Simpan foto
            cv2.imwrite('test_photo.jpg', frame)
            print("Foto tersimpan sebagai test_photo.jpg")
            
            # Process foto
            processed_frame, expression = detector.process_frame(frame)
            print(f"Ekspresi terdeteksi: {expression}")
            
            # Tampilkan hasil
            cv2.imshow('Hasil Deteksi', processed_frame)
            cv2.waitKey(3000)  # Tampilkan 3 detik
            break
        elif key == 27:  # ESC
            break
    
    cap.release()
    cv2.destroyAllWindows()

def print_usage_info():
    """Tampilkan informasi penggunaan"""
    print("\n=== INFORMASI PENGGUNAAN ===")
    print("Program ini dapat mendeteksi 4 jenis ekspresi:")
    print("1. 😊 SENANG - Sudut mulut naik, mata sedikit terbuka")
    print("2. 😢 SEDIH - Sudut mulut turun, alis turun")
    print("3. 😠 MARAH - Alis mengerut, mata menyipit")
    print("4. 😐 NEUTRAL - Ekspresi default/normal")
    print()
    print("TIPS untuk hasil yang lebih baik:")
    print("• Pastikan pencahayaan cukup")
    print("• Wajah menghadap kamera secara langsung")
    print("• Jarak optimal 50-100cm dari kamera")
    print("• Ekspresikan emosi dengan jelas")
    print()
    print("KONTROL:")
    print("• Mode Webcam: Tekan 'q' untuk keluar")
    print("• Mode Gambar: Tekan sembarang tombol untuk tutup")

if __name__ == "__main__":
    print_usage_info()
    
    choice = input("\nIngin coba demo foto? (y/n): ")
    if choice.lower() == 'y':
        demo_image_processing()
