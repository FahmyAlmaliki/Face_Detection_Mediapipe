import cv2
import numpy as np
import os

class SimpleExpressionDetector:
    def __init__(self):
        # Load Haar Cascade untuk deteksi wajah
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
        
        print("✓ Haar Cascades loaded successfully!")
        
    def detect_expression(self, face_roi):
        """Deteksi ekspresi sederhana berdasarkan mata dan senyum"""
        gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY) if len(face_roi.shape) == 3 else face_roi
        
        # Deteksi mata
        eyes = self.eye_cascade.detectMultiScale(gray_face, 1.1, 5)
        
        # Deteksi senyum
        smiles = self.smile_cascade.detectMultiScale(gray_face, 1.8, 20)
        
        # Analisis sederhana
        if len(smiles) > 0:
            return "😊 Senang"
        elif len(eyes) < 2:  # Mata tertutup/menyipit bisa menandakan marah
            return "😠 Marah"
        elif len(eyes) == 2:  # Mata normal tapi tidak senyum
            # Analisis tambahan berdasarkan posisi mata
            if len(eyes) == 2:
                eye1_y = eyes[0][1] + eyes[0][3]//2
                eye2_y = eyes[1][1] + eyes[1][3]//2
                
                # Jika mata pada posisi normal
                if abs(eye1_y - eye2_y) < 10:
                    return "😐 Neutral"
                else:
                    return "😢 Sedih"
        else:
            return "😐 Neutral"
    
    def process_frame(self, frame):
        """Process satu frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Deteksi wajah
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        for (x, y, w, h) in faces:
            # Gambar rectangle untuk wajah
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Extract ROI wajah
            face_roi = frame[y:y+h, x:x+w]
            face_gray = gray[y:y+h, x:x+w]
            
            # Deteksi ekspresi
            expression = self.detect_expression(face_roi)
            
            # Gambar label ekspresi
            label = f'Ekspresi: {expression}'
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            
            # Background untuk text
            (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
            cv2.rectangle(frame, (x, y-35), (x + text_width + 10, y-5), (0, 0, 0), -1)
            
            # Text ekspresi
            cv2.putText(frame, label, (x+5, y-15), font, font_scale, (0, 255, 0), thickness)
            
            # Deteksi dan gambar mata
            eyes = self.eye_cascade.detectMultiScale(face_gray, 1.1, 5)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(frame, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (255, 0, 0), 2)
            
            # Deteksi dan gambar senyum
            smiles = self.smile_cascade.detectMultiScale(face_gray, 1.8, 20)
            for (sx, sy, sw, sh) in smiles:
                cv2.rectangle(frame, (x+sx, y+sy), (x+sx+sw, y+sy+sh), (0, 0, 255), 2)
            
            return frame, expression
        
        return frame, "Tidak Ada Wajah"
    
    def run_webcam(self):
        """Jalankan deteksi dengan webcam"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Tidak dapat mengakses webcam")
            return
        
        print("🎥 Memulai deteksi wajah dan ekspresi...")
        print("Tekan 'q' untuk keluar")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Tidak dapat membaca frame")
                break
            
            # Process frame
            processed_frame, expression = self.process_frame(frame)
            
            # Tambah instruksi
            cv2.putText(processed_frame, "Tekan 'q' untuk keluar", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Tambah panduan
            guide = [
                "Panduan:",
                "- Kotak Hijau: Wajah",
                "- Kotak Biru: Mata", 
                "- Kotak Merah: Senyum"
            ]
            
            for i, text in enumerate(guide):
                cv2.putText(processed_frame, text, 
                           (10, 60 + i * 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Tampilkan frame
            cv2.imshow('Face Expression Detection - OpenCV Haar Cascade', processed_frame)
            
            # Keluar jika 'q' ditekan
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def process_image(self, image_path):
        """Process gambar"""
        if not os.path.exists(image_path):
            print(f"Error: File {image_path} tidak ditemukan")
            return
        
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Tidak dapat membaca gambar {image_path}")
            return
        
        processed_image, expression = self.process_frame(image)
        
        print(f"Ekspresi terdeteksi: {expression}")
        
        # Tampilkan gambar
        cv2.imshow('Face Expression Detection - OpenCV Haar Cascade', processed_image)
        print("Tekan sembarang tombol untuk menutup...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main():
    print("=" * 60)
    print("🎭 DETEKSI WAJAH DAN EKSPRESI 🎭")
    print("Menggunakan OpenCV Haar Cascade")
    print("=" * 60)
    
    try:
        detector = SimpleExpressionDetector()
        
        print("Mode yang tersedia:")
        print("1. Webcam (Real-time)")
        print("2. Gambar (File)")
        print("=" * 60)
        
        choice = input("Pilih mode (1/2): ")
        
        if choice == "1":
            print("\n🎥 Memulai mode webcam...")
            detector.run_webcam()
        elif choice == "2":
            image_path = input("Masukkan path gambar: ")
            print(f"\n🖼️ Memproses gambar: {image_path}")
            detector.process_image(image_path)
        else:
            print("❌ Pilihan tidak valid")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Pastikan webcam tidak digunakan aplikasi lain")

if __name__ == "__main__":
    main()
