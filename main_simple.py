import cv2
import mediapipe as mp
import numpy as np
import time

class SimpleFaceExpressionDetector:
    def __init__(self):
        # Initialize MediaPipe
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize face mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Key landmarks for expression detection
        self.LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        self.MOUTH = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318]
        self.EYEBROWS = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46, 285, 295, 282, 283, 276, 300, 293, 334, 296, 336]
        
    def extract_features(self, landmarks):
        """Extract facial features for expression classification"""
        if not landmarks:
            return None
            
        # Convert landmarks to numpy array
        points = np.array([[lm.x, lm.y] for lm in landmarks.landmark])
        
        features = []
        
        # Eye aspect ratio (EAR) - for detecting blinks and eye openness
        left_eye_points = points[self.LEFT_EYE]
        right_eye_points = points[self.RIGHT_EYE]
        
        # Calculate left eye aspect ratio
        left_ear = self.calculate_ear(left_eye_points)
        right_ear = self.calculate_ear(right_eye_points)
        features.extend([left_ear, right_ear])
        
        # Mouth aspect ratio (MAR) - for detecting mouth openness
        mouth_points = points[self.MOUTH]
        mar = self.calculate_mar(mouth_points)
        features.append(mar)
        
        # Eyebrow height - for detecting raised/lowered eyebrows
        eyebrow_points = points[self.EYEBROWS]
        eyebrow_height = np.mean(eyebrow_points[:, 1])
        features.append(eyebrow_height)
        
        # Mouth corner positions - for detecting smile/frown
        mouth_corners = [points[78], points[308]]  # Left and right mouth corners
        mouth_center = points[13]  # Center of upper lip
        
        left_corner_height = mouth_corners[0][1] - mouth_center[1]
        right_corner_height = mouth_corners[1][1] - mouth_center[1]
        features.extend([left_corner_height, right_corner_height])
        
        # Distance between eyebrows (for anger detection)
        eyebrow_distance = np.linalg.norm(points[70] - points[295])
        features.append(eyebrow_distance)
        
        return np.array(features)
    
    def calculate_ear(self, eye_points):
        """Calculate Eye Aspect Ratio"""
        if len(eye_points) < 6:
            return 0
        
        # Vertical eye landmarks
        A = np.linalg.norm(eye_points[1] - eye_points[5])
        B = np.linalg.norm(eye_points[2] - eye_points[4])
        
        # Horizontal eye landmark
        C = np.linalg.norm(eye_points[0] - eye_points[3])
        
        # Eye aspect ratio
        if C == 0:
            return 0
        ear = (A + B) / (2.0 * C)
        return ear
    
    def calculate_mar(self, mouth_points):
        """Calculate Mouth Aspect Ratio"""
        if len(mouth_points) < 13:
            return 0
            
        # Vertical mouth landmarks
        A = np.linalg.norm(mouth_points[2] - mouth_points[10])
        B = np.linalg.norm(mouth_points[4] - mouth_points[8])
        
        # Horizontal mouth landmark
        C = np.linalg.norm(mouth_points[0] - mouth_points[6])
        
        # Mouth aspect ratio
        if C == 0:
            return 0
        mar = (A + B) / (2.0 * C)
        return mar
    
    def classify_expression(self, features):
        """Simple rule-based expression classification"""
        if features is None or len(features) < 7:
            return "Tidak Terdeteksi"
        
        ear_left, ear_right, mar, eyebrow_height, left_corner, right_corner, eyebrow_distance = features
        
        avg_ear = (ear_left + ear_right) / 2
        avg_corner = (left_corner + right_corner) / 2
        
        # Rule-based classification dengan threshold yang disesuaikan
        if avg_corner < -0.005 and mar > 0.015:  # Mouth corners up and mouth slightly open
            return "😊 Senang"
        elif avg_corner > 0.005 and eyebrow_height > 0.42:  # Mouth corners down and eyebrows down
            return "😢 Sedih"
        elif eyebrow_distance < 0.06 and eyebrow_height > 0.43 and avg_ear < 0.27:  # Eyebrows close and down, eyes narrowed
            return "😠 Marah"
        else:
            return "😐 Neutral"
    
    def draw_landmarks(self, image, landmarks):
        """Draw face landmarks on image"""
        if landmarks:
            # Draw face mesh
            self.mp_drawing.draw_landmarks(
                image=image,
                landmark_list=landmarks,
                connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style()
            )
            
            # Draw key points
            self.mp_drawing.draw_landmarks(
                image=image,
                landmark_list=landmarks,
                connections=self.mp_face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_iris_connections_style()
            )
    
    def process_frame(self, frame):
        """Process a single frame"""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.face_mesh.process(rgb_frame)
        
        # Draw landmarks and classify expression
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Draw landmarks
                self.draw_landmarks(frame, face_landmarks)
                
                # Extract features and classify expression
                features = self.extract_features(face_landmarks)
                expression = self.classify_expression(features)
                
                # Get face bounding box
                h, w, _ = frame.shape
                landmarks_px = [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks.landmark]
                
                # Calculate bounding box
                x_coords = [p[0] for p in landmarks_px]
                y_coords = [p[1] for p in landmarks_px]
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                
                # Draw bounding box
                cv2.rectangle(frame, (x_min - 20, y_min - 20), (x_max + 20, y_max + 20), (0, 255, 0), 2)
                
                # Draw expression label with background
                label = f'Ekspresi: {expression}'
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                thickness = 2
                
                # Get text size
                (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
                
                # Draw background rectangle
                cv2.rectangle(frame, (x_min - 20, y_min - 50), 
                             (x_min - 20 + text_width + 10, y_min - 50 + text_height + 10), 
                             (0, 0, 0), -1)
                
                # Draw text
                cv2.putText(frame, label, 
                           (x_min - 15, y_min - 30), 
                           font, font_scale, (0, 255, 0), thickness)
                
                # Show features for debugging (optional)
                if features is not None and len(features) >= 7:
                    debug_text = f'EAR: {features[0]:.3f}, MAR: {features[2]:.3f}, Corner: {(features[4]+features[5])/2:.3f}'
                    cv2.putText(frame, debug_text, 
                               (10, h - 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                return frame, expression
        
        return frame, "Tidak Ada Wajah"
    
    def run_webcam(self):
        """Run face detection on webcam"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Tidak dapat mengakses webcam")
            return
        
        # Set webcam resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("Memulai deteksi wajah... Tekan 'q' untuk keluar")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Tidak dapat membaca frame dari webcam")
                break
            
            # Process frame
            processed_frame, expression = self.process_frame(frame)
            
            # Add instructions
            cv2.putText(processed_frame, "Tekan 'q' untuk keluar", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Add expression guide
            guide_text = [
                "Panduan Ekspresi:",
                "😊 Senang: Senyum lebar",
                "😢 Sedih: Mulut turun",
                "😠 Marah: Alis mengerut",
                "😐 Neutral: Rileks"
            ]
            
            for i, text in enumerate(guide_text):
                cv2.putText(processed_frame, text, 
                           (10, 60 + i * 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Display frame
            cv2.imshow('Face Expression Detection - MediaPipe', processed_frame)
            
            # Break on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def process_image(self, image_path):
        """Process a single image"""
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Tidak dapat membaca gambar {image_path}")
            return
        
        processed_image, expression = self.process_frame(image)
        
        print(f"Ekspresi terdeteksi: {expression}")
        
        # Display image
        cv2.imshow('Face Expression Detection - MediaPipe', processed_image)
        print("Tekan sembarang tombol untuk menutup...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main():
    detector = SimpleFaceExpressionDetector()
    
    print("=" * 50)
    print("🎭 PROGRAM DETEKSI WAJAH DAN EKSPRESI 🎭")
    print("Menggunakan MediaPipe Face Mesh")
    print("=" * 50)
    print("1. Webcam (Real-time)")
    print("2. Gambar (File)")
    print("=" * 50)
    
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

if __name__ == "__main__":
    main()
