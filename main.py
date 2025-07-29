import cv2
import mediapipe as mp
import numpy as np
import time

class FaceExpressionDetector:
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
        
        # Simple rule-based classifier (can be replaced with ML model)
        self.expression_labels = ['Neutral', 'Senang', 'Sedih', 'Marah']
        
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
        ear = (A + B) / (2.0 * C)
        return ear
    
    def calculate_mar(self, mouth_points):
        """Calculate Mouth Aspect Ratio"""
        # Vertical mouth landmarks
        A = np.linalg.norm(mouth_points[2] - mouth_points[10])
        B = np.linalg.norm(mouth_points[4] - mouth_points[8])
        
        # Horizontal mouth landmark
        C = np.linalg.norm(mouth_points[0] - mouth_points[6])
        
        # Mouth aspect ratio
        mar = (A + B) / (2.0 * C)
        return mar
    
    def classify_expression(self, features):
        """Simple rule-based expression classification"""
        if features is None:
            return "Tidak Terdeteksi"
        
        ear_left, ear_right, mar, eyebrow_height, left_corner, right_corner, eyebrow_distance = features
        
        avg_ear = (ear_left + ear_right) / 2
        avg_corner = (left_corner + right_corner) / 2
        
        # Rule-based classification
        if avg_corner < -0.01 and mar > 0.02:  # Mouth corners up and mouth slightly open
            return "Senang"
        elif avg_corner > 0.01 and eyebrow_height > 0.4:  # Mouth corners down and eyebrows down
            return "Sedih"
        elif eyebrow_distance < 0.05 and eyebrow_height > 0.45 and avg_ear < 0.25:  # Eyebrows close and down, eyes narrowed
            return "Marah"
        else:
            return "Neutral"
    
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
                
                # Draw expression label
                cv2.putText(frame, f'Ekspresi: {expression}', 
                           (x_min - 20, y_min - 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                return frame, expression
        
        return frame, "Tidak Ada Wajah"
    
    def run_webcam(self):
        """Run face detection on webcam"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Tidak dapat mengakses webcam")
            return
        
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
            
            # Display frame
            cv2.imshow('Face Expression Detection', processed_frame)
            
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
        cv2.imshow('Face Expression Detection', processed_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main():
    detector = FaceExpressionDetector()
    
    print("=== Program Deteksi Wajah dan Ekspresi ===")
    print("1. Webcam")
    print("2. Gambar")
    
    choice = input("Pilih mode (1/2): ")
    
    if choice == "1":
        detector.run_webcam()
    elif choice == "2":
        image_path = input("Masukkan path gambar: ")
        detector.process_image(image_path)
    else:
        print("Pilihan tidak valid")

if __name__ == "__main__":
    main()