"""
Konfigurasi dan konstanta untuk Face Expression Detector
"""

# MediaPipe Configuration
MEDIAPIPE_CONFIG = {
    'static_image_mode': False,
    'max_num_faces': 1,
    'refine_landmarks': True,
    'min_detection_confidence': 0.5,
    'min_tracking_confidence': 0.5
}

# Face Landmark Indices (berdasarkan MediaPipe Face Mesh)
LANDMARKS = {
    # Mata kiri (16 titik)
    'LEFT_EYE': [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
    
    # Mata kanan (16 titik)
    'RIGHT_EYE': [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
    
    # Mulut (13 titik)
    'MOUTH': [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318],
    
    # Alis mata (20 titik)
    'EYEBROWS': [70, 63, 105, 66, 107, 55, 65, 52, 53, 46, 285, 295, 282, 283, 276, 300, 293, 334, 296, 336],
    
    # Titik khusus untuk deteksi ekspresi
    'MOUTH_CORNERS': [78, 308],  # Sudut mulut kiri dan kanan
    'MOUTH_CENTER': [13],        # Titik tengah bibir atas
    'EYEBROW_CENTER': [70, 295], # Titik tengah alis kiri dan kanan
}

# Threshold untuk klasifikasi ekspresi
EXPRESSION_THRESHOLDS = {
    'HAPPY': {
        'mouth_corner_threshold': -0.01,  # Sudut mulut naik
        'mouth_open_threshold': 0.02      # Mulut sedikit terbuka
    },
    'SAD': {
        'mouth_corner_threshold': 0.01,   # Sudut mulut turun
        'eyebrow_height_threshold': 0.4   # Alis turun
    },
    'ANGRY': {
        'eyebrow_distance_threshold': 0.05,  # Alis mengerut
        'eyebrow_height_threshold': 0.45,    # Alis turun
        'eye_openness_threshold': 0.25       # Mata menyipit
    }
}

# Label ekspresi
EXPRESSION_LABELS = {
    'NEUTRAL': 'Neutral',
    'HAPPY': 'Senang',
    'SAD': 'Sedih',
    'ANGRY': 'Marah',
    'NO_FACE': 'Tidak Ada Wajah',
    'NOT_DETECTED': 'Tidak Terdeteksi'
}

# Warna untuk visualisasi (BGR format untuk OpenCV)
COLORS = {
    'GREEN': (0, 255, 0),      # Untuk bounding box dan teks normal
    'RED': (0, 0, 255),        # Untuk error/warning
    'BLUE': (255, 0, 0),       # Untuk informasi
    'WHITE': (255, 255, 255),  # Untuk teks putih
    'YELLOW': (0, 255, 255),   # Untuk highlight
}

# Pengaturan font
FONT_CONFIG = {
    'font': 0,  # cv2.FONT_HERSHEY_SIMPLEX
    'scale': 0.7,
    'thickness': 2
}

# Pengaturan webcam
WEBCAM_CONFIG = {
    'width': 640,
    'height': 480,
    'fps': 30
}
