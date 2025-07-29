# Face Detection dan Expression Classification

Program ini menyediakan dua pendekatan untuk mendeteksi wajah dan mengklasifikasikan ekspresi wajah menjadi tiga kategori: **Senang**, **Sedih**, dan **Marah**.

## 🎯 Fitur

### Versi 1: MediaPipe (main.py)
- **Deteksi Wajah Real-time**: Menggunakan MediaPipe Face Mesh untuk deteksi landmark wajah yang akurat (468 titik)
- **Kontur Wajah**: Menampilkan detail kontur mata, bibir, dan fitur wajah
- **Klasifikasi Ekspresi Advanced**: Menggunakan fitur geometris dari landmark

### Versi 2: OpenCV Haar Cascade (face_detection_opencv.py)
- **Deteksi Wajah Klasik**: Menggunakan Haar Cascade yang lebih ringan
- **Deteksi Mata dan Senyum**: Visualisasi komponen wajah
- **Klasifikasi Ekspresi Sederhana**: Rule-based detection

## 🚀 Instalasi

1. Clone repository ini:
```bash
git clone https://github.com/FahmyAlmaliki/Face_Detection_Mediapipe.git
cd Face_Detection_Mediapipe
```

2. Install dependencies:

### Untuk Versi OpenCV (Recommended):
```bash
pip install opencv-python==4.5.5.64 numpy==1.21.6
```

### Untuk Versi MediaPipe (Jika bermasalah dengan dependencies):
```bash
pip install -r requirements.txt
```

## 📱 Cara Penggunaan

### Versi OpenCV (Lebih Stabil):
```bash
python face_detection_opencv.py
```

### Versi MediaPipe (Lebih Akurat):
```bash
python main.py
```

Pilih mode:
- **Mode 1 (Webcam)**: Deteksi real-time menggunakan webcam
- **Mode 2 (Gambar)**: Deteksi pada gambar statis

### Kontrol
- Tekan `q` untuk keluar dari mode webcam
- Tekan sembarang tombol untuk tutup hasil gambar

## 🎭 Klasifikasi Ekspresi

### Versi MediaPipe:
1. **😊 Senang**: Sudut mulut naik + mulut sedikit terbuka
2. **😢 Sedih**: Sudut mulut turun + alis turun
3. **😠 Marah**: Alis mengerut + mata menyipit
4. **😐 Neutral**: Ekspresi default

### Versi OpenCV:
1. **😊 Senang**: Terdeteksi senyum (Haar Cascade)
2. **😠 Marah**: Mata menyipit/tertutup
3. **😢 Sedih**: Mata tidak sejajar
4. **😐 Neutral**: Ekspresi normal

## 🔧 Troubleshooting

### Jika MediaPipe bermasalah:
```bash
# Gunakan versi yang kompatibel
pip uninstall mediapipe numpy opencv-python
pip install opencv-python==4.5.5.64 numpy==1.21.6 mediapipe==0.8.11
```

### Jika webcam tidak berfungsi:
- Pastikan webcam tidak digunakan aplikasi lain
- Coba ubah index webcam di `cv2.VideoCapture(0)` menjadi `cv2.VideoCapture(1)`

### Performance:
- **OpenCV**: Lebih cepat, cocok untuk perangkat dengan spec rendah
- **MediaPipe**: Lebih akurat, butuh resource lebih tinggi

## 📊 Perbandingan Metode

| Aspek | OpenCV Haar Cascade | MediaPipe Face Mesh |
|-------|-------------------|-------------------|
| **Kecepatan** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Akurasi** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Stabilitas** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Detail Deteksi** | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Resource Usage** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |

## 📋 Requirements

### Minimal (OpenCV):
- Python 3.7+
- OpenCV 4.5.5+
- NumPy 1.21.6+

### Full (MediaPipe):
- Python 3.7+
- OpenCV 4.5.5+
- MediaPipe 0.8.11+
- NumPy 1.21.6+

## 🔄 Pengembangan Lebih Lanjut

1. **Machine Learning**: Implementasi CNN untuk klasifikasi yang lebih akurat
2. **Real-time Performance**: Optimasi untuk fps yang lebih tinggi
3. **Multi-face Detection**: Support deteksi multiple wajah
4. **Emotion Recognition**: Tambah emosi seperti terkejut, jijik, takut
5. **Mobile Deployment**: Port ke Android/iOS

## 📝 Files Structure

```
Face_Detection_Mediapipe/
├── main.py                    # MediaPipe version
├── face_detection_opencv.py   # OpenCV version  
├── test_opencv.py            # OpenCV testing
├── demo.py                   # Demo utilities
├── config.py                 # Configuration
├── requirements.txt          # Dependencies
└── README.md                # Documentation
```

## 🆘 Support

Jika mengalami masalah:
1. Coba versi OpenCV terlebih dahulu (`face_detection_opencv.py`)
2. Pastikan webcam berfungsi dengan `test_opencv.py`
3. Check dependencies dengan Python yang sesuai

## 📄 Lisensi

MIT License