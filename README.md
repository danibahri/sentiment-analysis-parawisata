# 🏝️ Sentiment Analysis - Gili Labak Tourism

Aplikasi web interaktif untuk analisis sentimen komentar wisata Gili Labak menggunakan Streamlit. Aplikasi ini menampilkan proses analisis sentimen secara real-time dari awal hingga akhir.

## ✨ Fitur

- 🔤 **Input Real-time**: Masukkan komentar dan dapatkan analisis langsung
- 📊 **Proses Step-by-step**: Melihat setiap tahap preprocessing dan analisis
- 🎯 **Sentiment Analysis**: Klasifikasi sentimen (Positif, Negatif, Netral)
- 🏷️ **Aspect-based Analysis**: Analisis berdasarkan 5 aspek utama:
  - 🚗 Aksesibilitas
  - 🏢 Fasilitas
  - 🧽 Kebersihan
  - 🌴 Keindahan Alam
  - 🛎️ Layanan
- 📈 **Visualisasi**: Progress bar dan hasil yang mudah dipahami
- 🎨 **UI Modern**: Interface yang menarik dan responsive

## 🚀 Cara Menjalankan

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Jalankan Aplikasi

```bash
streamlit run app.py
```

### 3. Buka Browser

Aplikasi akan terbuka otomatis di `http://localhost:8501`
atau bisa
Link Aplikasi `https://sentiment-analysis-parawisata.streamlit.app/`

## 📖 Cara Penggunaan

1. **Input Komentar**: Masukkan komentar di sidebar
2. **Klik Analisis**: Tekan tombol "🚀 Analisis Sentimen"
3. **Lihat Proses**: Amati setiap tahap preprocessing:
   - Teks asli
   - Text cleaning
   - Normalisasi
   - Tokenisasi
   - Analisis sentimen
   - Analisis aspek
4. **Hasil**: Lihat hasil sentiment dan analisis aspek

## 🛠️ Teknologi yang Digunakan

- **Streamlit**: Framework web app
- **NLTK**: Natural Language Processing
- **Sastrawi**: Indonesian text processing
- **Pandas**: Data manipulation
- **Matplotlib/Seaborn**: Visualisasi
- **Scikit-learn**: Machine learning utilities

## 💡 Contoh Komentar

**Positif**: "Pantai Gili Labak sangat indah dan bersih, pemandangannya amazing banget!"

**Negatif**: "Aksesnya susah banget, toiletnya kotor, dan mahal"

**Netral**: "Pulau kecil di madura yang bisa dikunjungi untuk liburan"

## 📂 Struktur Project

```
Infor-parawisata/
├── app.py              # Aplikasi Streamlit utama
├── main.ipynb          # Notebook pengembangan model
├── dataset.csv         # Dataset training
├── requirements.txt    # Dependencies
└── README.md          # Dokumentasi
```

## 🔧 Model Details

Model menggunakan pendekatan rule-based dengan:

- Dictionary kata positif dan negatif
- Preprocessing text (cleaning, normalization, stemming)
- Context-aware aspect sentiment analysis
- Confidence scoring untuk setiap prediksi

## 📞 Support

Jika ada pertanyaan atau masalah, silakan buat issue di repository ini.
