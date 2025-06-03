# Demo Script - Sentiment Analysis Gili Labak Tourism

## Cara Menjalankan Aplikasi:

1. **Buka Terminal/Command Prompt** dan navigasi ke folder project:

   ```bash
   cd "c:\laragon\www\Infor-parawisata"
   ```

2. **Jalankan aplikasi Streamlit**:

   ```bash
   streamlit run app.py
   ```

3. **Buka browser** ke `http://localhost:8501`

## Contoh Input untuk Demo:

### 1. Komentar Positif:

```
Pantai Gili Labak sangat indah dan bersih banget! Pemandangannya amazing, airnya jernih, dan pasirnya putih. Recommended untuk liburan keluarga. Fasilitas juga lumayan bagus.
```

### 2. Komentar Negatif:

```
Aksesnya susah banget, perjalanan lama dan mahal. Toilet kotor, warung sedikit. Fasilitas kurang bagus, sinyal tidak ada. Rugi jauh-jauh ke sini.
```

### 3. Komentar Netral:

```
Gili Labak adalah pulau kecil di Madura yang bisa dikunjungi untuk wisata. Ada kapal dari pelabuhan untuk ke sana. Perjalanan memakan waktu beberapa jam.
```

### 4. Komentar Mixed (Positif-Negatif):

```
Pantainya indah dan view-nya bagus banget, tapi aksesnya jauh dan susah. Airnya bersih tapi fasilitas kurang lengkap. Overall masih worth it untuk dikunjungi.
```

## Fitur yang Akan Ditampilkan:

1. **Progress Bar Real-time** - Menunjukkan setiap tahap processing
2. **Step-by-step Processing**:

   - Text asli
   - Text cleaning (remove URL, mentions, hashtags)
   - Normalisasi (stemming, remove stopwords)
   - Tokenisasi
   - Sentiment analysis
   - Aspect-based analysis

3. **Hasil Analisis**:

   - Sentimen keseluruhan (Positif/Negatif/Netral)
   - Kata-kata yang terdeteksi (positif/negatif)
   - Analisis per aspek (Aksesibilitas, Fasilitas, Kebersihan, Keindahan Alam, Layanan)
   - Confidence score untuk setiap aspek

4. **Visualisasi**:
   - Progress indicator
   - Color-coded sentiment results
   - Metrics dashboard

## Tips Penggunaan:

- Gunakan komentar dalam bahasa Indonesia
- Sertakan kata-kata yang relevan dengan wisata
- Coba variasi panjang komentar (pendek vs panjang)
- Test dengan campuran sentimen positif dan negatif

## Troubleshooting:

Jika ada error saat pertama kali menjalankan:

1. Pastikan semua packages terinstall dengan: `pip install -r requirements.txt`
2. Download NLTK data dengan menjalankan script pertama kali
3. Restart aplikasi jika ada masalah loading

Selamat menggunakan aplikasi Sentiment Analysis! 🏝️✨
