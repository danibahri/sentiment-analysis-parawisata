import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

# Download NLTK data jika belum ada
import nltk
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
except:
    pass

# Konfigurasi halaman
st.set_page_config(
    page_title="Sentiment Analysis - Gili Labak Tourism",
    page_icon="🏝️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS untuk styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .process-header {
        font-size: 1.5rem;
        color: #2E7D32;
        background-color: #E8F5E8;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .step-box {
        background-color: #F5F5F5;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #1E88E5;
    }
    .result-box {
        background-color: #E3F2FD;
        padding: 20px;
        border-radius: 10px;
        margin: 15px 0;
    }
    .positive {
        color: #4CAF50;
        font-weight: bold;
    }
    .negative {
        color: #F44336;
        font-weight: bold;
    }
    .neutral {
        color: #FF9800;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Inisialisasi model dan data
@st.cache_resource
def initialize_models():
    # Stemmer
    factory = StemmerFactory()
    stemmer_id = factory.create_stemmer()
    stemmer_en = PorterStemmer()
    
    # Stopwords
    try:
        stop_words_id = set(stopwords.words('indonesian'))
    except:
        stop_words_id = set()
    stop_words_en = set(stopwords.words('english'))
    all_stop_words = stop_words_id.union(stop_words_en)
    
    # Slang mapping
    slang_mapping = {
        'gk': 'tidak', 'ga': 'tidak', 'gak': 'tidak', 'ngk': 'tidak', 'tdk': 'tidak', 'g': 'tidak',
        'gw': 'saya', 'gue': 'saya', 'lu': 'kamu', 'loe': 'kamu', 'elo': 'kamu',
        'yg': 'yang', 'udh': 'sudah', 'sdh': 'sudah', 'dah': 'sudah',
        'bs': 'bisa', 'bgt': 'banget', 'bngt': 'banget', 'dgn': 'dengan',
        'pake': 'pakai', 'utk': 'untuk', 'bwt': 'buat', 'jg': 'juga',
        'aja': 'saja', 'aj': 'saja', 'blm': 'belum', 'emg': 'memang',
        'krn': 'karena', 'karna': 'karena', 'tp': 'tapi', 'tpi': 'tapi',
        'klo': 'kalau', 'kl': 'kalau', 'bsk': 'besok', 'dr': 'dari',
        'pd': 'pada', 'lg': 'lagi', 'sih': '', 'nih': '', 'ni': '',
        'deh': '', 'dong': '', 'donk': '', 'lah': '', 'tuh': 'itu',
        'ngga': 'tidak', 'nggak': 'tidak', 'brp': 'berapa'
    }
    
    # Kata positif dan negatif
    positive_words = [
        'bagus', 'mantap', 'indah', 'keren', 'amazing', 'cantik', 'wow', 'great', 
        'nice', 'good', 'love', 'suka', 'recommended', 'awesome',
        'beautiful', 'mantab', 'mantul', 'mantep', 'sukses', 'terimakasih', 'thanks'
    ]
    
    negative_words = [
        'kotor', 'jelek', 'buruk', 'mahal', 'kurang', 'bad', 'poor', 'expensive', 
        'dirty', 'panas', 'sulit', 'susah', 'takut', 'parah', 'jauh', 'capek', 'rugi'
    ]
    
    # Aspek
    aspects = {
        'aksesibilitas': ['jalan', 'akses', 'perjalanan', 'transportasi', 'kapal', 'perah', 'nyebrang', 'jauh'],
        'fasilitas': ['toilet', 'fasilitas', 'penginapan', 'homestay', 'pondok', 'warung', 'sinyal'],
        'kebersihan': ['bersih', 'kotor', 'sampah', 'clean'],
        'keindahan_alam': ['pantai', 'indah', 'cantik', 'beautiful', 'view', 'bagus', 'keren', 'bawah air', 'air', 'ombak'],
        'layanan': ['pelampung', 'safety', 'guide', 'paket', 'travel', 'trip', 'biaya', 'harga', 'open']
    }
    
    return stemmer_id, stemmer_en, all_stop_words, slang_mapping, positive_words, negative_words, aspects

# Fungsi preprocessing
def clean_text(text, progress_callback=None):
    """Membersihkan teks dari URL, mention, hashtag, dll"""
    if progress_callback:
        progress_callback("🧹 Membersihkan teks...")
    
    if isinstance(text, str): 
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        # Remove mentions (@username)
        text = re.sub(r'@\w+', '', text)
        # Remove hashtags (keep the text without #)
        text = re.sub(r'#(\w+)', r'\1', text)
        # Remove emojis and special characters
        text = re.sub(r'[^\w\s]', ' ', text)
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        # Convert to lowercase
        text = text.lower()
        return text
    else:
        return ""

def normalize_text(text, stemmer_id, stemmer_en, all_stop_words, slang_mapping, progress_callback=None):
    """Normalisasi teks dengan stemming dan penghapusan stopwords"""
    if progress_callback:
        progress_callback("🔄 Melakukan normalisasi teks...")
    
    if not isinstance(text, str) or not text:
        return ""
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Normalisasi kata gaul
    normalized_tokens = [slang_mapping.get(word, word) for word in tokens]
    
    # Menghapus stopwords
    filtered_tokens = [word for word in normalized_tokens if word not in all_stop_words and len(word) > 1]
    
    # Stemming
    stemmed_tokens = []
    for token in filtered_tokens:
        if len(token) > 3:
            # Indonesian stemming
            stemmed_id = stemmer_id.stem(token)
            if len(stemmed_id) < len(token) - 1:
                stemmed_tokens.append(stemmed_id)
            else:
                # English stemming
                stemmed_en = stemmer_en.stem(token)
                if len(stemmed_en) < len(token) - 1:
                    stemmed_tokens.append(stemmed_en)
                else:
                    stemmed_tokens.append(token)
        else:
            stemmed_tokens.append(token)
    
    return ' '.join(stemmed_tokens)

def analyze_sentiment(text, positive_words, negative_words, progress_callback=None):
    """Analisis sentimen berdasarkan kata positif dan negatif"""
    if progress_callback:
        progress_callback("🎯 Menganalisis sentimen...")
    
    if not isinstance(text, str) or not text:
        return 'netral'
    
    words = text.split()
    pos_count = sum(1 for word in words if word in positive_words)
    neg_count = sum(1 for word in words if word in negative_words)
    
    if pos_count > neg_count:
        return 'positif'
    elif neg_count > pos_count:
        return 'negatif'
    else:
        return 'netral'

def extract_aspects(text, aspects, positive_words, negative_words, progress_callback=None):
    """Ekstraksi aspek dan analisis sentimen per aspek"""
    if progress_callback:
        progress_callback("🔍 Menganalisis aspek...")
    
    if not isinstance(text, str) or not text:
        return {}
    
    result = {}
    words = text.split()
    
    for aspect_name, keywords in aspects.items():
        found_keywords = [word for word in words if word in keywords]
        if found_keywords:
            # Analisis sentimen untuk aspek ini
            aspect_sentiment = analyze_aspect_sentiment(text, found_keywords, positive_words, negative_words)
            confidence = calculate_confidence(text, found_keywords)
            
            result[aspect_name] = {
                'sentiment': aspect_sentiment,
                'keywords': found_keywords,
                'confidence': confidence
            }
    
    return result

def analyze_aspect_sentiment(text, aspect_keywords, positive_words, negative_words):
    """Analisis sentimen untuk aspek tertentu"""
    windows = []
    words = text.split()
    
    # Extract context windows around aspect keywords
    for keyword in aspect_keywords:
        if keyword in words:
            for i, word in enumerate(words):
                if word == keyword:
                    start = max(0, i-5)
                    end = min(len(words), i+6)
                    window = ' '.join(words[start:end])
                    windows.append(window)
    
    # Analyze sentiment in each context window
    sentiments = []
    for window in windows:
        window_words = window.split()
        pos_count = sum(1 for word in window_words if word in positive_words)
        neg_count = sum(1 for word in window_words if word in negative_words)
        
        if pos_count > neg_count:
            sentiments.append('positif')
        elif neg_count > pos_count:
            sentiments.append('negatif')
        else:
            sentiments.append('netral')
    
    if not sentiments:
        return 'netral'
    
    # Determine overall aspect sentiment
    pos_count = sentiments.count('positif')
    neg_count = sentiments.count('negatif')
    neu_count = sentiments.count('netral')
    
    if pos_count > neg_count and pos_count >= neu_count:
        return 'positif'
    elif neg_count > pos_count and neg_count >= neu_count:
        return 'negatif'
    else:
        return 'netral'

def calculate_confidence(text, keywords):
    """Menghitung confidence score"""
    keyword_density = len(keywords) / max(1, len(text.split()))
    confidence = 0.6 + keyword_density * 0.3
    return min(0.95, confidence)

# Main App
def main():
    st.markdown('<h1 class="main-header">🏝️ Sentiment Analysis - Gili Labak Tourism</h1>', unsafe_allow_html=True)
    
    # Initialize models
    with st.spinner("🔄 Memuat model..."):
        stemmer_id, stemmer_en, all_stop_words, slang_mapping, positive_words, negative_words, aspects = initialize_models()
    
    # Sidebar
    st.sidebar.header("📝 Input Komentar")
    st.sidebar.markdown("Masukkan komentar tentang wisata Gili Labak untuk dianalisis sentimennya.")
    
    # Input area
    user_input = st.sidebar.text_area(
        "Masukkan komentar Anda:",
        placeholder="Contoh: Pantai Gili Labak sangat indah dan bersih, tapi aksesnya agak susah...",
        height=100
    )
    
    analyze_button = st.sidebar.button("🚀 Analisis Sentimen", type="primary")
    
    # Main content area
    if analyze_button and user_input.strip():
        # Progress container
        progress_container = st.container()
        
        with progress_container:
            st.markdown('<div class="process-header">📊 Proses Analisis Sentiment</div>', unsafe_allow_html=True)
            
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Step 1: Original text
            progress_bar.progress(10)
            status_text.text("Step 1: Menampilkan teks asli...")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="step-box">', unsafe_allow_html=True)
                st.markdown("**🔤 Teks Asli:**")
                st.write(f'"{user_input}"')
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Step 2: Text cleaning
            progress_bar.progress(30)
            status_text.text("Step 2: Membersihkan teks...")
            
            cleaned_text = clean_text(user_input)
            
            with col2:
                st.markdown('<div class="step-box">', unsafe_allow_html=True)
                st.markdown("**🧹 Teks Bersih:**")
                st.write(f'"{cleaned_text}"')
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Step 3: Normalization
            progress_bar.progress(50)
            status_text.text("Step 3: Normalisasi teks...")
            
            normalized_text = normalize_text(cleaned_text, stemmer_id, stemmer_en, all_stop_words, slang_mapping)
            
            col3, col4 = st.columns(2)
            
            with col3:
                st.markdown('<div class="step-box">', unsafe_allow_html=True)
                st.markdown("**🔄 Teks Normalisasi:**")
                st.write(f'"{normalized_text}"')
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Step 4: Tokenization
            progress_bar.progress(70)
            status_text.text("Step 4: Tokenisasi...")
            
            tokens = word_tokenize(normalized_text) if normalized_text else []
            
            with col4:
                st.markdown('<div class="step-box">', unsafe_allow_html=True)
                st.markdown("**🔤 Token:**")
                st.write(tokens)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Step 5: Sentiment Analysis
            progress_bar.progress(85)
            status_text.text("Step 5: Analisis sentimen...")
            
            sentiment = analyze_sentiment(normalized_text, positive_words, negative_words)
            
            # Step 6: Aspect Analysis
            progress_bar.progress(95)
            status_text.text("Step 6: Analisis aspek...")
            
            aspect_results = extract_aspects(normalized_text, aspects, positive_words, negative_words)
            
            # Complete
            progress_bar.progress(100)
            status_text.text("✅ Analisis selesai!")
            
            # Results
            st.markdown('<div class="process-header">🎯 Hasil Analisis</div>', unsafe_allow_html=True)
            
            # Overall sentiment
            sentiment_color = "positive" if sentiment == "positif" else "negative" if sentiment == "negatif" else "neutral"
            
            col_result1, col_result2 = st.columns(2)
            
            with col_result1:
                st.markdown('<div class="result-box">', unsafe_allow_html=True)
                st.markdown("**📊 Sentimen Keseluruhan:**")
                st.markdown(f'<div class="{sentiment_color}">🎯 {sentiment.upper()}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Word analysis
            with col_result2:
                st.markdown('<div class="result-box">', unsafe_allow_html=True)
                st.markdown("**📈 Analisis Kata:**")
                
                words = normalized_text.split() if normalized_text else []
                pos_words = [word for word in words if word in positive_words]
                neg_words = [word for word in words if word in negative_words]
                
                if pos_words:
                    st.markdown(f'✅ **Kata Positif:** {", ".join(pos_words)}')
                if neg_words:
                    st.markdown(f'❌ **Kata Negatif:** {", ".join(neg_words)}')
                if not pos_words and not neg_words:
                    st.markdown("⚪ Tidak ditemukan kata dengan indikasi sentimen yang kuat")
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Aspect analysis
            if aspect_results:
                st.markdown('<div class="process-header">🎪 Analisis Berdasarkan Aspek</div>', unsafe_allow_html=True)
                
                aspect_cols = st.columns(len(aspect_results))
                
                for i, (aspect_name, aspect_data) in enumerate(aspect_results.items()):
                    with aspect_cols[i]:
                        aspect_sentiment = aspect_data['sentiment']
                        aspect_color = "positive" if aspect_sentiment == "positif" else "negative" if aspect_sentiment == "negatif" else "neutral"
                        
                        st.markdown('<div class="result-box">', unsafe_allow_html=True)
                        st.markdown(f"**🏷️ {aspect_name.replace('_', ' ').title()}**")
                        st.markdown(f'<div class="{aspect_color}">Sentimen: {aspect_sentiment.upper()}</div>', unsafe_allow_html=True)
                        st.markdown(f"🔑 Keywords: {', '.join(aspect_data['keywords'])}")
                        st.markdown(f"📊 Confidence: {aspect_data['confidence']:.2%}")
                        st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("🔍 Tidak ditemukan aspek spesifik dalam komentar ini.")
            
            # Summary
            st.markdown('<div class="process-header">📋 Ringkasan</div>', unsafe_allow_html=True)
            
            summary_col1, summary_col2, summary_col3 = st.columns(3)
            
            with summary_col1:
                st.metric("📝 Jumlah Kata", len(tokens))
            
            with summary_col2:
                st.metric("🎯 Sentimen", sentiment.title())
            
            with summary_col3:
                st.metric("🏷️ Aspek Terdeteksi", len(aspect_results))
    
    elif analyze_button and not user_input.strip():
        st.warning("⚠️ Silakan masukkan komentar terlebih dahulu!")
    
    # Information section
    with st.expander("ℹ️ Informasi Model"):
        st.markdown("""
        **Model Sentiment Analysis ini menggunakan:**
        - 🧹 **Text Cleaning**: Pembersihan URL, mention, hashtag, dan karakter khusus
        - 🔄 **Normalisasi**: Stemming bahasa Indonesia dan Inggris, penghapusan stopwords
        - 🎯 **Rule-based Classification**: Menggunakan kamus kata positif dan negatif
        - 🏷️ **Aspect-based Analysis**: Analisis sentimen berdasarkan 5 aspek utama:
          - 🚗 Aksesibilitas
          - 🏢 Fasilitas  
          - 🧽 Kebersihan
          - 🌴 Keindahan Alam
          - 🛎️ Layanan
        """)
    
    # Example section
    with st.expander("💡 Contoh Komentar"):
        st.markdown("""
        **Contoh komentar yang bisa dianalisis:**
        
        ✅ **Positif**: "Pantai Gili Labak sangat indah dan bersih, pemandangannya amazing banget!"
        
        ❌ **Negatif**: "Aksesnya susah banget, toiletnya kotor, dan mahal"
        
        ⚪ **Netral**: "Pulau kecil di madura yang bisa dikunjungi untuk liburan"
        """)

if __name__ == "__main__":
    main()
