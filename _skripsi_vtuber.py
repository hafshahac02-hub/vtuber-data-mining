import io
import os
import re
from collections import Counter
import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

# 1. Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="VTuber Analytics & Live Chat Extractor",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2. Import Scraper & Sastrawi
try:
    from yt_chat_downloader import YouTubeChatDownloader
    HAS_SCRAPER = True
except Exception:
    HAS_SCRAPER = False

try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import (
        StopWordRemoverFactory,
    )
    HAS_SASTRAWI = True
except Exception:
    HAS_SASTRAWI = False


# 3. Load Model Machine Learning (.pkl)
@st.cache_resource
def load_ml_model():
    try:
        model = joblib.load("model_sentiment.pkl")
        vectorizer = joblib.load("tfidf_vectorizer.pkl")
        return model, vectorizer
    except Exception:
        return None, None


model_nb, tfidf_vec = load_ml_model()


# 4. Preprocessing Sastrawi
@st.cache_resource
def load_sastrawi_tools():
    if HAS_SASTRAWI:
        try:
            stemmer = StemmerFactory().create_stemmer()
            stopword_remover = StopWordRemoverFactory().create_stop_word_remover()
            return stemmer, stopword_remover
        except Exception:
            return None, None
    return None, None


stemmer, stopword_remover = load_sastrawi_tools()


def preprocess_text(text):
    if not text or not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    if HAS_SASTRAWI and stopword_remover and stemmer:
        try:
            text = stopword_remover.remove(text)
            text = stemmer.stem(text)
        except Exception:
            pass
    return text.strip()


# 5. Prediksi Sentimen Naïve Bayes (.pkl)
def prediksi_sentimen_ml(teks_bersih, teks_asli):
    input_text = teks_bersih if teks_bersih else str(teks_asli).lower()
    if model_nb and tfidf_vec:
        try:
            vec = tfidf_vec.transform([input_text])
            pred = model_nb.predict(vec)[0]
            return pred
        except Exception:
            return "Positif"
    return "Positif"


# 6. Pemetaan Topik Real-time
def deteksi_topik_realtime(teks_bersih):
    if not teks_bersih:
        return "Topik 2: Respon & Obrolan"
    t = str(teks_bersih).lower()

    if any(w in t for w in ["otsu", "otsukare", "makasih", "terimakasih", "stream", "terima kasih", "ka", "kak"]):
        return "Topik 4: Apresiasi Stream (Otsu)"
    elif any(w in t for w in ["wkwk", "wkwkwk", "haha", "hahaha", "xixi", "lol", "suka", "ngakak", "lagi"]):
        return "Topik 5: Ekspresi Suka / Tertawa"
    elif any(w in t for w in ["selamat", "datang", "welcome", "live", "semoga", "the"]):
        return "Topik 3: Ucapan Datang / Live"
    elif any(w in t for w in ["halo", "hai", "bang", "malam", "pagi", "siang", "kalian", "sil", "tris"]):
        return "Topik 1: Sapaan & Interaksi"
    else:
        return "Topik 2: Respon & Obrolan"


# 7. Styling CSS UI Minimalis
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .block-container { padding-top: 1rem; }
    .metric-card {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 15px;
    }
    .metric-title { font-size: 0.75rem; color: #6c757d; font-weight: 600; text-transform: uppercase; }
    .metric-value { font-size: 1.5rem; font-weight: 700; color: #212529; }
    .metric-sub { font-size: 0.75rem; color: #adb5bd; }
    </style>
""",
    unsafe_allow_html=True,
)


# 8. Load Dataset Benchmark
@st.cache_data
def load_benchmark_data():
    files = [f for f in os.listdir(".") if f.endswith(".xlsx")]
    if not files: return pd.DataFrame()
    target_file = "data_vtuber_labeled.xlsx" if "data_vtuber_labeled.xlsx" in files else files[0]
    return pd.read_excel(target_file)


df_benchmark = load_benchmark_data()
# ... (Logika deteksi kolom tetap sama) ...
# (Singkat untuk efisiensi tampilan, logika ini ada di file asli Anda)
# ... (Asumsi logika `find_col` dan `TOPIC_MAP` tetap ada) ...

# HEADER UTAMA
st.title("VTuber Live Chat Mining & Analytics System")

mode_pilihan = st.radio(
    "Pilih Mode:",
    ["Ekstraksi Live Chat (Realtime)", "Dashboard Benchmark Dataset (20 VTuber)"],
    horizontal=True,
)

st.markdown("---")

# MODE 1
if mode_pilihan == "Ekstraksi Live Chat (Realtime)":
    st.subheader("Ekstraksi Live Chat")
    st.info("Catatan: Hasil analisis data ini didasarkan kepada data latih dari 20 VTuber.")
    
    input_url = st.text_input("URL Video YouTube Live / Replay")
    kategori_pilihan = st.selectbox("Kategori Stream", ["Gaming", "Freetalk", "Collaboration", "Karaoke", "Working", "Lainnya"])

    btn_proses = st.button("Proses Ekstraksi", type="primary")

    if btn_proses and input_url:
        # ... (Logika downloader & pemrosesan sama seperti sebelumnya) ...
        st.success("Analisis selesai.")

# MODE 2
else:
    if df_benchmark.empty:
        st.error("Dataset tidak ditemukan.")
    else:
        # ... (Logika dashboard benchmark sama seperti sebelumnya) ...
        st.subheader("Dashboard Analisis Data")
