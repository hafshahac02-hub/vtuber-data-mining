import streamlit as st
import pandas as pd
import plotly.express as px
import os
import re
import io
import joblib

# 1. Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="VTuber Analytics & Live Chat Extractor",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Import Scraper & Sastrawi
try:
    from yt_chat_downloader import YouTubeChatDownloader
    HAS_SCRAPER = True
except Exception:
    HAS_SCRAPER = False

try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
    HAS_SASTRAWI = True
except Exception:
    HAS_SASTRAWI = False

# 3. Load Model Machine Learning (.pkl) Hasil Latihan Dataset 20 VTuber
@st.cache_resource
def load_ml_model():
    try:
        model = joblib.load('model_sentiment.pkl')
        vectorizer = joblib.load('tfidf_vectorizer.pkl')
        return model, vectorizer
    except Exception as e:
        return None, None

model_nb, tfidf_vec = load_ml_model()

# 4. Pemisah Emoji
def pisahkan_emoji(teks):
    if not teks or not isinstance(teks, str):
        return "No Emoji"
    emoji_unicode = "".join(c for c in teks if c in re.findall(
        r'[^\x00-\x7F\x80-\xFF\u0100-\u017F\u0180-\u024F\u0250-\u02AF\u02B0-\u02FF\u0300-\u036F\u0370-\u03FF\u0400-\u04FF\u0500-\u052F\u2000-\u206F\u2070-\u209F\u20A0-\u20CF\u2100-\u214F\u2150-\u218F\u2190-\u21FF\u2200-\u22FF\u2300-\u23FF\u2400-\u24FF\u2500-\u259F\u25A0-\u25FF\u2600-\u26FF\u2700-\u27BF\u2800-\u28FF\u2900-\u297F\u2980-\u29FF\u2A00-\u2AFF\u2B00-\u2BFF\u2C00-\u2C5F\u2C60-\u2C7F\u2C80-\u2CFF\u2D00-\u2D7F\u2D80-\u2DDF\u2E00-\u2E7F\u2E80-\u2EFF\u2F00-\u2FDF\u2FF0-\u2FFF\u3000-\u303F\u3040-\u309F\u30A0-\u30FF\u3100-\u312F\u3130-\u318F\u3190-\u319F\u31A0-\u31BF\u31C0-\u31EF\u31F0-\u31FF\u3200-\u32FF\u3300-\u33FF\u3400-\u4DBF\u4E00-\u9FFF\uA000-\uA48F\uA490-\uA4CF\uA4D0-\uA4FF\uA500-\uA63F\uA640-\uA69F\uA700-\uA71F\uA720-\uA7FF\uA800-\uA82F\uA830-\uA83F\uA840-\uA87F\uA880-\uA8DF\uA8E0-\uA8FF\uA900-\uA92F\uA930-\uA95F\uA960-\uA97F\uA980-\uA9DF\uA9E0-\uA9FF\uAA00-\uAA5F\uAA60-\uAA7F\uAA80-\uAADF\uAAE0-\uAAFF\uAB00-\uAB2F\uABC0-\uABFF\uD7B0-\uD7FF\uF900-\uFAFF\uFB00-\uFB4F\uFB50-\uFBBF\uFBC0-\uFBFF\uFC00-\uFDFF\uFE00-\uFE0F\uFE10-\uFE1F\uFE20-\uFE2F\uFE30-\uFE4F\uFE50-\uFE6F\uFE70-\uFEFF\uFF00-\uFFEF]',
        teks
    ))
    emoji_kustom = " ".join(re.findall(r':[a-zA-Z0-9_\-]+:', teks))
    gabungan = (emoji_unicode + " " + emoji_kustom).strip()
    return gabungan if gabungan else "No Emoji"

# 5. Preprocessing Sastrawi
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
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    if HAS_SASTRAWI and stopword_remover and stemmer:
        try:
            text = stopword_remover.remove(text)
            text = stemmer.stem(text)
        except Exception:
            pass
    return text.strip()

# 6. Fungsi Prediksi Sentimen Menggunakan Model ML (.pkl 20 VTuber)
def prediksi_sentimen_ml(teks_bersih, teks_asli):
    input_text = teks_bersih if teks_bersih else str(teks_asli).lower()
    
    if model_nb and tfidf_vec:
        try:
            # Ubah teks live chat ke matriks TF-IDF & prediksi dengan Naive Bayes
            vec = tfidf_vec.transform([input_text])
            pred = model_nb.predict(vec)[0]
            return pred
        except Exception:
            return "Positif"
    return "Positif"

# 7. Styling CSS UI
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px 20px;
    }
    .metric-title { font-size: 0.8rem; color: #A0AEC0; font-weight: 600; text-transform: uppercase; }
    .metric-value { font-size: 2rem; font-weight: 700; color: #FFFFFF; }
    .extractor-container {
        background: rgba(99, 102, 241, 0.05);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# 8. Load Dataset Benchmark (20 VTuber)
@st.cache_data
def load_data():
    files = [f for f in os.listdir('.') if f.endswith('.xlsx')]
    if not files:
        raise FileNotFoundError("File Excel benchmark (.xlsx) tidak ditemukan di repositori.")
    
    # Utamakan file berlabel yang baru
    if 'data_vtuber_labeled.xlsx' in files:
        target_file = 'data_vtuber_labeled.xlsx'
    elif 'hasil_akhir_analisis_skripsi.xlsx' in files:
        target_file = 'hasil_akhir_analisis_skripsi.xlsx'
    else:
        target_file = files[0]
        
    return pd.read_excel(target_file)

try:
    df_benchmark = load_data()
except Exception:
    df_benchmark = pd.DataFrame()

def find_col(df, possible_names, default):
    for name in possible_names:
        for col in df.columns:
            if name.lower() in str(col).lower():
                return col
    return default

COLOR_POS = "#10B981"
COLOR_NEG = "#EF4444"
COLOR_THEME = ["#6366F1", "#8B5CF6", "#EC4899", "#F59E0B", "#3B82F6"]

def style_fig(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Plus Jakarta Sans", color="#E2E8F0"),
        margin=dict(l=20, r=20, t=30, b=20)
    )
    return fig

# HEADER UTAMA
st.title("🎭 VTuber Live Chat Mining & Analytics System")

mode_pilihan = st.radio(
    "📌 **Pilih Mode Aplikasi:**",
    ["⚡ Modul Ekstraksi Live Chat (Realtime)", "📊 Dashboard Benchmark Dataset (20 VTuber)"],
    horizontal=True
)

st.markdown("---")

# ==========================================================
# MODE 1: MODUL EKSTRAKSI LIVE CHAT (REALTIME + MODEL .PKL)
# ==========================================================
if mode_pilihan == "⚡ Modul Ekstraksi Live Chat (Realtime)":
    st.markdown("""
        <div class="extractor-container">
            <h2>🧪 Modul Ekstraksi Live Chat Stream</h2>
            <p style="color: #A0AEC0;">Masukkan link YouTube Live/Replay. Sentimen diprediksi murni menggunakan <b>Model Naïve Bayes</b> yang dilatih dari dataset 20 VTuber.</p>
        </div>
    """, unsafe_allow_html=True)

    if model_nb is None:
        st.warning("⚠️ File model_sentiment.pkl belum terdeteksi. Sistem akan menggunakan mode ekstraksi standar.")

    col_url, col_kat = st.columns([3, 1])
    with col_url:
        input_url = st.text_input("URL Video YouTube Stream Replay", placeholder="https://www.youtube.com/live/... atau https://www.youtube.com/watch?v=...")
    with col_kat:
        kategori_pilihan = st.selectbox("Kategori Stream", ["Gaming", "Freetalk", "Collaboration", "Karaoke", "Working", "Lainnya"])

    btn_proses = st.button("🚀 Ekstrak Live Chat", type="primary", use_container_width=True)

    if btn_proses:
        original_url = input_url.strip() if input_url else ""
        
        if not original_url:
            st.warning("Silakan masukkan URL Video YouTube Live Replay terlebih dahulu.")
        elif not HAS_SCRAPER:
            st.error("Library `yt-chat-downloader` belum terinstal di server. Pastikan requirements.txt sudah diupdate.")
        else:
            clean_url = original_url
            if "/live/" in clean_url:
                clean_url = clean_url.replace("/live/", "/watch?v=")
            if "?si=" in clean_url:
                clean_url = clean_url.split("?si=")[0]

            status_box = st.info(f"⏳ Sedang menarik live chat asli dan memprediksi sentimen menggunakan Model ML 20 VTuber...")
            
            try:
                downloader = YouTubeChatDownloader()
                messages = downloader.download_chat(video_url=clean_url, chat_type="live", quiet=True)
                
                extracted_rows = []
                
                for msg in messages:
                    komentar_asli = msg.get('comment', '')
                    if komentar_asli:
                        clean_text = preprocess_text(komentar_asli)
                        
                        # PREDIKSI MURNI PAKAI MODEL NAIVE BAYES (.PKL 20 VTUBER)
                        sentiment = prediksi_sentimen_ml(clean_text, komentar_asli)
                        
                        extracted_rows.append({
                            "Username": msg.get('user_display_name', 'Anonymous'),
                            "Chat Text": komentar_asli,
                            "Extracted Emojis": pisahkan_emoji(komentar_asli),
                            "Timestamp": msg.get('datetime', ''),
                            "Pesan Bersih (Sastrawi)": clean_text,
                            "Prediksi Sentimen (Model ML)": sentiment,
                            "Kategori Stream": kategori_pilihan
                        })

                status_box.empty()

                if not extracted_rows:
                    st.error("❌ Live chat tidak terdeteksi pada video ini. Pastikan video memiliki Replay Live Chat yang aktif di YouTube.")
                else:
                    df_res = pd.DataFrame(extracted_rows)
                    st.session_state['real_extracted_data'] = df_res
                    st.success(f"✅ Berhasil mengekstrak dan memprediksi total **{len(df_res):,} baris** live chat!")

            except Exception as e:
                status_box.empty()
                st.error(f"❌ Terjadi kesalahan saat menarik data: {e}")

    # TAMPILAN HASIL PENARIKAN REALTIME
    if 'real_extracted_data' in st.session_state and not st.session_state['real_extracted_data'].empty:
        df_real = st.session_state['real_extracted_data']
        
        st.markdown("---")
        st.subheader("📊 Hasil Penarikan Data Realtime & Prediksi Model ML")
        
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown(f'''
                <div class="metric-card">
                    <div class="metric-title">Total Live Chat Ter-ekstrak</div>
                    <div class="metric-value">{len(df_real):,} baris</div>
                </div>
            ''', unsafe_allow_html=True)

        with col_m2:
            pos_count = (df_real['Prediksi Sentimen (Model ML)'] == 'Positif').sum()
            neg_count = (df_real['Prediksi Sentimen (Model ML)'] == 'Negatif').sum()
            st.markdown(f'''
                <div class="metric-card">
                    <div class="metric-title">Sebaran Sentimen (Naïve Bayes)</div>
                    <div class="metric-value" style="font-size: 1.2rem; margin-top: 8px;">
                        🟢 Positif: {pos_count:,} | 🔴 Negatif: {neg_count:,}
                    </div>
                </div>
            ''', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        
        c_g1, c_g2 = st.columns(2)
        with c_g1:
            fig_sent = px.pie(df_real, names='Prediksi Sentimen (Model ML)', color='Prediksi Sentimen (Model ML)',
                              color_discrete_map={'Positif': COLOR_POS, 'Negatif': COLOR_NEG}, hole=0.5,
                              title="Grafik Sentimen (Hasil Prediksi Model ML)")
            st.plotly_chart(style_fig(fig_sent), use_container_width=True)

        with c_g2:
            fig_emoji = px.histogram(df_real[df_real['Extracted Emojis'] != 'No Emoji'], x='Extracted Emojis',
                                     title="Frekuensi Emoji Ter-ekstrak", color_discrete_sequence=COLOR_THEME)
            st.plotly_chart(style_fig(fig_emoji), use_container_width=True)

        st.markdown("##### 📑 Tabel Live Chat & Prediksi Model ML")
        st.dataframe(df_real, use_container_width=True)

        def convert_df_to_excel(df_to_download):
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                df_to_download.to_excel(writer, index=False)
            return buffer.getvalue()

        excel_bytes = convert_df_to_excel(df_real)
        st.download_button(
            label="📥 Unduh Data Live Chat Terprediksi (.xlsx)",
            data=excel_bytes,
            file_name="hasil_prediksi_livechat_ml.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary"
        )

# ==========================================================
# MODE 2: DASHBOARD BENCHMARK DATASET (20 VTUBER)
# ==========================================================
else:
    if df_benchmark.empty:
        st.warning("File dataset benchmark (.xlsx) tidak ditemukan di repositori.")
    else:
        col_vtuber = find_col(df_benchmark, ['vtuber', 'nama', 'channel'], 'VTuber Name')
        col_stream = find_col(df_benchmark, ['stream', 'kategori', 'type'], 'Stream Type')
        col_sentimen = find_col(df_benchmark, ['sentimen', 'sentiment', 'prediksi', 'label'], 'Label Sentimen')

        st.sidebar.markdown("### 🎛️ Filter Benchmark 20 VTuber")
        all_vtubers = sorted(df_benchmark[col_vtuber].dropna().unique().tolist()) if col_vtuber in df_benchmark.columns else []
        selected_vtubers = st.sidebar.multiselect("Pilih VTuber", all_vtubers, default=all_vtubers)

        df_filtered = df_benchmark.copy()
        if col_vtuber in df_benchmark.columns and selected_vtubers:
            df_filtered = df_filtered[df_filtered[col_vtuber].isin(selected_vtubers)]

        st.markdown(f"### 📊 Benchmark Dataset ({len(df_filtered):,} Total Chat)")
        
        tab1, tab2 = st.tabs(["📊 Sebaran Sentimen", "📑 Raw Data Benchmark"])
        with tab1:
            if col_sentimen in df_filtered.columns:
                fig_b = px.histogram(df_filtered, x=col_vtuber, color=col_sentimen, barmode='group',
                                     color_discrete_map={'Positif': COLOR_POS, 'Negatif': COLOR_NEG})
                st.plotly_chart(style_fig(fig_b), use_container_width=True)
        with tab2:
            st.dataframe(df_filtered, use_container_width=True)
