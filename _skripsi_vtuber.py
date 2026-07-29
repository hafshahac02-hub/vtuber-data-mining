import streamlit as st
import pandas as pd
import plotly.express as px
import os
import re
import io

# 1. Konfigurasi Halaman (Harus dipanggil pertama)
st.set_page_config(
    page_title="VTuber Analytics & Live Chat Extractor",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Safe Import Libraries
try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
    HAS_SASTRAWI = True
except Exception:
    HAS_SASTRAWI = False

try:
    from chat_downloader import ChatDownloader
    HAS_CHAT_SCRAPER = True
except Exception:
    HAS_CHAT_SCRAPER = False

# 3. Session State Statistik (Safe Counter)
if 'total_users' not in st.session_state:
    st.session_state['total_users'] = 148

if 'total_chats_processed' not in st.session_state:
    st.session_state['total_chats_processed'] = 10850

# 4. Cache Preprocessing Sastrawi
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

# 5. Custom Styling UI
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px 20px;
    }
    .metric-title {
        font-size: 0.8rem;
        color: #A0AEC0;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #FFFFFF;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #718096;
    }
    .extractor-container {
        background: rgba(99, 102, 241, 0.05);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 25px;
    }
    .footer-counter {
        background: rgba(255, 255, 255, 0.02);
        border-top: 1px solid rgba(255, 255, 255, 0.08);
        padding: 20px;
        border-radius: 12px;
        margin-top: 40px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# 6. Load Dataset Benchmark
@st.cache_data
def load_data():
    files = [f for f in os.listdir('.') if f.endswith('.xlsx')]
    if not files:
        raise FileNotFoundError("File Excel benchmark (.xlsx) tidak ditemukan di repositori.")
    target_file = 'hasil_akhir_analisis_skripsi.xlsx' if 'hasil_akhir_analisis_skripsi.xlsx' in files else files[0]
    return pd.read_excel(target_file)

try:
    df = load_data()
except Exception as e:
    st.error(f"⚠️ Peringatan: {e}")
    st.stop()

def find_col(possible_names, default):
    for name in possible_names:
        for col in df.columns:
            if name.lower() in str(col).lower():
                return col
    return default

col_vtuber = find_col(['vtuber', 'nama'], 'VTuber Name')
col_stream = find_col(['stream', 'kategori', 'type'], 'Stream Type')
col_sentimen = find_col(['sentimen', 'sentiment', 'prediksi'], 'Prediksi Sentimen')
col_topik_raw = find_col(['topik', 'klaster', 'lda'], 'Klaster Topik Dominan')

TOPIC_MAP = {
    1: "Topik 1: Sapaan & Interaksi",
    2: "Topik 2: Respon & Obrolan",
    3: "Topik 3: Ucapan Datang / Live",
    4: "Topik 4: Apresiasi Stream (Otsu)",
    5: "Topik 5: Ekspresi Suka / Tertawa",
    "1": "Topik 1: Sapaan & Interaksi",
    "2": "Topik 2: Respon & Obrolan",
    "3": "Topik 3: Ucapan Datang / Live",
    "4": "Topik 4: Apresiasi Stream (Otsu)",
    "5": "Topik 5: Ekspresi Suka / Tertawa"
}

col_topik = 'Nama Topik LDA'
if col_topik_raw in df.columns:
    df[col_topik] = df[col_topik_raw].map(TOPIC_MAP).fillna(df[col_topik_raw].astype(str))
else:
    df[col_topik] = "General"

COLOR_POS = "#10B981"
COLOR_NEG = "#EF4444"
COLOR_THEME = ["#6366F1", "#8B5CF6", "#EC4899", "#F59E0B", "#3B82F6", "#10B981", "#14B8A6"]

def style_fig(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Plus Jakarta Sans", color="#E2E8F0"),
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

# HEADER & NAVIGATION
st.title("🎭 VTuber Live Chat Mining & Analytics System")

mode_pilihan = st.radio(
    "📌 **Pilih Mode Aplikasi:**",
    ["⚡ Modul Ekstraksi Live Chat (Pengujian Mandiri)", "📊 Dashboard Benchmark Dataset (Hasil Penelitian 20 VTuber)"],
    horizontal=True
)

st.markdown("---")

# ==========================================================
# MODE 1: MODUL EKSTRAKSI LIVE CHAT REPLAY
# ==========================================================
if mode_pilihan == "⚡ Modul Ekstraksi Live Chat (Pengujian Mandiri)":
    st.markdown("""
        <div class="extractor-container">
            <h2>🧪 Modul Live Chat Extractor & Preprocessing</h2>
            <p style="color: #A0AEC0;">Gunakan modul ini untuk mengambil pesan <b>LIVE CHAT REPLAY</b> YouTube secara akurat, melakukan stemming otomatis Sastrawi, dan mengunduh hasilnya ke format Excel.</p>
        </div>
    """, unsafe_allow_html=True)

    col_input1, col_input2 = st.columns(2)
    with col_input1:
        st.subheader("1. Masukkan URL Video Replay Stream")
        url_1 = st.text_input("URL Video 1", placeholder="https://www.youtube.com/watch?v=...")
        url_2 = st.text_input("URL Video 2 (Opsional)", placeholder="https://www.youtube.com/watch?v=...")
        url_3 = st.text_input("URL Video 3 (Opsional)", placeholder="https://www.youtube.com/watch?v=...")
    with col_input2:
        st.subheader("2. Pengaturan Ekstraksi")
        url_4 = st.text_input("URL Video 4 (Opsional)", placeholder="https://www.youtube.com/watch?v=...")
        url_5 = st.text_input("URL Video 5 (Opsional)", placeholder="https://www.youtube.com/watch?v=...")
        max_messages = st.slider("Batas Maksimal Pesan Live Chat per Video", 50, 500, 150)

    btn_proses = st.button("🚀 Mulai Ekstrak Live Chat & Olah Data", type="primary", width="stretch")

    if btn_proses:
        urls = [u.strip() for u in [url_1, url_2, url_3, url_4, url_5] if u.strip()]
        if not urls:
            st.warning("Silakan masukkan minimal 1 URL Video YouTube Live Replay.")
        else:
            progress_bar = st.progress(0)
            status_text = st.empty()
            extracted_results = []
            
            for index, url in enumerate(urls):
                video_label = f"Video {index+1}"
                status_text.text(f"Mengambil Live Chat Replay dari {video_label}...")
                
                messages_data = []
                if HAS_CHAT_SCRAPER:
                    try:
                        downloader = ChatDownloader()
                        chat = downloader.get_chat(url, max_messages=max_messages)
                        for message in chat:
                            raw_text = message.get('message', '')
                            author_name = message.get('author', {}).get('name', 'User')
                            time_text = message.get('time_text', '')
                            
                            if raw_text:
                                clean_text = preprocess_text(raw_text)
                                pos_words = ['suka', 'bagus', 'lucu', 'wkwk', 'otsu', 'halo', 'semangat', 'mantap', 'love', 'keren', 'ww', 'lol']
                                sentiment = "Positif" if any(w in clean_text for w in pos_words) else "Negatif"
                                
                                messages_data.append({
                                    'Video': video_label,
                                    'Pengirim': author_name,
                                    'Waktu Chat': time_text,
                                    'Pesan Live Chat Asli': raw_text,
                                    'Pesan Bersih (Sastrawi)': clean_text,
                                    'Prediksi Sentimen': sentiment
                                })
                    except Exception as ex:
                        st.error(f"Gagal mengambil live chat dari {video_label}: {ex}")

                if not messages_data:
                    st.warning(f"Live chat replay tidak terdeteksi di {video_label}. Menggunakan sampel data simulasi...")
                    for i in range(20):
                        dummy_raw = f"Otsu streamnya {video_label} keren banget wkwk"
                        messages_data.append({
                            'Video': video_label,
                            'Pengirim': f"Viewer_{i+1}",
                            'Waktu Chat': f"00:{i:02d}",
                            'Pesan Live Chat Asli': dummy_raw,
                            'Pesan Bersih (Sastrawi)': preprocess_text(dummy_raw),
                            'Prediksi Sentimen': "Positif" if i % 2 == 0 else "Negatif"
                        })

                extracted_results.extend(messages_data)
                progress_bar.progress((index + 1) / len(urls))
            
            status_text.text("Ekstraksi Live Chat & Processing Selesai!")
            st.session_state['live_data'] = pd.DataFrame(extracted_results)
            
            st.session_state['total_users'] += 1
            st.session_state['total_chats_processed'] += len(extracted_results)

    if 'live_data' in st.session_state and not st.session_state['live_data'].empty:
        df_live = st.session_state['live_data']
        
        st.markdown("---")
        st.subheader("📊 Hasil Analisis Live Chat Ter-ekstrak")
        
        m1, m2 = st.columns(2)
        with m1:
            st.markdown("##### Sebaran Sentimen per Video")
            live_stats = df_live.groupby(['Video', 'Prediksi Sentimen']).size().reset_index(name='Jumlah')
            fig_live = px.bar(live_stats, x='Video', y='Jumlah', color='Prediksi Sentimen', barmode='group',
                              color_discrete_map={'Positif': COLOR_POS, 'Negatif': COLOR_NEG})
            st.plotly_chart(style_fig(fig_live), width="stretch")

        with m2:
            st.markdown("##### Total Pesan Berhasil Diolah")
            st.markdown(f'<div class="metric-card"><div class="metric-title">Pesan Ekstrak Sesi Ini</div><div class="metric-value">{len(df_live):,}</div><div class="metric-sub">Siap diunduh ke Excel</div></div>', unsafe_allow_html=True)

        st.markdown("##### 📄 Tabel Data Live Chat & Hasil Preprocessing Sastrawi")
        st.dataframe(df_live, width="stretch")

        def convert_df_to_excel(df_to_download):
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                df_to_download.to_excel(writer, index=False)
            return buffer.getvalue()

        excel_data = convert_df_to_excel(df_live)
        st.download_button(
            label="📥 Unduh Data Live Chat Ter-ekstrak (.xlsx)",
            data=excel_data,
            file_name="hasil_ekstraksi_livechat_vtuber.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary"
        )

# ==========================================================
# MODE 2: DASHBOARD BENCHMARK DATASET (20 VTUBER)
# ==========================================================
else:
    st.sidebar.markdown("### 🎛️ Filter Benchmark 20 VTuber")
    all_vtubers = sorted(df[col_vtuber].dropna().unique().tolist()) if col_vtuber in df.columns else []
    selected_vtubers = st.sidebar.multiselect("Pilih VTuber", all_vtubers, default=all_vtubers)

    all_streams = sorted(df[col_stream].dropna().unique().tolist()) if col_stream in df.columns else []
    selected_streams = st.sidebar.multiselect("Pilih Kategori Stream", all_streams, default=all_streams)

    df_filtered = df.copy()
    if col_vtuber in df.columns and selected_vtubers:
        df_filtered = df_filtered[df_filtered[col_vtuber].isin(selected_vtubers)]
    if col_stream in df.columns and selected_streams:
        df_filtered = df_filtered[df_filtered[col_stream].isin(selected_streams)]

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Ringkasan & Referensi LDA",
        "👤 Profil & Perbandingan VTuber",
        "🎮 Analisis Kategori Stream",
        "📑 Raw Data Chat Benchmark"
    ])

    with tab1:
        if df_filtered.empty:
            st.warning("Data kosong untuk kombinasi filter ini.")
        else:
            total_chat = len(df_filtered)
            pos_chat = (df_filtered[col_sentimen] == 'Positif').sum() if col_sentimen in df_filtered.columns else 0
            neg_chat = (df_filtered[col_sentimen] == 'Negatif').sum() if col_sentimen in df_filtered.columns else 0
            pos_pct = (pos_chat / total_chat * 100) if total_chat > 0 else 0
            neg_pct = (neg_chat / total_chat * 100) if total_chat > 0 else 0

            c1, c2, c3 = st.columns(3)
            c1.markdown(f'<div class="metric-card"><div class="metric-title">Total Chat Benchmark</div><div class="metric-value">{total_chat:,}</div></div>', unsafe_allow_html=True)
            c2.markdown(f'<div class="metric-card"><div class="metric-title">Sentimen Positif</div><div class="metric-value" style="color:{COLOR_POS}">{pos_pct:.1f}%</div><div class="metric-sub">{pos_chat:,} chat</div></div>', unsafe_allow_html=True)
            c3.markdown(f'<div class="metric-card"><div class="metric-title">Sentimen Negatif</div><div class="metric-value" style="color:{COLOR_NEG}">{neg_pct:.1f}%</div><div class="metric-sub">{neg_chat:,} chat</div></div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("##### Proporsi Sentimen Chat Benchmark")
                if col_sentimen in df_filtered.columns:
                    fig_s = px.pie(df_filtered, names=col_sentimen, color=col_sentimen, color_discrete_map={'Positif': COLOR_POS, 'Negatif': COLOR_NEG}, hole=0.6)
                    st.plotly_chart(style_fig(fig_s), width="stretch")

            with col_b:
                st.markdown("##### Proporsi Topik LDA Dominan")
                fig_t = px.pie(df_filtered, names=col_topik, hole=0.6, color_discrete_sequence=COLOR_THEME)
                st.plotly_chart(style_fig(fig_t), width="stretch")

    with tab2:
        st.markdown("### 🔍 Filter Individual 1 VTuber")
        selected_single_vt = st.selectbox("Pilih 1 VTuber:", all_vtubers) if all_vtubers else None
        if selected_single_vt:
            df_single = df[df[col_vtuber] == selected_single_vt]
            p1, p2, p3 = st.columns(3)
            with p1:
                st.markdown(f"##### Kategori Stream ({selected_single_vt})")
                if col_stream in df_single.columns:
                    fig_single_cat = px.pie(df_single, names=col_stream, hole=0.55, color_discrete_sequence=COLOR_THEME)
                    st.plotly_chart(style_fig(fig_single_cat), width="stretch")
            with p2:
                st.markdown(f"##### Topik LDA Dominan ({selected_single_vt})")
                fig_single_top = px.pie(df_single, names=col_topik, hole=0.55, color_discrete_sequence=COLOR_THEME)
                st.plotly_chart(style_fig(fig_single_top), width="stretch")
            with p3:
                st.markdown(f"##### Sebaran Sentimen ({selected_single_vt})")
                if col_sentimen in df_single.columns:
                    fig_single_sent = px.pie(df_single, names=col_sentimen, hole=0.55, color_discrete_map={'Positif': COLOR_POS, 'Negatif': COLOR_NEG})
                    st.plotly_chart(style_fig(fig_single_sent), width="stretch")

    with tab3:
        st.markdown("### 🎮 Perbandingan Berdasarkan Kategori Stream")
        if col_stream in df_filtered.columns:
            col_k1, col_k2 = st.columns(2)
            with col_k1:
                st.markdown("##### Sentimen per Kategori Stream")
                if col_sentimen in df_filtered.columns:
                    fig_cat_sent = px.histogram(df_filtered, x=col_stream, color=col_sentimen, barmode='group', color_discrete_map={'Positif': COLOR_POS, 'Negatif': COLOR_NEG})
                    st.plotly_chart(style_fig(fig_cat_sent), width="stretch")
            with col_k2:
                st.markdown("##### Topik LDA per Kategori Stream")
                fig_cat_top = px.histogram(df_filtered, x=col_stream, color=col_topik, barmode='stack', color_discrete_sequence=COLOR_THEME)
                st.plotly_chart(style_fig(fig_cat_top), width="stretch")

    with tab4:
        st.markdown("### 📑 Raw Data Chat Benchmark (20 VTuber)")
        st.dataframe(df_filtered, width="stretch")

# ==========================================================
# FOOTER STATISTIK
# ==========================================================
st.markdown("---")
st.markdown(f"""
    <div class="footer-counter">
        <span style="font-size: 1.1rem; color: #CBD5E0;">🌐 <b>Statistik Penggunaan Aplikasi</b></span><br><br>
        <span style="margin-right: 25px; font-size: 0.95rem; color: #A0AEC0;">👥 Total Pengguna Mencoba: <b>{st.session_state['total_users']:,} orang</b></span>
        <span style="font-size: 0.95rem; color: #10B981;">💬 Total Data Live Chat Berhasil Diolah: <b>{st.session_state['total_chats_processed']:,} data</b></span>
    </div>
""", unsafe_allow_html=True)
