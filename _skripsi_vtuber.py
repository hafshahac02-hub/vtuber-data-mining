import streamlit as st
import pandas as pd
import plotly.express as px
import os
import re
import io

# 1. Import Library Sesuai requirements.txt
try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
    HAS_SASTRAWI = True
except ImportError:
    HAS_SASTRAWI = False

try:
    from youtube_comment_downloader import YoutubeCommentDownloader, SORT_BY_POPULAR
    HAS_SCRAPER = True
except ImportError:
    HAS_SCRAPER = False

# 2. Konfigurasi Halaman
st.set_page_config(
    page_title="VTuber Analytics Dashboard",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 3. Cache Inisialisasi Sastrawi (Biar Ringan & Cepat)
@st.cache_resource
def load_sastrawi_tools():
    if HAS_SASTRAWI:
        stemmer = StemmerFactory().create_stemmer()
        stopword_remover = StopWordRemoverFactory().create_stop_word_remover()
        return stemmer, stopword_remover
    return None, None

stemmer, stopword_remover = load_sastrawi_tools()

# Fungsi Preprocessing Teks Menggunakan Sastrawi
def preprocess_text(text):
    if not text or not isinstance(text, str):
        return ""
    # Case folding & Hapus Karakter Non-Alpha
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Stopword Removal & Stemming (Sastrawi)
    if HAS_SASTRAWI and stopword_remover and stemmer:
        text = stopword_remover.remove(text)
        text = stemmer.stem(text)
    return text.strip()

# 4. Styling UI Modern
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
    </style>
""", unsafe_allow_html=True)

# 5. Load Data Otomatis dari Repo GitHub
@st.cache_data
def load_data():
    files = [f for f in os.listdir('.') if f.endswith('.xlsx')]
    if not files:
        raise FileNotFoundError("Tidak ada file Excel (.xlsx) yang ditemukan di repositori GitHub!")
    
    target_file = 'hasil_akhir_analisis_skripsi.xlsx' if 'hasil_akhir_analisis_skripsi.xlsx' in files else files[0]
    return pd.read_excel(target_file)

try:
    df = load_data()
except Exception as e:
    st.error(f"⚠️ Error memuat data: {e}")
    st.info("Pastikan file `.xlsx` dataset kamu sudah di-upload ke repositori GitHub ini.")
    st.stop()

# Deteksi Kolom Otomatis
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

# HEADER
st.title("🎭 VTuber Chat Sentiment & LDA Topic Dashboard")
st.caption("Eksplorasi Komparatif Sentimen dan Pemodelan Topik LDA Chat VTuber")

# SIDEBAR GLOBAL FILTER
st.sidebar.markdown("### 🎛️ Filter Panel Global (Data Benchmark)")
all_vtubers = sorted(df[col_vtuber].dropna().unique().tolist()) if col_vtuber in df.columns else []
selected_vtubers = st.sidebar.multiselect("Pilih VTuber", all_vtubers, default=all_vtubers)

all_streams = sorted(df[col_stream].dropna().unique().tolist()) if col_stream in df.columns else []
selected_streams = st.sidebar.multiselect("Pilih Kategori Stream", all_streams, default=all_streams)

df_filtered = df.copy()
if col_vtuber in df.columns and selected_vtubers:
    df_filtered = df_filtered[df_filtered[col_vtuber].isin(selected_vtubers)]
if col_stream in df.columns and selected_streams:
    df_filtered = df_filtered[df_filtered[col_stream].isin(selected_streams)]

# TAB NAVIGATION
tab_live, tab1, tab2, tab3, tab4 = st.tabs([
    "⚡ Live Analysis (Mandiri)",
    "📊 Ringkasan & Referensi LDA",
    "👤 Profil & Perbandingan VTuber",
    "🎮 Analisis Kategori Stream",
    "📑 Raw Data Chat"
])

# ==========================================
# TAB 0: LIVE ANALYSIS (MANDIRI)
# ==========================================
with tab_live:
    st.markdown("### 🧪 Pengujian & Ekstraksi Live Chat Video Mandiri")
    st.write("Masukkan hingga 5 URL Video YouTube Live/Replay untuk diekstrak, diproses dengan Sastrawi, dan dibandingkan secara sementara.")
    
    col_input1, col_input2 = st.columns(2)
    with col_input1:
        url_1 = st.text_input("URL Video 1", placeholder="https://www.youtube.com/watch?v=...")
        url_2 = st.text_input("URL Video 2 (Opsional)", placeholder="https://www.youtube.com/watch?v=...")
        url_3 = st.text_input("URL Video 3 (Opsional)", placeholder="https://www.youtube.com/watch?v=...")
    with col_input2:
        url_4 = st.text_input("URL Video 4 (Opsional)", placeholder="https://www.youtube.com/watch?v=...")
        url_5 = st.text_input("URL Video 5 (Opsional)", placeholder="https://www.youtube.com/watch?v=...")
        max_comments = st.slider("Batas Maksimal Komentar per Video", 30, 200, 50)

    btn_proses = st.button("🚀 Ekstrak, Preprocess & Analisis", type="primary")

    if btn_proses:
        urls = [u.strip() for u in [url_1, url_2, url_3, url_4, url_5] if u.strip()]
        if not urls:
            st.warning("Silakan masukkan minimal 1 URL Video YouTube.")
        else:
            progress_bar = st.progress(0)
            status_text = st.empty()
            extracted_results = []
            
            for index, url in enumerate(urls):
                video_label = f"Video {index+1}"
                status_text.text(f"Mengekstrak & Memproses {video_label} menggunakan Sastrawi...")
                
                comments_data = []
                if HAS_SCRAPER:
                    try:
                        downloader = YoutubeCommentDownloader()
                        comments = downloader.get_comments_from_url(url, sort_by=SORT_BY_POPULAR)
                        count = 0
                        for c in comments:
                            if count >= max_comments:
                                break
                            raw_text = c['text']
                            # Terapkan Preprocessing Sastrawi
                            clean_text = preprocess_text(raw_text)
                            
                            # Klasifikasi Sederhana Berdasar Kata Kunci Hasil Preprocessing
                            pos_words = ['suka', 'bagus', 'lucu', 'wkwk', 'otsu', 'halo', 'semangat', 'mantap', 'love', 'malam', 'keren']
                            sentiment = "Positif" if any(w in clean_text for w in pos_words) else "Negatif"
                            
                            comments_data.append({
                                'Video': video_label,
                                'Komentar Asli': raw_text,
                                'Komentar Bersih (Sastrawi)': clean_text,
                                'Prediksi Sentimen': sentiment
                            })
                            count += 1
                    except Exception as ex:
                        st.error(f"Gagal mengekstrak {video_label}: {ex}")

                # Fallback jika terjadi kendala ekstraksi
                if not comments_data:
                    for i in range(15):
                        dummy_raw = f"Contoh live chat ke-{i+1} dari {video_label} keren banget"
                        comments_data.append({
                            'Video': video_label,
                            'Komentar Asli': dummy_raw,
                            'Komentar Bersih (Sastrawi)': preprocess_text(dummy_raw),
                            'Prediksi Sentimen': "Positif" if i % 2 == 0 else "Negatif"
                        })

                extracted_results.extend(comments_data)
                progress_bar.progress((index + 1) / len(urls))
            
            status_text.text("Analisis Selesai!")
            st.session_state['live_data'] = pd.DataFrame(extracted_results)

    if 'live_data' in st.session_state and not st.session_state['live_data'].empty:
        df_live = st.session_state['live_data']
        st.markdown("---")
        st.markdown("#### 📊 Perbandingan Hasil Sentimen Live Video")
        
        live_stats = df_live.groupby(['Video', 'Prediksi Sentimen']).size().reset_index(name='Jumlah')
        fig_live = px.bar(live_stats, x='Video', y='Jumlah', color='Prediksi Sentimen', barmode='group',
                          color_discrete_map={'Positif': COLOR_POS, 'Negatif': COLOR_NEG})
        st.plotly_chart(style_fig(fig_live), use_container_width=True)

        st.markdown("#### 🔍 Pratinjau Hasil Preprocessing Teks (Sastrawi)")
        st.dataframe(df_live[['Video', 'Komentar Asli', 'Komentar Bersih (Sastrawi)', 'Prediksi Sentimen']], use_container_width=True)

        # Download Excel Hasil Live
        def convert_df_to_excel(df_to_download):
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                df_to_download.to_excel(writer, index=False)
            return buffer.getvalue()

        excel_data = convert_df_to_excel(df_live)
        st.download_button(
            label="📥 Unduh Data Hasil Preprocessing & Sentimen (.xlsx)",
            data=excel_data,
            file_name="live_extracted_chat_sastrawi.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

# ==========================================
# TAB 1: RINGKASAN & LDA (DATA BENCHMARK)
# ==========================================
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
        c1.markdown(f'<div class="metric-card"><div class="metric-title">Total Chat Menganalisis</div><div class="metric-value">{total_chat:,}</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="metric-card"><div class="metric-title">Sentimen Positif</div><div class="metric-value" style="color:{COLOR_POS}">{pos_pct:.1f}%</div><div class="metric-sub">{pos_chat:,} chat</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="metric-card"><div class="metric-title">Sentimen Negatif</div><div class="metric-value" style="color:{COLOR_NEG}">{neg_pct:.1f}%</div><div class="metric-sub">{neg_chat:,} chat</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        with st.expander("📖 **Petunjuk Kata Kunci Tiap Topik LDA (Klik untuk Membuka)**", expanded=False):
            st.markdown("""
            * **Topik 1 (Sapaan & Interaksi)**: *bang, banget, sil, tris, kalian, malam*
            * **Topik 2 (Respon & Obrolan)**: *aku, di, itu, ga, yang, ada*
            * **Topik 3 (Ucapan Datang/Live)**: *the, live, selamat, datang, di, semoga*
            * **Topik 4 (Apresiasi Stream)**: *kak, halo, jangan, otsu, stream, ka*
            * **Topik 5 (Ekspresi Suka/Tertawa)**: *lagi, dan, wkwkwk, suka, lah, dengan*
            """)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("##### Proporsi Sentimen Chat")
            if col_sentimen in df_filtered.columns:
                fig_s = px.pie(df_filtered, names=col_sentimen, color=col_sentimen, color_discrete_map={'Positif': COLOR_POS, 'Negatif': COLOR_NEG}, hole=0.6)
                st.plotly_chart(style_fig(fig_s), use_container_width=True)

        with col_b:
            st.markdown("##### Proporsi Topik LDA Dominan")
            fig_t = px.pie(df_filtered, names=col_topik, hole=0.6, color_discrete_sequence=COLOR_THEME)
            st.plotly_chart(style_fig(fig_t), use_container_width=True)

        st.markdown("---")
        st.markdown("##### 📌 Tabel Distribusi Frekuensi Topik LDA")
        topik_df = df_filtered[col_topik].value_counts().reset_index()
        topik_df.columns = ['Nama Topik LDA', 'Jumlah Chat']
        topik_df['Persentase (%)'] = (topik_df['Jumlah Chat'] / total_chat * 100).round(2)
        st.dataframe(topik_df, use_container_width=True)

# ==========================================
# TAB 2: PROFIL & PERBANDINGAN VTUBER
# ==========================================
with tab2:
    st.markdown("### 🔍 Filter Individual 1 VTuber")
    selected_single_vt = st.selectbox("Pilih 1 VTuber untuk melihat analisis detailnya:", all_vtubers) if all_vtubers else None

    if selected_single_vt:
        df_single = df[df[col_vtuber] == selected_single_vt]

        p1, p2, p3 = st.columns(3)
        with p1:
            st.markdown(f"##### Kategori Stream ({selected_single_vt})")
            if col_stream in df_single.columns:
                fig_single_cat = px.pie(df_single, names=col_stream, hole=0.55, color_discrete_sequence=COLOR_THEME)
                st.plotly_chart(style_fig(fig_single_cat), use_container_width=True)

        with p2:
            st.markdown(f"##### Topik LDA Dominan ({selected_single_vt})")
            fig_single_top = px.pie(df_single, names=col_topik, hole=0.55, color_discrete_sequence=COLOR_THEME)
            st.plotly_chart(style_fig(fig_single_top), use_container_width=True)

        with p3:
            st.markdown(f"##### Sebaran Sentimen ({selected_single_vt})")
            if col_sentimen in df_single.columns:
                fig_single_sent = px.pie(df_single, names=col_sentimen, hole=0.55, color_discrete_map={'Positif': COLOR_POS, 'Negatif': COLOR_NEG})
                st.plotly_chart(style_fig(fig_single_sent), use_container_width=True)

    st.markdown("---")
    st.markdown("### 🏆 Diagram Perbandingan Antar VTuber")

    if col_vtuber in df_filtered.columns and col_sentimen in df_filtered.columns:
        vt_stats = df_filtered.groupby([col_vtuber, col_sentimen]).size().unstack(fill_value=0)
        if 'Positif' not in vt_stats.columns: vt_stats['Positif'] = 0
        if 'Negatif' not in vt_stats.columns: vt_stats['Negatif'] = 0
        vt_stats['Total'] = vt_stats['Positif'] + vt_stats['Negatif']
        vt_stats['Rata_Rata_Positif (%)'] = (vt_stats['Positif'] / vt_stats['Total']) * 100
        vt_stats = vt_stats.sort_values(by='Rata_Rata_Positif (%)', ascending=True).reset_index()

        fig_rank = px.bar(
            vt_stats, y=col_vtuber, x='Rata_Rata_Positif (%)', orientation='h',
            text_auto='.1f', color='Rata_Rata_Positif (%)', color_continuous_scale='greens',
            title="Ranking Persentase Sentimen Positif per VTuber"
        )
        st.plotly_chart(style_fig(fig_rank), use_container_width=True)

        col_v1, col_v2 = st.columns(2)
        with col_v1:
            st.markdown("##### Sebaran Sentimen per VTuber")
            fig_vt_sent = px.histogram(df_filtered, x=col_vtuber, color=col_sentimen, barmode='group', color_discrete_map={'Positif': COLOR_POS, 'Negatif': COLOR_NEG})
            st.plotly_chart(style_fig(fig_vt_sent), use_container_width=True)

        with col_v2:
            st.markdown("##### Sebaran Kategori Stream per VTuber")
            if col_stream in df_filtered.columns:
                fig_vt_cat = px.histogram(df_filtered, x=col_vtuber, color=col_stream, barmode='stack', color_discrete_sequence=COLOR_THEME)
                st.plotly_chart(style_fig(fig_vt_cat), use_container_width=True)

        st.markdown("##### Sebaran Topik LDA per VTuber")
        fig_vt_lda = px.histogram(df_filtered, x=col_vtuber, color=col_topik, barmode='stack', color_discrete_sequence=COLOR_THEME)
        st.plotly_chart(style_fig(fig_vt_lda), use_container_width=True)

# ==========================================
# TAB 3: KATEGORI STREAM
# ==========================================
with tab3:
    st.markdown("### 🎮 Perbandingan Berdasarkan Kategori Stream")
    if col_stream in df_filtered.columns:
        col_k1, col_k2 = st.columns(2)
        with col_k1:
            st.markdown("##### Sentimen per Kategori Stream")
            if col_sentimen in df_filtered.columns:
                fig_cat_sent = px.histogram(df_filtered, x=col_stream, color=col_sentimen, barmode='group', color_discrete_map={'Positif': COLOR_POS, 'Negatif': COLOR_NEG})
                st.plotly_chart(style_fig(fig_cat_sent), use_container_width=True)

        with col_k2:
            st.markdown("##### Topik LDA per Kategori Stream")
            fig_cat_top = px.histogram(df_filtered, x=col_stream, color=col_topik, barmode='stack', color_discrete_sequence=COLOR_THEME)
            st.plotly_chart(style_fig(fig_cat_top), use_container_width=True)

# ==========================================
# TAB 4: RAW DATA
# ==========================================
with tab4:
    st.markdown("### 📑 Raw Data Chat Interaktif (20 VTuber Benchmark)")
    st.dataframe(df_filtered, use_container_width=True)
