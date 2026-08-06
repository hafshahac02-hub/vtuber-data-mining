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
    except Exception:
        return None, None

model_nb, tfidf_vec = load_ml_model()

# 4. Fungsi Pemisah Emoji
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
            vec = tfidf_vec.transform([input_text])
            pred = model_nb.predict(vec)[0]
            return pred
        except Exception:
            return "Positif"
    return "Positif"

# 7. Styling CSS UI Modern
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
    .metric-title { font-size: 0.8rem; color: #A0AEC0; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }
    .metric-value { font-size: 1.8rem; font-weight: 700; color: #FFFFFF; }
    .metric-sub { font-size: 0.8rem; color: #718096; }
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
def load_benchmark_data():
    files = [f for f in os.listdir('.') if f.endswith('.xlsx')]
    if not files:
        return pd.DataFrame()
    
    if 'data_vtuber_labeled.xlsx' in files:
        target_file = 'data_vtuber_labeled.xlsx'
    elif 'hasil_akhir_analisis_skripsi.xlsx' in files:
        target_file = 'hasil_akhir_analisis_skripsi.xlsx'
    else:
        target_file = files[0]
        
    return pd.read_excel(target_file)

df_benchmark = load_benchmark_data()

def find_col(df, possible_names, default):
    if df.empty:
        return default
    for name in possible_names:
        for col in df.columns:
            if name.lower() in str(col).lower():
                return col
    return default

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
        st.warning("⚠️ File model_sentiment.pkl belum terdeteksi. Silakan upload file .pkl ke repositori GitHub.")

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
        st.error("⚠️ File dataset benchmark (.xlsx) tidak ditemukan di repositori GitHub!")
        st.info("Pastikan file `data_vtuber_labeled.xlsx` atau `hasil_akhir_analisis_skripsi.xlsx` ada di repo.")
    else:
        # Deteksi Kolom Otomatis
        col_vtuber = find_col(df_benchmark, ['vtuber', 'nama', 'channel'], 'VTuber Name')
        col_stream = find_col(df_benchmark, ['stream', 'kategori', 'type'], 'Stream Type')
        col_sentimen = find_col(df_benchmark, ['sentimen', 'sentiment', 'prediksi', 'label'], 'Prediksi Sentimen')
        col_topik_raw = find_col(df_benchmark, ['topik', 'klaster', 'lda'], 'Klaster Topik Dominan')

        # Map Nomor Topik LDA ke Label Deskriptif
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
        if col_topik_raw in df_benchmark.columns:
            df_benchmark[col_topik] = df_benchmark[col_topik_raw].map(TOPIC_MAP).fillna(df_benchmark[col_topik_raw].astype(str))
        else:
            df_benchmark[col_topik] = "General"

        # FILTER PANEL SIDEBAR GLOBAL
        st.sidebar.markdown("### 🎛️ Filter Panel Global")
        all_vtubers = sorted(df_benchmark[col_vtuber].dropna().unique().tolist()) if col_vtuber in df_benchmark.columns else []
        selected_vtubers = st.sidebar.multiselect("Pilih VTuber", all_vtubers, default=all_vtubers)

        all_streams = sorted(df_benchmark[col_stream].dropna().unique().tolist()) if col_stream in df_benchmark.columns else []
        selected_streams = st.sidebar.multiselect("Pilih Kategori Stream", all_streams, default=all_streams)

        # Apply Filter
        df_filtered = df_benchmark.copy()
        if col_vtuber in df_benchmark.columns and selected_vtubers:
            df_filtered = df_filtered[df_filtered[col_vtuber].isin(selected_vtubers)]
        if col_stream in df_benchmark.columns and selected_streams:
            df_filtered = df_filtered[df_filtered[col_stream].isin(selected_streams)]

        if df_filtered.empty:
            st.warning("Data kosong untuk kombinasi filter ini.")
        else:
            # NAVIGASI TAB (Tabel Raw Data sudah dihilangkan)
            tab1, tab2, tab3 = st.tabs([
                "📊 Ringkasan & Referensi LDA",
                "👤 Profil & Perbandingan VTuber",
                "🎮 Analisis Kategori Stream"
            ])

            # TAB 1: RINGKASAN & LDA
            with tab1:
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

            # TAB 2: PROFIL & PERBANDINGAN VTUBER
            with tab2:
                st.markdown("### 🔍 Filter Individual 1 VTuber")
                selected_single_vt = st.selectbox("Pilih 1 VTuber untuk melihat analisis detailnya:", all_vtubers) if all_vtubers else None

                if selected_single_vt:
                    df_single = df_benchmark[df_benchmark[col_vtuber] == selected_single_vt]

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

            # TAB 3: KATEGORI STREAM
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
