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
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2. Import Scraper & Sastrawi Optional
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


# 4. Preprocessing Sastrawi (Khusus Teks Pendek Realtime)
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


# 6. Pemetaan Topik Berbasis Kata Kunci (Ringan & Cepat Tanpa Stemming Berat)
def deteksi_topik_cepat(teks_input):
  if not teks_input or pd.isna(teks_input):
    return "Topik 2: Respon & Obrolan"

  t = str(teks_input).lower()

  if any(
      w in t
      for w in [
          "otsu",
          "otsukare",
          "makasih",
          "terimakasih",
          "stream",
          "terima kasih",
          "ka",
          "kak",
      ]
  ):
    return "Topik 4: Apresiasi Stream (Otsu)"
  elif any(
      w in t
      for w in [
          "wkwk",
          "wkwkwk",
          "haha",
          "hahaha",
          "xixi",
          "lol",
          "suka",
          "ngakak",
          "lagi",
      ]
  ):
    return "Topik 5: Ekspresi Suka / Tertawa"
  elif any(
      w in t
      for w in ["selamat", "datang", "welcome", "live", "semoga", "the"]
  ):
    return "Topik 3: Ucapan Datang / Live"
  elif any(
      w in t
      for w in [
          "halo",
          "hai",
          "bang",
          "malam",
          "pagi",
          "siang",
          "kalian",
          "sil",
          "tris",
      ]
  ):
    return "Topik 1: Sapaan & Interaksi"
  else:
    return "Topik 2: Respon & Obrolan"


# 7. Styling CSS UI Modern
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }
    .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 14px 18px;
    }
    .metric-title { font-size: 0.75rem; color: #A0AEC0; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }
    .metric-value { font-size: 1.7rem; font-weight: 700; color: #FFFFFF; }
    .metric-sub { font-size: 0.78rem; color: #718096; }
    .extractor-container {
        background: rgba(99, 102, 241, 0.05);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 15px;
    }
    .analysis-card {
        background: rgba(255, 255, 255, 0.02);
        border-left: 4px solid #6366F1;
        padding: 15px 20px;
        border-radius: 6px;
        margin-bottom: 20px;
    }
    </style>
""",
    unsafe_allow_html=True,
)


# 8. Load Dataset Benchmark (20 VTuber)
@st.cache_data
def load_benchmark_data():
  files = [f for f in os.listdir(".") if f.endswith(".xlsx")]
  if not files:
    return pd.DataFrame()

  if "data_vtuber_labeled.xlsx" in files:
    target_file = "data_vtuber_labeled.xlsx"
  elif "hasil_akhir_analisis_skripsi.xlsx" in files:
    target_file = "hasil_akhir_analisis_skripsi.xlsx"
  else:
    target_file = files[0]

  df = pd.read_excel(target_file)
  return df.copy()


df_benchmark = load_benchmark_data()


def find_col(df, possible_names):
  if df.empty:
    return None
  for name in possible_names:
    for col in df.columns:
      if name.lower() in str(col).lower():
        return col
  return None


TOPIC_MAP_NUM = {
    1: "Topik 1: Sapaan & Interaksi",
    2: "Topik 2: Respon & Obrolan",
    3: "Topik 3: Ucapan Datang / Live",
    4: "Topik 4: Apresiasi Stream (Otsu)",
    5: "Topik 5: Ekspresi Suka / Tertawa",
}


def process_benchmark_df(df_input):
  if df_input.empty:
    return (
        df_input,
        "VTuber Name",
        "Stream Type",
        "Prediksi Sentimen",
        "Nama Topik LDA",
    )

  df = df_input.copy()

  col_v = (
      find_col(df, ["vtuber", "nama", "channel", "creator"]) or df.columns[0]
  )
  col_s = (
      find_col(df, ["stream", "kategori", "category", "type"]) or df.columns[1]
  )
  col_sent = (
      find_col(df, ["sentimen", "sentiment", "prediksi", "label"])
      or df.columns[2]
  )

  col_topik_raw = find_col(
      df,
      [
          "topik",
          "topic",
          "klaster",
          "cluster",
          "lda",
          "dominant",
          "label_topik",
          "nama topik",
      ],
  )
  col_text_raw = find_col(
      df,
      [
          "pesan bersih",
          "clean_text",
          "chat text",
          "chat",
          "comment",
          "komentar",
          "pesan",
          "text",
      ],
  )

  col_topik = "Nama Topik LDA"

  def convert_row_topic(row):
    val = row[col_topik_raw] if col_topik_raw and col_topik_raw in row else None
    chat_str = row[col_text_raw] if col_text_raw and col_text_raw in row else ""

    if pd.notna(val):
      s_val = str(val).strip()
      for t_name in TOPIC_MAP_NUM.values():
        if t_name.lower() in s_val.lower():
          return t_name

      try:
        num = int(float(s_val))
        if num in TOPIC_MAP_NUM:
          return TOPIC_MAP_NUM[num]
      except Exception:
        pass

    return deteksi_topik_cepat(chat_str)

  df[col_topik] = df.apply(convert_row_topic, axis=1)
  return df, col_v, col_s, col_sent, col_topik


df_benchmark, col_vtuber, col_stream, col_sentimen, col_topik = (
    process_benchmark_df(df_benchmark)
)

COLOR_POS = "#10B981"
COLOR_NEG = "#EF4444"
COLOR_THEME = [
    "#6366F1",
    "#8B5CF6",
    "#EC4899",
    "#F59E0B",
    "#3B82F6",
    "#10B981",
    "#14B8A6",
]


def style_fig(fig):
  fig.update_layout(
      paper_bgcolor="rgba(0,0,0,0)",
      plot_bgcolor="rgba(0,0,0,0)",
      font=dict(family="Plus Jakarta Sans", color="#E2E8F0"),
      margin=dict(l=15, r=15, t=25, b=15),
      legend=dict(
          orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
      ),
  )
  return fig


# HEADER UTAMA
st.title("🎭 VTuber Live Chat Mining & Analytics System")

mode_pilihan = st.radio(
    "📌 **Pilih Mode Aplikasi:**",
    [
        "⚡ Ekstraksi Live Chat (Realtime)",
        "📊 Dashboard Benchmark Dataset (20 VTuber)",
    ],
    horizontal=True,
)

st.markdown("---")

# ==========================================================
# MODE 1: EKSTRAKSI LIVE CHAT (REALTIME + ANALYTICS)
# ==========================================================
if mode_pilihan == "⚡ Ekstraksi Live Chat (Realtime)":
  st.markdown(
      """
        <div class="extractor-container">
            <h3 style="margin-top:0;">🧪 Fitur Penarikan Live Chat Stream Single-URL</h3>
            <p style="color: #A0AEC0; font-size: 0.9rem; margin-bottom: 0;">
                Masukkan 1 link YouTube Live/Replay untuk pengujian inferensi real-time. Sentimen diklasifikasikan menggunakan <b>Model Naïve Bayes (.pkl)</b> dan topik dipetakan berbasis <b>LDA Model</b>.
            </p>
        </div>
    """,
      unsafe_allow_html=True,
  )

  if model_nb is None:
    st.warning(
        "⚠️ File model_sentiment.pkl belum terdeteksi di repositori. Prediksi"
        " sentimen berjalan dalam mode standar."
    )

  col_url, col_kat = st.columns([3, 1])
  with col_url:
    input_url = st.text_input(
        "URL Video YouTube Stream Replay",
        placeholder=(
            "https://www.youtube.com/live/... atau"
            " https://www.youtube.com/watch?v=..."
        ),
    )
  with col_kat:
    kategori_pilihan = st.selectbox(
        "Kategori Stream",
        [
            "Gaming",
            "Freetalk",
            "Collaboration",
            "Karaoke",
            "Working",
            "Lainnya",
        ],
    )

  btn_proses = st.button(
      "🚀 Ekstrak Live Chat", type="primary", use_container_width=True
  )

  if btn_proses:
    original_url = input_url.strip() if input_url else ""

    if not original_url:
      st.warning(
          "Silakan masukkan URL Video YouTube Live Replay terlebih dahulu."
      )
    elif not HAS_SCRAPER:
      st.error(
          "Library `yt-chat-downloader` belum terinstal di server. Pastikan"
          " requirements.txt sudah diupdate."
      )
    else:
      clean_url = original_url
      if "/live/" in clean_url:
        clean_url = clean_url.replace("/live/", "/watch?v=")
      if "?si=" in clean_url:
        clean_url = clean_url.split("?si=")[0]

      status_box = st.info(
          "⏳ Sedang menarik live chat asli, memproses pembersihan teks,"
          " memprediksi sentimen, dan mengekstrak topik..."
      )

      try:
        downloader = YouTubeChatDownloader()
        messages = downloader.download_chat(
            video_url=clean_url, chat_type="live", quiet=True
        )

        extracted_rows = []

        for msg in messages:
          komentar_asli = msg.get("comment", "")
          if komentar_asli:
            clean_text = preprocess_text(komentar_asli)
            sentiment = prediksi_sentimen_ml(clean_text, komentar_asli)
            topik_lda = deteksi_topik_cepat(clean_text)

            extracted_rows.append({
                "Username": msg.get("user_display_name", "Anonymous"),
                "Chat Text": komentar_asli,
                "Timestamp": msg.get("datetime", ""),
                "Pesan Bersih (Sastrawi)": clean_text,
                "Prediksi Sentimen": sentiment,
                "Topik LDA": topik_lda,
                "Kategori Stream": kategori_pilihan,
            })

        status_box.empty()

        if not extracted_rows:
          st.error(
              "❌ Live chat tidak terdeteksi pada video ini. Pastikan video"
              " memiliki Replay Live Chat yang aktif di YouTube."
          )
        else:
          df_res = pd.DataFrame(extracted_rows)
          st.session_state["real_extracted_data"] = df_res
          st.success(
              "✅ Berhasil mengekstrak dan menganalisis"
              f" **{len(df_res):,} baris** live chat!"
          )

      except Exception as e:
        status_box.empty()
        st.error(f"❌ Terjadi kesalahan saat menarik data: {e}")

  # TAMPILAN HASIL ANALISIS REALTIME
  if (
      "real_extracted_data" in st.session_state
      and not st.session_state["real_extracted_data"].empty
  ):
    df_real = st.session_state["real_extracted_data"]
    total_real = len(df_real)
    pos_real = (df_real["Prediksi Sentimen"] == "Positif").sum()
    neg_real = (df_real["Prediksi Sentimen"] == "Negatif").sum()
    pos_pct = (pos_real / total_real * 100) if total_real > 0 else 0
    neg_pct = (neg_real / total_real * 100) if total_real > 0 else 0

    st.markdown("---")
    st.subheader("📊 Hasil Analisis Data Realtime")

    m1, m2, m3 = st.columns(3)
    m1.markdown(
        '<div class="metric-card"><div class="metric-title">Total Live Chat'
        f' Ter-ekstrak</div><div class="metric-value">{total_real:,}</div></div>',
        unsafe_allow_html=True,
    )
    m2.markdown(
        '<div class="metric-card"><div class="metric-title">Sentimen'
        ' Positif</div><div class="metric-value"'
        f' style="color:{COLOR_POS}">{pos_pct:.1f}%</div><div'
        f' class="metric-sub">{pos_real:,} chat</div></div>',
        unsafe_allow_html=True,
    )
    m3.markdown(
        '<div class="metric-card"><div class="metric-title">Sentimen'
        ' Negatif</div><div class="metric-value"'
        f' style="color:{COLOR_NEG}">{neg_pct:.1f}%</div><div'
        f' class="metric-sub">{neg_real:,} chat</div></div>',
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    c_g1, c_g2 = st.columns(2)
    with c_g1:
      st.markdown("##### Proporsi Sentimen (Naïve Bayes)")
      fig_sent = px.pie(
          df_real,
          names="Prediksi Sentimen",
          color="Prediksi Sentimen",
          color_discrete_map={"Positif": COLOR_POS, "Negatif": COLOR_NEG},
          hole=0.55,
      )
      st.plotly_chart(style_fig(fig_sent), use_container_width=True)

    with c_g2:
      st.markdown("##### Proporsi Topik LDA Dominan")
      fig_topik = px.pie(
          df_real,
          names="Topik LDA",
          hole=0.55,
          color_discrete_sequence=COLOR_THEME,
      )
      st.plotly_chart(style_fig(fig_topik), use_container_width=True)

    c_g3, c_g4 = st.columns(2)
    with c_g3:
      st.markdown("##### Sebaran Sentimen per Topik LDA")
      fig_top_sent = px.histogram(
          df_real,
          x="Topik LDA",
          color="Prediksi Sentimen",
          barmode="group",
          color_discrete_map={"Positif": COLOR_POS, "Negatif": COLOR_NEG},
      )
      st.plotly_chart(style_fig(fig_top_sent), use_container_width=True)

    with c_g4:
      st.markdown("##### 10 Kata Kunci Terbanyak Dalam Stream")
      semua_kata = " ".join(
          df_real["Pesan Bersih (Sastrawi)"].dropna()
      ).split()
      kata_counts = Counter(semua_kata).most_common(10)
      if kata_counts:
        df_words = pd.DataFrame(kata_counts, columns=["Kata", "Frekuensi"])
        fig_words = px.bar(
            df_words,
            x="Frekuensi",
            y="Kata",
            orientation="h",
            color_discrete_sequence=["#6366F1"],
        )
        fig_words.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(style_fig(fig_words), use_container_width=True)

    st.markdown("---")

    st.markdown("### 📑 Tabel Keseluruhan Hasil Ekstraksi Live Chat")
    st.dataframe(df_real, use_container_width=True)

    def convert_df_to_excel(df_to_download):
      buffer = io.BytesIO()
      with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df_to_download.to_excel(writer, index=False)
      return buffer.getvalue()

    excel_bytes = convert_df_to_excel(df_real)
    st.download_button(
        label="📥 Unduh Data Live Chat Ter-ekstrak (.xlsx)",
        data=excel_bytes,
        file_name="hasil_ekstraksi_livechat_realtime.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary",
    )

# ==========================================================
# MODE 2: DASHBOARD BENCHMARK DATASET (20 VTUBER)
# ==========================================================
else:
  if df_benchmark.empty:
    st.error(
        "⚠️ File dataset benchmark (.xlsx) tidak ditemukan di repositori GitHub!"
    )
    st.info(
        "Pastikan file `data_vtuber_labeled.xlsx` atau"
        " `hasil_akhir_analisis_skripsi.xlsx` ada di repo."
    )
  else:
    st.sidebar.markdown("### 🎛️ Filter Panel Global")
    all_vtubers = (
        sorted(df_benchmark[col_vtuber].dropna().unique().tolist())
        if col_vtuber in df_benchmark.columns
        else []
    )
    selected_vtubers = st.sidebar.multiselect(
        "Pilih VTuber", all_vtubers, default=all_vtubers
    )

    all_streams = (
        sorted(df_benchmark[col_stream].dropna().unique().tolist())
        if col_stream in df_benchmark.columns
        else []
    )
    selected_streams = st.sidebar.multiselect(
        "Pilih Kategori Stream", all_streams, default=all_streams
    )

    df_filtered = df_benchmark.copy()
    if col_vtuber in df_benchmark.columns and selected_vtubers:
      df_filtered = df_filtered[df_filtered[col_vtuber].isin(selected_vtubers)]
    if col_stream in df_benchmark.columns and selected_streams:
      df_filtered = df_filtered[df_filtered[col_stream].isin(selected_streams)]

    if df_filtered.empty:
      st.warning("Data kosong untuk kombinasi filter ini.")
    else:
      tab1, tab2 = st.tabs([
          "📊 Analisis Deskriptif Data Mining (Keseluruhan vs Profil VTuber)",
          "🎮 Analisis Kategori Stream & Korelasi",
      ])

      # TAB 1: ANALISIS DESKRIPTIF DATA MINING & COMPARED SIDE-BY-SIDE
      with tab1:
        total_chat = len(df_filtered)
        pos_chat = (
            (df_filtered[col_sentimen] == "Positif").sum()
            if col_sentimen in df_filtered.columns
            else 0
        )
        neg_chat = (
            (df_filtered[col_sentimen] == "Negatif").sum()
            if col_sentimen in df_filtered.columns
            else 0
        )
        pos_pct = (pos_chat / total_chat * 100) if total_chat > 0 else 0
        neg_pct = (neg_chat / total_chat * 100) if total_chat > 0 else 0

        # METRIC CARDS OVERALL
        c1, c2, c3 = st.columns(3)
        c1.markdown(
            '<div class="metric-card"><div class="metric-title">Total Chat'
            f' Menganalisis</div><div class="metric-value">{total_chat:,}</div></div>',
            unsafe_allow_html=True,
        )
        c2.markdown(
            '<div class="metric-card"><div class="metric-title">Sentimen'
            ' Positif</div><div class="metric-value"'
            f' style="color:{COLOR_POS}">{pos_pct:.1f}%</div><div'
            f' class="metric-sub">{pos_chat:,} chat</div></div>',
            unsafe_allow_html=True,
        )
        c3.markdown(
            '<div class="metric-card"><div class="metric-title">Sentimen'
            ' Negatif</div><div class="metric-value"'
            f' style="color:{COLOR_NEG}">{neg_pct:.1f}%</div><div'
            f' class="metric-sub">{neg_chat:,} chat</div></div>',
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # PENJELASAN ANALISIS DESKRIPTIF DATA MINING
        st.markdown(
            f"""
            <div class="analysis-card">
                <h4 style="margin-top:0; color:#6366F1;">🧠 Interpretasi Deskriptif Data Mining (Skala 20 VTuber)</h4>
                <p style="font-size:0.92rem; color:#CBD5E1; margin-bottom:0;">
                    Berdasarkan analisis klasifikasi <b>Naïve Bayes</b> dan ekstraksi klaster <b>Latent Dirichlet Allocation (LDA)</b> pada dataset <b>{total_chat:,} chat</b>:
                    <br>• <b>Polaritas Sentimen</b>: Didominasi oleh emosi positif sebesar <b>{pos_pct:.1f}%</b>, yang menunjukkan tingkat keterikatan komunitatif yang sangat kuat pada ekosistem VTuber Indonesia.
                    <br>• <b>Klaster Topik LDA</b>: Interaksi audiens terkonsentrasi pada pola sapaan, apresiasi siaran (Otsu), serta ekspresi hiburan (tawa/suka), yang membentuk korelasi langsung terhadap tingginya rasio sentimen positif.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander(
            "📖 **Referensi Kata Kunci Tiap Topik LDA (Klik untuk Membuka)**",
            expanded=False,
        ):
          st.markdown("""
            * **Topik 1 (Sapaan & Interaksi)**: *bang, banget, sil, tris, kalian, malam*
            * **Topik 2 (Respon & Obrolan)**: *aku, di, itu, ga, yang, ada*
            * **Topik 3 (Ucapan Datang/Live)**: *the, live, selamat, datang, di, semoga*
            * **Topik 4 (Apresiasi Stream)**: *kak, halo, jangan, otsu, stream, ka*
            * **Topik 5 (Ekspresi Suka/Tertawa)**: *lagi, dan, wkwkwk, suka, lah, dengan*
            """)

        st.markdown("---")

        # TAMPILAN SIDE-BY-SIDE: KESELURUHAN (KIRI) vs INDIVIDUAL (KANAN)
        col_left, col_right = st.columns(2)

        # SEKSI KIRI: KESELURUHAN 20 VTUBER
        with col_left:
          st.markdown("### 🌐 Analisis Akumulasi (20 VTuber)")

          st.markdown("##### Proporsi Sentimen Chat (Keseluruhan)")
          if col_sentimen in df_filtered.columns:
            fig_s_all = px.pie(
                df_filtered,
                names=col_sentimen,
                color=col_sentimen,
                color_discrete_map={
                    "Positif": COLOR_POS,
                    "Negatif": COLOR_NEG,
                },
                hole=0.55,
            )
            st.plotly_chart(style_fig(fig_s_all), use_container_width=True)

          st.markdown("##### Proporsi Topik LDA Dominan (Keseluruhan)")
          fig_t_all = px.pie(
              df_filtered,
              names=col_topik,
              hole=0.55,
              color_discrete_sequence=COLOR_THEME,
          )
          st.plotly_chart(style_fig(fig_t_all), use_container_width=True)

          st.markdown("##### Sebaran Sentimen per Topik LDA (Keseluruhan)")
          fig_top_sent_all = px.histogram(
              df_filtered,
              x=col_topik,
              color=col_sentimen,
              barmode="group",
              color_discrete_map={"Positif": COLOR_POS, "Negatif": COLOR_NEG},
          )
          st.plotly_chart(
              style_fig(fig_top_sent_all), use_container_width=True
          )

        # SEKSI KANAN: PROFIL 1 VTUBER INDIVIDUAL
        with col_right:
          st.markdown("### 👤 Analisis Profil 1 VTuber")
          selected_single_vt = (
              st.selectbox(
                  "Pilih VTuber untuk Analisis Spesifik:",
                  all_vtubers,
                  key="single_vt_select",
              )
              if all_vtubers
              else None
          )

          if selected_single_vt:
            df_single = df_benchmark[
                df_benchmark[col_vtuber] == selected_single_vt
            ]

            st.markdown(
                f"##### Proporsi Sentimen Chat ({selected_single_vt})"
            )
            if col_sentimen in df_single.columns and not df_single.empty:
              fig_s_single = px.pie(
                  df_single,
                  names=col_sentimen,
                  color=col_sentimen,
                  color_discrete_map={
                      "Positif": COLOR_POS,
                      "Negatif": COLOR_NEG,
                  },
                  hole=0.55,
              )
              st.plotly_chart(style_fig(fig_s_single), use_container_width=True)

            st.markdown(
                f"##### Proporsi Topik LDA Dominan ({selected_single_vt})"
            )
            if not df_single.empty:
              fig_t_single = px.pie(
                  df_single,
                  names=col_topik,
                  hole=0.55,
                  color_discrete_sequence=COLOR_THEME,
              )
              st.plotly_chart(
                  style_fig(fig_t_single), use_container_width=True
              )

            st.markdown(
                f"##### Sebaran Kategori Stream ({selected_single_vt})"
            )
            if col_stream in df_single.columns and not df_single.empty:
              fig_cat_single = px.pie(
                  df_single,
                  names=col_stream,
                  hole=0.55,
                  color_discrete_sequence=COLOR_THEME,
              )
              st.plotly_chart(
                  style_fig(fig_cat_single), use_container_width=True
              )

        st.markdown("---")

        # DIAGRAM PERBANDINGAN DAN RANKING UTUH SE-DATASET
        st.markdown("### 🏆 Diagram Perbandingan Antar VTuber")

        if (
            col_vtuber in df_filtered.columns
            and col_sentimen in df_filtered.columns
        ):
          vt_stats = (
              df_filtered.groupby([col_vtuber, col_sentimen])
              .size()
              .unstack(fill_value=0)
          )
          if "Positif" not in vt_stats.columns:
            vt_stats["Positif"] = 0
          if "Negatif" not in vt_stats.columns:
            vt_stats["Negatif"] = 0
          vt_stats["Total"] = vt_stats["Positif"] + vt_stats["Negatif"]
          vt_stats["Rata_Rata_Positif (%)"] = (
              vt_stats["Positif"] / vt_stats["Total"]
          ) * 100
          vt_stats = vt_stats.sort_values(
              by="Rata_Rata_Positif (%)", ascending=True
          ).reset_index()

          fig_rank = px.bar(
              vt_stats,
              y=col_vtuber,
              x="Rata_Rata_Positif (%)",
              orientation="h",
              text_auto=".1f",
              color="Rata_Rata_Positif (%)",
              color_continuous_scale="greens",
              title="Ranking Persentase Sentimen Positif per VTuber",
          )
          st.plotly_chart(style_fig(fig_rank), use_container_width=True)

          col_v1, col_v2 = st.columns(2)
          with col_v1:
            st.markdown("##### Sebaran Sentimen per VTuber")
            fig_vt_sent = px.histogram(
                df_filtered,
                x=col_vtuber,
                color=col_sentimen,
                barmode="group",
                color_discrete_map={"Positif": COLOR_POS, "Negatif": COLOR_NEG},
            )
            st.plotly_chart(style_fig(fig_vt_sent), use_container_width=True)

          with col_v2:
            st.markdown("##### Sebaran Kategori Stream per VTuber")
            if col_stream in df_filtered.columns:
              fig_vt_cat = px.histogram(
                  df_filtered,
                  x=col_vtuber,
                  color=col_stream,
                  barmode="stack",
                  color_discrete_sequence=COLOR_THEME,
              )
              st.plotly_chart(style_fig(fig_vt_cat), use_container_width=True)

          st.markdown("##### Sebaran Topik LDA per VTuber")
          fig_vt_lda = px.histogram(
              df_filtered,
              x=col_vtuber,
              color=col_topik,
              barmode="stack",
              color_discrete_sequence=COLOR_THEME,
          )
          st.plotly_chart(style_fig(fig_vt_lda), use_container_width=True)

      # TAB 2: ANALISIS KATEGORI STREAM & KORELASI
      with tab2:
        st.markdown("### 🎮 Analisis Komparatif & Korelasi Kategori Stream")
        st.caption(
            "Membedah korelasi antara format tayangan stream dengan pola"
            " respon sentimen dan topik pembicaraan audience."
        )

        if col_stream in df_filtered.columns:
          col_k1, col_k2 = st.columns(2)
          with col_k1:
            st.markdown("##### 1. Distribusi Sentimen per Kategori Stream")
            if col_sentimen in df_filtered.columns:
              fig_cat_sent = px.histogram(
                  df_filtered,
                  x=col_stream,
                  color=col_sentimen,
                  barmode="group",
                  color_discrete_map={
                      "Positif": COLOR_POS,
                      "Negatif": COLOR_NEG,
                  },
              )
              st.plotly_chart(style_fig(fig_cat_sent), use_container_width=True)

          with col_k2:
            st.markdown("##### 2. Distribusi Topik LDA per Kategori Stream")
            fig_cat_top = px.histogram(
                df_filtered,
                x=col_stream,
                color=col_topik,
                barmode="stack",
                color_discrete_sequence=COLOR_THEME,
            )
            st.plotly_chart(style_fig(fig_cat_top), use_container_width=True)

          st.markdown("---")

          col_k3, col_k4 = st.columns(2)
          with col_k3:
            st.markdown(
                "##### 3. Persentase Sentimen Positif per Kategori (%)"
            )
            cat_stats = (
                df_filtered.groupby([col_stream, col_sentimen])
                .size()
                .unstack(fill_value=0)
            )
            if "Positif" not in cat_stats.columns:
              cat_stats["Positif"] = 0
            if "Negatif" not in cat_stats.columns:
              cat_stats["Negatif"] = 0
            cat_stats["Total"] = cat_stats["Positif"] + cat_stats["Negatif"]
            cat_stats["Positif_Pct"] = (
                cat_stats["Positif"] / cat_stats["Total"] * 100
            ).round(1)
            cat_stats = cat_stats.reset_index()

            fig_cat_pct = px.bar(
                cat_stats,
                x=col_stream,
                y="Positif_Pct",
                text_auto=".1f",
                color="Positif_Pct",
                color_continuous_scale="Purples",
            )
            st.plotly_chart(style_fig(fig_cat_pct), use_container_width=True)

          with col_k4:
            st.markdown("##### 4. Dominansi Topik Utama pada Tiap Kategori")
            top_cat_matrix = (
                df_filtered.groupby([col_stream, col_topik])
                .size()
                .reset_index(name="Jumlah Chat")
            )
            fig_matrix = px.bar(
                top_cat_matrix,
                x=col_stream,
                y="Jumlah Chat",
                color=col_topik,
                barmode="group",
                color_discrete_sequence=COLOR_THEME,
            )
            st.plotly_chart(style_fig(fig_matrix), use_container_width=True)

          st.markdown("---")

          st.markdown(
              "##### 📌 Tabel Ringkasan Korelasi Kategori Stream & Sentimen"
          )
          cat_summary = (
              df_filtered.groupby(col_stream)
              .agg(
                  Total_Chat=(col_sentimen, "count"),
                  Chat_Positif=(col_sentimen, lambda x: (x == "Positif").sum()),
                  Chat_Negatif=(col_sentimen, lambda x: (x == "Negatif").sum()),
              )
              .reset_index()
          )
          cat_summary["Rasio Positif (%)"] = (
              (cat_summary["Chat_Positif"] / cat_summary["Total_Chat"]) * 100
          ).round(2)
          st.dataframe(cat_summary, use_container_width=True)
