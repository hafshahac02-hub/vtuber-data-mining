import io
import os
import re
import traceback
from collections import Counter

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
  from PIL import Image  # noqa: F401

  HAS_PIL = True
except Exception:
  HAS_PIL = False

try:
  import kaleido  # noqa: F401

  HAS_KALEIDO = True
except Exception:
  HAS_KALEIDO = False

# Konfigurasi halaman
st.set_page_config(
    page_title="VTuber Analytics & Live Chat Extractor",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Import scraper & Sastrawi
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


# Load model machine learning (.pkl) hasil latihan dataset 20 VTuber
@st.cache_resource
def load_ml_model():
  try:
    model = joblib.load("model_sentiment.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer
  except Exception:
    return None, None


model_nb, tfidf_vec = load_ml_model()


# Preprocessing Sastrawi
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


# Prediksi sentimen Naive Bayes (.pkl)
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


# Label topik LDA — satu sumber kebenaran dipakai di seluruh app.
# Isi & urutan keyword TIDAK diubah, hanya penamaan/tampilannya dirapikan.
TOPIC_LABELS = {
    1: "Topik 1 · Sapaan & Interaksi",
    2: "Topik 2 · Obrolan Umum",
    3: "Topik 3 · Ucapan Pembuka Live",
    4: "Topik 4 · Apresiasi ke Streamer",
    5: "Topik 5 · Ekspresi Tawa & Suka",
}

TOPIC_KEYWORD_HINTS = {
    1: "bang, banget, sil, tris, kalian, malam",
    2: "aku, di, itu, ga, yang, ada",
    3: "the, live, selamat, datang, di, semoga",
    4: "kak, halo, jangan, otsu, stream, ka",
    5: "lagi, dan, wkwkwk, suka, lah, dengan",
}


def deteksi_topik_realtime(teks_bersih):
  if not teks_bersih:
    return TOPIC_LABELS[2]
  t = str(teks_bersih).lower()

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
    return TOPIC_LABELS[4]
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
    return TOPIC_LABELS[5]
  elif any(
      w in t
      for w in ["selamat", "datang", "welcome", "live", "semoga", "the"]
  ):
    return TOPIC_LABELS[3]
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
    return TOPIC_LABELS[1]
  else:
    return TOPIC_LABELS[2]


# Styling CSS UI
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }
    .block-container { padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1300px; }

    h1, h2, h3, h4, h5 { font-weight: 700 !important; letter-spacing: -0.01em; }
    h1 { font-size: 1.9rem !important; }

    .metric-card {
        background: linear-gradient(180deg, rgba(255,255,255,0.045), rgba(255,255,255,0.015));
        border: 1px solid rgba(255, 255, 255, 0.09);
        border-radius: 14px;
        padding: 16px 20px;
        transition: border-color 0.15s ease;
    }
    .metric-card:hover { border-color: rgba(99, 102, 241, 0.45); }
    .metric-title {
        font-size: 0.72rem; color: #94A3B8; font-weight: 700;
        text-transform: uppercase; letter-spacing: 0.6px; margin-bottom: 4px;
    }
    .metric-value { font-size: 1.85rem; font-weight: 800; color: #F8FAFC; line-height: 1.15; }
    .metric-sub { font-size: 0.78rem; color: #64748B; margin-top: 2px; }

    .extractor-container {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.08), rgba(139, 92, 246, 0.04));
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 16px;
        padding: 22px 24px;
        margin-bottom: 16px;
    }
    .extractor-container h3 { margin: 0 0 6px 0; }

    .section-chip {
        display: inline-block;
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        color: #A5B4FC;
        background: rgba(99, 102, 241, 0.12);
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 999px;
        padding: 3px 12px;
        margin-bottom: 10px;
    }

    div[data-testid="stExpander"] { border-radius: 12px; border-color: rgba(255,255,255,0.08); }
    div[data-testid="stTabs"] button { font-weight: 600; }
    </style>
""",
    unsafe_allow_html=True,
)


def metric_card(col, title, value, sub=None, color=None):
  # HTML dirangkai jadi satu baris (tanpa indentasi menjorok) supaya tidak
  # ditangkap markdown Streamlit sebagai indented code block — itu penyebab
  # tag penutup </div> sempat muncul sebagai teks mentah di kartu.
  color_style = f' style="color:{color}"' if color else ""
  sub_html = f'<div class="metric-sub">{sub}</div>' if sub else ""
  html = (
      f'<div class="metric-card">'
      f'<div class="metric-title">{title}</div>'
      f'<div class="metric-value"{color_style}>{value}</div>'
      f"{sub_html}"
      f"</div>"
  )
  col.markdown(html, unsafe_allow_html=True)


def section_header(text):
  st.markdown(f'<div class="section-chip">{text}</div>', unsafe_allow_html=True)


# Load dataset benchmark (20 VTuber)
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

  return pd.read_excel(target_file)


df_benchmark = load_benchmark_data()


# Deteksi kolom & pemetaan topik otomatis (anti-"General")
def find_col(df, possible_names, default=None):
  if df.empty:
    return default
  for name in possible_names:
    for col in df.columns:
      if name.lower() in str(col).lower():
        return col
  return default


col_vtuber = find_col(
    df_benchmark, ["vtuber", "nama", "channel", "creator"], "VTuber Name"
)
col_stream = find_col(
    df_benchmark, ["stream", "kategori", "category", "type"], "Stream Type"
)
col_sentimen = find_col(
    df_benchmark, ["sentimen", "sentiment", "prediksi", "label"], "Prediksi Sentimen"
)
col_topik_raw = find_col(
    df_benchmark, ["topik", "topic", "klaster", "cluster", "lda", "dominant"]
)
col_text_raw = find_col(
    df_benchmark,
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

# Map nomor topik LDA ke label deskriptif (dibangun otomatis dari
# TOPIC_LABELS supaya penamaan topik selalu konsisten di seluruh dashboard)
TOPIC_MAP = {}
for _num, _label in TOPIC_LABELS.items():
  TOPIC_MAP[_num] = _label
  TOPIC_MAP[str(_num)] = _label
  TOPIC_MAP[float(_num)] = _label
TOPIC_MAP[0] = TOPIC_LABELS[1]
TOPIC_MAP["0"] = TOPIC_LABELS[1]
TOPIC_MAP[0.0] = TOPIC_LABELS[1]

col_topik = "Nama Topik LDA"

if not df_benchmark.empty:
  if col_topik_raw and col_topik_raw in df_benchmark.columns:
    mapped = df_benchmark[col_topik_raw].map(TOPIC_MAP)
    df_benchmark[col_topik] = mapped.fillna(
        df_benchmark[col_topik_raw].astype(str)
    )
  else:
    df_benchmark[col_topik] = "General"

  # Kalau nilainya "General"/kosong, lakukan analisis kata kunci cepat
  mask_invalid = df_benchmark[col_topik].isna() | df_benchmark[
      col_topik
  ].astype(str).str.lower().isin(["general", "nan", "", "none", "null"])

  if mask_invalid.any():
    if col_text_raw and col_text_raw in df_benchmark.columns:
      t_series = df_benchmark[col_text_raw].astype(str).str.lower()
      c4 = t_series.str.contains(
          r"otsu|otsukare|makasih|terimakasih|stream|terima kasih|\bka\b|\bkak\b",
          regex=True,
      )
      c5 = t_series.str.contains(
          r"wkwk|wkwkwk|haha|hahaha|xixi|lol|suka|ngakak|lagi", regex=True
      )
      c3 = t_series.str.contains(
          r"selamat|datang|welcome|live|semoga|\bthe\b", regex=True
      )
      c1 = t_series.str.contains(
          r"halo|hai|bang|malam|pagi|siang|kalian|sil|tris", regex=True
      )

      fallback = pd.Series(TOPIC_LABELS[2], index=df_benchmark.index)
      fallback[c1] = TOPIC_LABELS[1]
      fallback[c3] = TOPIC_LABELS[3]
      fallback[c5] = TOPIC_LABELS[5]
      fallback[c4] = TOPIC_LABELS[4]

      df_benchmark[col_topik] = df_benchmark[col_topik].where(
          ~mask_invalid, fallback
      )
    else:
      df_benchmark[col_topik] = df_benchmark[col_topik].replace(
          ["General", "nan", "", "none", "null"], TOPIC_LABELS[2]
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


# Header utama
st.title("VTuber Live Chat Mining & Analytics System")
st.caption(
    "Dashboard riset untuk memantau sentimen dan topik obrolan penonton"
    " VTuber Indonesia secara langsung maupun berbasis data historis."
)

st.sidebar.markdown("### Navigasi Mode Aplikasi")
mode_pilihan = st.sidebar.radio(
    "Pilih Mode Aplikasi:",
    [
        "Ekstraksi Live Chat (Realtime)",
        "Dashboard Benchmark Dataset (20 VTuber)",
    ],
)

st.sidebar.markdown("---")

# ==========================================================
# MODE 1: EKSTRAKSI LIVE CHAT (REALTIME + ANALYTICS)
# ==========================================================
if mode_pilihan == "Ekstraksi Live Chat (Realtime)":
  st.markdown(
      """
        <div class="extractor-container">
            <h3>Ekstraksi Live Chat dari Satu Video YouTube</h3>
            <p style="color: #A0AEC0; font-size: 0.9rem; margin-bottom: 0;">
                Tempelkan tautan video YouTube (live maupun replay) untuk menjalankan analisis
                secara langsung. Setiap chat akan diklasifikasikan sentimennya lewat model
                <b>Naive Bayes</b>, sementara topik pembicaraannya dipetakan berdasarkan pola
                dari model <b>LDA</b>.
            </p>
        </div>
    """,
      unsafe_allow_html=True,
  )

  st.info(
      "Model yang dipakai pada fitur ini dilatih menggunakan data dari 20 kanal"
      " VTuber yang sudah diteliti sebelumnya."
  )

  if model_nb is None:
    st.warning(
        "Berkas model_sentiment.pkl belum ditemukan di repositori, sehingga"
        " prediksi sentimen untuk sementara memakai nilai bawaan (default)."
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
      "Ekstrak Live Chat", type="primary", use_container_width=True
  )

  if btn_proses:
    original_url = input_url.strip() if input_url else ""

    if not original_url:
      st.warning("Masukkan URL video YouTube live/replay terlebih dahulu.")
    elif not HAS_SCRAPER:
      st.error(
          "Pustaka yt-chat-downloader belum terpasang di server. Periksa"
          " kembali apakah requirements.txt sudah mencakup pustaka ini."
      )
    else:
      clean_url = original_url
      if "/live/" in clean_url:
        clean_url = clean_url.replace("/live/", "/watch?v=")
      if "?si=" in clean_url:
        clean_url = clean_url.split("?si=")[0]

      status_box = st.info(
          "Sedang mengambil live chat, membersihkan teksnya, lalu menjalankan"
          " prediksi sentimen dan pemetaan topik..."
      )

      # quiet=False supaya warning asli dari library (mis. "chat is
      # disabled", "members-only") tercetak ke log Streamlit Cloud,
      # bukan disembunyikan.
      try:
        downloader = YouTubeChatDownloader()
        messages = downloader.download_chat(
            video_url=clean_url, chat_type="live", quiet=False
        )

        extracted_rows = []
        raw_message_count = 0

        for msg in messages:
          raw_message_count += 1
          komentar_asli = msg.get("comment", "")
          if komentar_asli:
            clean_text = preprocess_text(komentar_asli)
            sentiment = prediksi_sentimen_ml(clean_text, komentar_asli)
            topik_lda = deteksi_topik_realtime(clean_text)

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
          if raw_message_count == 0:
            st.error(
                "Live chat tidak ditemukan pada video ini "
                "(0 pesan mentah diterima dari scraper). Kemungkinan "
                "penyebab: (1) replay chat dimatikan oleh streamer untuk "
                "video ini, (2) video butuh login/age-restricted/members-only, "
                "atau (3) IP server Streamlit Cloud diblokir/diarahkan ke "
                "halaman consent oleh YouTube sehingga scraper tidak melihat "
                "chat sama sekali. Cek 'Manage app > Logs' di Streamlit Cloud "
                "untuk pesan warning asli dari library scraper-nya."
            )
          else:
            st.error(
                f"Scraper menerima {raw_message_count} pesan mentah, tapi "
                "semuanya tanpa field 'comment' yang valid — kemungkinan "
                "format respons dari YouTube berubah dan library scraper "
                "perlu di-update ke versi terbaru."
            )
        else:
          df_res = pd.DataFrame(extracted_rows)
          st.session_state["real_extracted_data"] = df_res
          st.success(
              f"Berhasil mengekstrak dan menganalisis **{len(df_res):,} baris**"
              " live chat."
          )

      except Exception as e:
        status_box.empty()
        st.error(f"Terjadi kesalahan saat menarik data: {type(e).__name__}: {e}")
        with st.expander("Detail traceback (untuk debugging)"):
          st.code(traceback.format_exc())

  # Tampilan hasil analisis realtime
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
    unique_users = df_real["Username"].nunique()
    avg_len = df_real["Chat Text"].astype(str).str.len().mean()

    # Parsing timestamp untuk chart berbasis waktu (tidak dipaksa gagal
    # kalau formatnya tidak konsisten dari scraper)
    df_time = df_real.copy()
    df_time["Timestamp_parsed"] = pd.to_datetime(
        df_time["Timestamp"], errors="coerce"
    )
    has_valid_time = df_time["Timestamp_parsed"].notna().sum() >= 3

    st.markdown("---")
    st.subheader("Hasil Analisis Data Realtime")

    # Dikumpulkan supaya bisa dibundel jadi satu laporan PDF di bagian bawah
    charts_for_pdf = {}

    m1, m2, m3, m4 = st.columns(4)
    metric_card(m1, "Total Live Chat Ter-ekstrak", f"{total_real:,}")
    metric_card(
        m2, "Sentimen Positif", f"{pos_pct:.1f}%", f"{pos_real:,} chat", COLOR_POS
    )
    metric_card(
        m3, "Sentimen Negatif", f"{neg_pct:.1f}%", f"{neg_real:,} chat", COLOR_NEG
    )
    metric_card(
        m4, "Penonton Unik Mengobrol", f"{unique_users:,}",
        f"rata-rata {avg_len:.0f} karakter/chat",
    )

    st.markdown("<br>", unsafe_allow_html=True)

    section_header("Sentimen & Topik")
    c_g1, c_g2 = st.columns(2)
    with c_g1:
      st.markdown("##### Proporsi Sentimen (Naive Bayes)")
      fig_sent = px.pie(
          df_real,
          names="Prediksi Sentimen",
          color="Prediksi Sentimen",
          color_discrete_map={"Positif": COLOR_POS, "Negatif": COLOR_NEG},
          hole=0.55,
      )
      st.plotly_chart(style_fig(fig_sent), use_container_width=True)
      charts_for_pdf["Proporsi Sentimen (Naive Bayes)"] = fig_sent

    with c_g2:
      st.markdown("##### Proporsi Topik LDA Dominan")
      fig_topik = px.pie(
          df_real,
          names="Topik LDA",
          hole=0.55,
          color_discrete_sequence=COLOR_THEME,
      )
      st.plotly_chart(style_fig(fig_topik), use_container_width=True)
      charts_for_pdf["Proporsi Topik LDA Dominan"] = fig_topik

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
      charts_for_pdf["Sebaran Sentimen per Topik LDA"] = fig_top_sent

    with c_g4:
      st.markdown("##### 10 Kata Kunci Terbanyak dalam Stream")
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
        charts_for_pdf["10 Kata Kunci Terbanyak"] = fig_words
      else:
        st.info("Belum ada kata bersih yang cukup untuk dihitung.")

    st.markdown("<br>", unsafe_allow_html=True)
    section_header("Aktivitas & Partisipasi Penonton")

    c_g5, c_g6 = st.columns(2)
    with c_g5:
      st.markdown("##### 10 Penonton Paling Aktif Mengobrol")
      top_users = (
          df_real["Username"].value_counts().head(10).reset_index()
      )
      top_users.columns = ["Username", "Jumlah Chat"]
      fig_users = px.bar(
          top_users,
          x="Jumlah Chat",
          y="Username",
          orientation="h",
          color_discrete_sequence=["#8B5CF6"],
      )
      fig_users.update_layout(yaxis=dict(autorange="reversed"))
      st.plotly_chart(style_fig(fig_users), use_container_width=True)
      charts_for_pdf["10 Penonton Paling Aktif"] = fig_users

    with c_g6:
      st.markdown("##### Distribusi Panjang Pesan per Sentimen")
      df_real_len = df_real.copy()
      df_real_len["Panjang Pesan"] = (
          df_real_len["Chat Text"].astype(str).str.len()
      )
      fig_len = px.box(
          df_real_len,
          x="Prediksi Sentimen",
          y="Panjang Pesan",
          color="Prediksi Sentimen",
          color_discrete_map={"Positif": COLOR_POS, "Negatif": COLOR_NEG},
      )
      st.plotly_chart(style_fig(fig_len), use_container_width=True)
      charts_for_pdf["Distribusi Panjang Pesan per Sentimen"] = fig_len

    if has_valid_time:
      df_time = df_time.dropna(subset=["Timestamp_parsed"]).sort_values(
          "Timestamp_parsed"
      )

      # Interval mengikuti durasi total stream, supaya video yang berjam-jam
      # tidak menghasilkan ratusan titik data yang bikin grafik bergerigi
      durasi_menit = (
          df_time["Timestamp_parsed"].max() - df_time["Timestamp_parsed"].min()
      ).total_seconds() / 60
      if durasi_menit <= 60:
        interval_menit = 5
      elif durasi_menit <= 180:
        interval_menit = 10
      elif durasi_menit <= 300:
        interval_menit = 20
      else:
        interval_menit = 30

      df_time["Menit ke-"] = df_time["Timestamp_parsed"].dt.floor(
          f"{interval_menit}min"
      )

      c_g7, c_g8 = st.columns(2)
      with c_g7:
        st.markdown(
            f"##### Volume Chat Sepanjang Stream (per {interval_menit} menit)"
        )
        vol_per_menit = (
            df_time.groupby("Menit ke-").size().reset_index(name="Jumlah Chat")
        )
        fig_vol = px.area(
            vol_per_menit,
            x="Menit ke-",
            y="Jumlah Chat",
            color_discrete_sequence=["#6366F1"],
        )
        st.plotly_chart(style_fig(fig_vol), use_container_width=True)
        charts_for_pdf[f"Volume Chat Sepanjang Stream (per {interval_menit} menit)"] = fig_vol

      with c_g8:
        st.markdown(
            f"##### Tren Sentimen Sepanjang Stream (per {interval_menit} menit)"
        )
        sent_per_menit = (
            df_time.groupby(["Menit ke-", "Prediksi Sentimen"])
            .size()
            .reset_index(name="Jumlah Chat")
        )
        fig_trend = px.line(
            sent_per_menit,
            x="Menit ke-",
            y="Jumlah Chat",
            color="Prediksi Sentimen",
            color_discrete_map={"Positif": COLOR_POS, "Negatif": COLOR_NEG},
        )
        st.plotly_chart(style_fig(fig_trend), use_container_width=True)
        charts_for_pdf[f"Tren Sentimen Sepanjang Stream (per {interval_menit} menit)"] = fig_trend
    else:
      st.caption(
          "Grafik tren berbasis waktu belum bisa ditampilkan karena format"
          " timestamp dari video ini tidak terbaca dengan jelas."
      )

    st.markdown("---")

    st.markdown("### Tabel Keseluruhan Hasil Ekstraksi Live Chat")
    st.dataframe(df_real, use_container_width=True)

    def convert_df_to_excel(df_to_download):
      buffer = io.BytesIO()
      with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df_to_download.to_excel(writer, index=False)
      return buffer.getvalue()

    def _fig_for_print(fig, judul_chart):
      # Chart di layar pakai tema gelap. Kalau dipakai langsung untuk
      # export PNG/PDF berlatar putih, teksnya jadi nyaris tak kelihatan —
      # jadi dibuat salinan bertema terang khusus cetak, tanpa mengubah
      # versi yang tampil di dashboard.
      fig_print = go.Figure(fig)
      fig_print.update_layout(
          title=dict(text=judul_chart, x=0.02, xanchor="left",
                      font=dict(size=18, color="#111827")),
          paper_bgcolor="white",
          plot_bgcolor="white",
          font=dict(family="Arial, Helvetica, sans-serif", color="#111827", size=13),
          legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right",
                       x=1, font=dict(color="#111827")),
          margin=dict(l=100, r=40, t=70, b=70),
      )
      fig_print.update_xaxes(color="#111827", gridcolor="#E5E7EB")
      fig_print.update_yaxes(color="#111827", gridcolor="#E5E7EB")
      return fig_print

    def convert_charts_to_onepage_pdf(fig_dict, judul, ringkasan_baris):
      from PIL import Image, ImageDraw, ImageFont

      COLS = 2
      CELL_W, CELL_H = 950, 620
      PAD = 24
      HEADER_H = 170

      n = len(fig_dict)
      rows = (n + COLS - 1) // COLS
      canvas_w = PAD * (COLS + 1) + CELL_W * COLS
      canvas_h = HEADER_H + PAD * (rows + 1) + CELL_H * rows

      canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
      draw = ImageDraw.Draw(canvas)

      try:
        font_title = ImageFont.truetype("DejaVuSans-Bold.ttf", 34)
        font_body = ImageFont.truetype("DejaVuSans.ttf", 20)
      except Exception:
        font_title = ImageFont.load_default()
        font_body = ImageFont.load_default()

      draw.text((PAD, 24), judul, fill="#111827", font=font_title)
      y_text = 74
      for baris in ringkasan_baris:
        draw.text((PAD, y_text), baris, fill="#374151", font=font_body)
        y_text += 28

      for i, (nama_chart, fig) in enumerate(fig_dict.items()):
        fig_print = _fig_for_print(fig, nama_chart)
        img_bytes = fig_print.to_image(
            format="png", width=CELL_W, height=CELL_H, scale=2
        )
        chart_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        chart_img = chart_img.resize((CELL_W, CELL_H))

        col = i % COLS
        row = i // COLS
        x = PAD + col * (CELL_W + PAD)
        y = HEADER_H + PAD + row * (CELL_H + PAD)
        canvas.paste(chart_img, (x, y))
        draw.rectangle(
            [x, y, x + CELL_W, y + CELL_H], outline="#D1D5DB", width=2
        )

      buffer = io.BytesIO()
      canvas.save(buffer, format="PDF", resolution=150.0)
      return buffer.getvalue()

    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
      excel_bytes = convert_df_to_excel(df_real)
      st.download_button(
          label="Unduh Data Live Chat Ter-ekstrak (.xlsx)",
          data=excel_bytes,
          file_name="hasil_ekstraksi_livechat_realtime.xlsx",
          mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
          type="primary",
          use_container_width=True,
      )

    with col_dl2:
      if not HAS_KALEIDO or not HAS_PIL:
        st.button(
            "Unduh Seluruh Grafik (.pdf)",
            disabled=True,
            use_container_width=True,
            help=(
                "Pustaka kaleido dan/atau Pillow belum terpasang di server."
                " Tambahkan kaleido==0.2.1 ke requirements.txt untuk"
                " mengaktifkan fitur ini."
            ),
        )
      else:
        if st.button(
            "Siapkan & Unduh Seluruh Grafik (.pdf)",
            use_container_width=True,
        ):
          with st.spinner("Menyusun laporan PDF satu halaman dari seluruh grafik..."):
            ringkasan = [
                f"Total live chat ter-ekstrak: {total_real:,}",
                f"Sentimen positif: {pos_real:,} ({pos_pct:.1f}%)   |"
                f"   Sentimen negatif: {neg_real:,} ({neg_pct:.1f}%)"
                f"   |   Penonton unik: {unique_users:,}",
            ]
            pdf_bytes = convert_charts_to_onepage_pdf(
                charts_for_pdf,
                "Laporan Grafik Analisis Live Chat Realtime",
                ringkasan,
            )
          st.download_button(
              label="Unduh Laporan Grafik (.pdf)",
              data=pdf_bytes,
              file_name="laporan_grafik_livechat_realtime.pdf",
              mime="application/pdf",
              type="primary",
              use_container_width=True,
          )

# ==========================================================
# MODE 2: DASHBOARD BENCHMARK DATASET (20 VTUBER)
# ==========================================================
else:
  if df_benchmark.empty:
    st.error("File dataset benchmark (.xlsx) tidak ditemukan di repositori.")
    st.info(
        "Pastikan berkas `data_vtuber_labeled.xlsx` atau"
        " `hasil_akhir_analisis_skripsi.xlsx` sudah tersedia di repo."
    )
  else:
    st.sidebar.markdown("### Filter Panel Global")
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
    selected_streams = all_streams

    df_filtered = df_benchmark.copy()
    if col_vtuber in df_benchmark.columns and selected_vtubers:
      df_filtered = df_filtered[df_filtered[col_vtuber].isin(selected_vtubers)]
    if col_stream in df_benchmark.columns and selected_streams:
      df_filtered = df_filtered[df_filtered[col_stream].isin(selected_streams)]

    if df_filtered.empty:
      st.warning("Data kosong untuk kombinasi filter ini.")
    else:
      tab1, tab2, tab3 = st.tabs([
          "Ringkasan & Referensi LDA",
          "Profil & Perbandingan VTuber",
          "Analisis Kategori Stream & Korelasi",
      ])

      # TAB 1: Ringkasan & LDA
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

        c1, c2, c3 = st.columns(3)
        metric_card(c1, "Total Chat Dianalisis", f"{total_chat:,}")
        metric_card(
            c2, "Sentimen Positif", f"{pos_pct:.1f}%", f"{pos_chat:,} chat", COLOR_POS
        )
        metric_card(
            c3, "Sentimen Negatif", f"{neg_pct:.1f}%", f"{neg_chat:,} chat", COLOR_NEG
        )

        st.markdown("<br>", unsafe_allow_html=True)

        with st.expander(
            "Kata Kunci Acuan untuk Tiap Topik LDA (klik untuk membuka)",
            expanded=False,
        ):
          for _num, _label in TOPIC_LABELS.items():
            st.markdown(f"* **{_label}** — *{TOPIC_KEYWORD_HINTS[_num]}*")

        col_a, col_b = st.columns(2)
        with col_a:
          st.markdown("##### Proporsi Sentimen Chat")
          if col_sentimen in df_filtered.columns:
            fig_s = px.pie(
                df_filtered,
                names=col_sentimen,
                color=col_sentimen,
                color_discrete_map={
                    "Positif": COLOR_POS,
                    "Negatif": COLOR_NEG,
                },
                hole=0.6,
            )
            st.plotly_chart(style_fig(fig_s), use_container_width=True)

        with col_b:
          st.markdown("##### Proporsi Topik LDA Dominan")
          fig_t = px.pie(
              df_filtered,
              names=col_topik,
              hole=0.6,
              color_discrete_sequence=COLOR_THEME,
          )
          st.plotly_chart(style_fig(fig_t), use_container_width=True)

        st.markdown("---")
        st.markdown("##### Tabel Distribusi Frekuensi Topik LDA")
        topik_df = df_filtered[col_topik].value_counts().reset_index()
        topik_df.columns = ["Nama Topik LDA", "Jumlah Chat"]
        topik_df["Persentase (%)"] = (
            topik_df["Jumlah Chat"] / total_chat * 100
        ).round(2)
        st.dataframe(topik_df, use_container_width=True)

      # TAB 2: Profil & Perbandingan VTuber
      with tab2:
        st.markdown("### Profil dan Informasi Dataset 20 VTuber")
        st.caption(
            "Gambaran umum sampel penelitian, dilengkapi perbandingan"
            " performa antar kanal independen."
        )

        col_prof_left, col_prof_right = st.columns([1, 1.4])

        with col_prof_left:
          st.markdown("#### Ringkasan Karakteristik Sampel")
          st.markdown("""
              Dataset ini terdiri dari **20 VTuber independen** yang aktif di ranah digital Indonesia.

              * **Rentang Pengambilan Data:** Februari 2026 hingga Juli 2026.
              * **Periode Pengerjaan Riset:** Mei hingga Juli 2026.
              * **Fokus Analisis:** Mengukur interaksi penonton, performa live stream, dominasi topik, serta sentimen obrolan real-time berbasis model pembelajaran mesin.
              """)
          st.info(
              "Data ini dipakai sebagai acuan pembanding untuk melihat pola"
              " interaksi audiens pada kanal-kanal VTuber independen."
          )

        with col_prof_right:
          st.markdown("#### Tabel Master 20 VTuber Independen")
          data_20_vtuber = [
              {
                  "No": 1,
                  "Nama Saluran": "Jelly si Curut Bodas Ch.",
                  "Username": "@JellyCB14",
                  "Jumlah Pengikut": 24600,
                  "Total Live": 73,
                  "Kategori Konten": "Gaming/ Collaboration",
              },
              {
                  "No": 2,
                  "Nama Saluran": "Vixynaa Mina ch.",
                  "Username": "@vixynaa",
                  "Jumlah Pengikut": 8070,
                  "Total Live": 29,
                  "Kategori Konten": "Gaming",
              },
              {
                  "No": 3,
                  "Nama Saluran": "Chiachasm / チアキャズム",
                  "Username": "@chiachasm",
                  "Jumlah Pengikut": 8090,
                  "Total Live": 70,
                  "Kategori Konten": "Gaming, Collaboration",
              },
              {
                  "No": 4,
                  "Nama Saluran": "Avnore Daryush",
                  "Username": "@avnoredaryush",
                  "Jumlah Pengikut": 8740,
                  "Total Live": 91,
                  "Kategori Konten": "Gaming",
              },
              {
                  "No": 5,
                  "Nama Saluran": "Cheihime Ch.",
                  "Username": "@cheihimechei",
                  "Jumlah Pengikut": 1270,
                  "Total Live": 38,
                  "Kategori Konten": "Freetalk, Gaming",
              },
              {
                  "No": 6,
                  "Nama Saluran": "Aymana Manisha",
                  "Username": "@AymanaManisha",
                  "Jumlah Pengikut": 6970,
                  "Total Live": 68,
                  "Kategori Konten": "Freetalk, Gaming",
              },
              {
                  "No": 7,
                  "Nama Saluran": "Sachunya Milující Ch.【Atelier ID】",
                  "Username": "@Sachunya.Milujici",
                  "Jumlah Pengikut": 1460,
                  "Total Live": 109,
                  "Kategori Konten": "Gaming, Karaoke, Collaboration",
              },
              {
                  "No": 8,
                  "Nama Saluran": "Cecillia Hanarisu Ch.",
                  "Username": "@CecilliaHanarisu",
                  "Jumlah Pengikut": 6990,
                  "Total Live": 54,
                  "Kategori Konten": "Freetalk, Gaming",
              },
              {
                  "No": 9,
                  "Nama Saluran": "Shieru Eris Ch.",
                  "Username": "@ShieruEris",
                  "Jumlah Pengikut": 3030,
                  "Total Live": 76,
                  "Kategori Konten": "Gaming, Collaboration",
              },
              {
                  "No": 10,
                  "Nama Saluran": "Riverio Akira Ch.",
                  "Username": "@RiverioAkiraCh",
                  "Jumlah Pengikut": 15800,
                  "Total Live": 58,
                  "Kategori Konten": "Gaming, Collaboration",
              },
              {
                  "No": 11,
                  "Nama Saluran": "Lavatia Laflarld",
                  "Username": "@lavatialaflarldofficial",
                  "Jumlah Pengikut": 20000,
                  "Total Live": 72,
                  "Kategori Konten": "Gaming, Collaboration",
              },
              {
                  "No": 12,
                  "Nama Saluran": "Takaki Naoki Youtube Channel",
                  "Username": "@takakinaoki",
                  "Jumlah Pengikut": 3370,
                  "Total Live": 26,
                  "Kategori Konten": "Gaming, Collaboration",
              },
              {
                  "No": 13,
                  "Nama Saluran": "Bianca Hantu",
                  "Username": "@BiancaHantu",
                  "Jumlah Pengikut": 7590,
                  "Total Live": 35,
                  "Kategori Konten": "Gaming, Karaoke, Collaboration",
              },
              {
                  "No": 14,
                  "Nama Saluran": "Nova Constella",
                  "Username": "@NovaConstella",
                  "Jumlah Pengikut": 4700,
                  "Total Live": 105,
                  "Kategori Konten": "Gaming, Collaboration",
              },
              {
                  "No": 15,
                  "Nama Saluran": "aisu kohi【NeoRise】",
                  "Username": "@Aisukohi_1",
                  "Jumlah Pengikut": 6580,
                  "Total Live": 79,
                  "Kategori Konten": "Collaboration",
              },
              {
                  "No": 16,
                  "Nama Saluran": "Dezzoko Yoezaro",
                  "Username": "@dezzokoyoezaro",
                  "Jumlah Pengikut": 4920,
                  "Total Live": 136,
                  "Kategori Konten": "Freetalk, Gaming",
              },
              {
                  "No": 17,
                  "Nama Saluran": "Teaqillla Ch. 【MIQELA】",
                  "Username": "@TeaqilllaVT",
                  "Jumlah Pengikut": 3340,
                  "Total Live": 100,
                  "Kategori Konten": "Freetalk, Gaming, Collaboration",
              },
              {
                  "No": 18,
                  "Nama Saluran": "Merra Merona",
                  "Username": "@MerraMeronaVTuberID",
                  "Jumlah Pengikut": 5250,
                  "Total Live": 79,
                  "Kategori Konten": "Gaming, Collaboration",
              },
              {
                  "No": 19,
                  "Nama Saluran": "Silveryshore Ch.【CozyCazt】",
                  "Username": "@silveryshore",
                  "Jumlah Pengikut": 9450,
                  "Total Live": 52,
                  "Kategori Konten": "Collaboration",
              },
              {
                  "No": 20,
                  "Nama Saluran": "Tris Fushimi Ch.",
                  "Username": "@trisfushimi",
                  "Jumlah Pengikut": 4040,
                  "Total Live": 55,
                  "Kategori Konten": "Gaming, Collaboration",
              },
          ]
          df_table_20 = pd.DataFrame(data_20_vtuber)
          st.dataframe(df_table_20, use_container_width=True, height=350)

        st.markdown("---")
        st.markdown("### Filter Individual 1 VTuber")
        selected_single_vt = (
            st.selectbox(
                "Pilih 1 VTuber untuk melihat analisis detailnya:", all_vtubers
            )
            if all_vtubers
            else None
        )

        if selected_single_vt:
          df_single = df_benchmark[
              df_benchmark[col_vtuber] == selected_single_vt
          ]

          p1, p2, p3 = st.columns(3)
          with p1:
            st.markdown(f"##### Kategori Stream ({selected_single_vt})")
            if col_stream in df_single.columns:
              fig_single_cat = px.pie(
                  df_single,
                  names=col_stream,
                  hole=0.55,
                  color_discrete_sequence=COLOR_THEME,
              )
              st.plotly_chart(
                  style_fig(fig_single_cat), use_container_width=True
              )

          with p2:
            st.markdown(f"##### Topik LDA Dominan ({selected_single_vt})")
            fig_single_top = px.pie(
                df_single,
                names=col_topik,
                hole=0.55,
                color_discrete_sequence=COLOR_THEME,
            )
            st.plotly_chart(
                style_fig(fig_single_top), use_container_width=True
            )

          with p3:
            st.markdown(f"##### Sebaran Sentimen ({selected_single_vt})")
            if col_sentimen in df_single.columns:
              fig_single_sent = px.pie(
                  df_single,
                  names=col_sentimen,
                  hole=0.55,
                  color_discrete_map={
                      "Positif": COLOR_POS,
                      "Negatif": COLOR_NEG,
                  },
              )
              st.plotly_chart(
                  style_fig(fig_single_sent), use_container_width=True
              )

        st.markdown("---")
        st.markdown("### Diagram Perbandingan Antar VTuber")

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

      # TAB 3: Analisis Kategori Stream & Korelasi
      with tab3:
        st.markdown("### Analisis Komparatif & Korelasi Kategori Stream")
        st.caption(
            "Membedah keterkaitan antara format tayangan stream dengan pola"
            " respon sentimen dan topik pembicaraan audiens."
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
              "##### Tabel Ringkasan Korelasi Kategori Stream & Sentimen"
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
