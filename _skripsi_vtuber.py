import io
import os
import re
import time
import traceback
from collections import Counter

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

try:
  from PIL import Image

  HAS_PIL = True
except Exception:
  HAS_PIL = False

try:
  import kaleido

  HAS_KALEIDO = True
except Exception:
  HAS_KALEIDO = False

st.set_page_config(
    page_title="VTuber Analytics & Live Chat Extractor",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
  from Sastrawi.Dictionary.ArrayDictionary import ArrayDictionary
  from Sastrawi.StopWordRemover.StopWordRemover import StopWordRemover

  HAS_SASTRAWI = True
except Exception:
  HAS_SASTRAWI = False


@st.cache_resource
def load_ml_model():
  try:
    model = joblib.load("model_sentiment.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer
  except Exception:
    return None, None


model_nb, tfidf_vec = load_ml_model()


CUSTOM_STOPWORDS_TAMBAHAN = [
    "ga", "gak", "ngga", "nggak", "gk", "yg", "krn", "karna", "tp",
    "dgn", "dg", "utk", "sm", "nih", "sih", "deh", "dong", "kok",
    "loh", "lho", "ya", "iya", "oke", "ok", "aja", "doang", "nya",
    "kan", "kali", "gitu", "gini", "gimana", "gmn", "emang", "emg",
    "udah", "udh", "blm", "belom", "trs", "jd", "jadinya", "biar",
    "lagi", "ka", "kak", "the", "dan", "lah", "dengan",
    "banget", "bgt", "min", "kk", "cak", "bro", "sist", "guys", "gaes",
    "an", "nan", "wah", "eh", "ehh", "hmm", "hm", "yah", "yaa", "yaah",
    "atau", "tetapi", "melainkan", "karena", "jika", "walaupun",
    "sedangkan", "di", "ke", "dari", "pada", "dalam", "oleh", "kepada",
    "daripada", "saya", "aku", "kamu", "dia", "mereka", "kita", "kami",
    "ia", "ini", "itu", "tersebut", "sang", "para", "yang", "adakah",
    "pun", "kah", "andaikata", "saja", "kalian", "adalah", "akan",
    "ada", "juga", "harus", "bisa", "dapat", "kalau",
]


@st.cache_resource
def load_sastrawi_tools():
  if not HAS_SASTRAWI:
    return None, None, set(CUSTOM_STOPWORDS_TAMBAHAN)
  try:
    stemmer = StemmerFactory().create_stemmer()

    daftar_stopword_default = StopWordRemoverFactory().get_stop_words()
    daftar_stopword_gabungan = list(
        set(daftar_stopword_default + CUSTOM_STOPWORDS_TAMBAHAN)
    )
    kamus_stopword = ArrayDictionary(daftar_stopword_gabungan)
    stopword_remover = StopWordRemover(kamus_stopword)

    return stemmer, stopword_remover, set(daftar_stopword_gabungan)
  except Exception as e:
    st.session_state["sastrawi_load_error"] = str(e)
    return None, None, set(CUSTOM_STOPWORDS_TAMBAHAN)


stemmer, stopword_remover, STOPWORD_SET_GABUNGAN = load_sastrawi_tools()


def bersihkan_ulang_dari_stopword(teks):
  if not isinstance(teks, str) or not teks:
    return ""
  kata_list = [k for k in teks.split() if k and k not in STOPWORD_SET_GABUNGAN]
  return " ".join(kata_list)


def _normalisasi_dasar(text):
  if not text or not isinstance(text, str):
    return ""
  text = text.lower()
  text = re.sub(r"http\S+|www\S+|https\S+", "", text)
  text = re.sub(r"[^a-zA-Z\s]", "", text)
  return text.strip()


def normalize_untuk_deteksi(text):
  teks = _normalisasi_dasar(text)
  if not teks:
    return ""
  if HAS_SASTRAWI and stemmer:
    try:
      teks = stemmer.stem(teks)
    except Exception:
      pass
  return teks


def preprocess_text(text):
  text = _normalisasi_dasar(text)
  if not text:
    return ""

  if HAS_SASTRAWI and stopword_remover and stemmer:
    try:
      text = stopword_remover.remove(text)
      text = stemmer.stem(text)
    except Exception as e:
      if "preprocess_error_shown" not in st.session_state:
        st.session_state["preprocess_error_shown"] = True
        st.warning(
            f"Pembersihan Sastrawi sempat gagal untuk sebagian pesan"
            f" ({type(e).__name__}: {e}). Kata stopword pada pesan itu"
            f" TIDAK ikut dibersihkan."
        )
  else:
    if "sastrawi_missing_warned" not in st.session_state:
      st.session_state["sastrawi_missing_warned"] = True
      pesan_error = st.session_state.get("sastrawi_load_error", "")
      st.warning(
          "Sastrawi tidak aktif di environment ini, jadi stopword (aku, di,"
          " itu, ga, yang, dst) TIDAK ikut dibersihkan dari teks."
          f" {('Detail: ' + pesan_error) if pesan_error else ''}"
          " Pastikan 'Sastrawi' ada di requirements.txt dan ke-install"
          " dengan benar di server."
      )
  return text.strip()


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


TOPIC_LABELS = {
    1: "Topik 1: Sapaan & Absen Awal Streaming",
    2: "Topik 2: Pamitan & Penutup Streaming",
    3: "Topik 3: Pujian & Dukungan ke Streamer",
    4: "Topik 4: Candaan & Reaksi Spontan",
    5: "Topik 5: Obrolan Bebas / Lain-lain",
}

TOPIC_SIGNAL_KEYWORDS = {
    1: [
        "halo", "hai", "haii", "hallo", "met", "selamat", "datang",
        "welcome", "pagi", "siang", "sore", "malam", "hadir", "absen",
        "nyimak", "nonton", "mulai", "live",
    ],
    2: [
        "otsu", "otsukare", "otsukaresama", "dadah", "bubye", "bye",
        "sampaijumpa", "udahan", "selesai", "cape", "capek", "pulang",
        "tidur", "istirahat", "pamit", "gnight", "goodnight", "off dulu",
    ],
    3: [
        "keren", "mantap", "top", "gg", "hebat", "jago", "semangat",
        "support", "cakep", "kece", "juara", "salut", "bagus", "apik",
        "bangga", "makasih", "terimakasih", "terima kasih",
    ],
    4: [
        "wkwk", "wkwkwk", "wkwkwkwk", "haha", "hahaha", "xixi", "lol",
        "ngakak", "kocak", "lucu", "astaga", "anjay", "gokil", "receh",
        "kaget",
    ],
}


def top_words_per_topic(df_sumber, kolom_teks, kolom_topik, n_kata=8, min_panjang=2):
  hasil = {}
  if (
      df_sumber is None
      or df_sumber.empty
      or kolom_teks not in df_sumber.columns
      or kolom_topik not in df_sumber.columns
  ):
    return hasil
  for topik, grup in df_sumber.groupby(kolom_topik):
    semua_kata = " ".join(grup[kolom_teks].dropna().astype(str)).split()
    semua_kata = [k for k in semua_kata if len(k) >= min_panjang]
    hasil[topik] = Counter(semua_kata).most_common(n_kata)
  return hasil


def _match_any_kata(teks, daftar_kata):
  return any(re.search(rf"\b{re.escape(k)}\b", teks) for k in daftar_kata)


def deteksi_topik_realtime(teks_untuk_deteksi):
  if not teks_untuk_deteksi:
    return TOPIC_LABELS[5]
  t = str(teks_untuk_deteksi).lower()

  if _match_any_kata(t, TOPIC_SIGNAL_KEYWORDS[1]):
    return TOPIC_LABELS[1]
  elif _match_any_kata(t, TOPIC_SIGNAL_KEYWORDS[2]):
    return TOPIC_LABELS[2]
  elif _match_any_kata(t, TOPIC_SIGNAL_KEYWORDS[3]):
    return TOPIC_LABELS[3]
  elif _match_any_kata(t, TOPIC_SIGNAL_KEYWORDS[4]):
    return TOPIC_LABELS[4]
  else:
    return TOPIC_LABELS[5]


STANDARD_YOUTUBE_CATEGORIES = [
    "Gaming",
    "Entertainment",
    "Music",
    "People & Blogs",
    "Comedy",
    "Education",
    "Howto & Style",
    "Film & Animation",
    "Sports",
    "Travel & Events",
    "News & Politics",
    "Science & Technology",
    "Autos & Vehicles",
    "Pets & Animals",
    "Nonprofits & Activism",
]


def _sederhanakan_topic_category(url_wikipedia):
  return url_wikipedia.rstrip("/").split("/")[-1].replace("_", " ")


def _dapatkan_youtube_api_key():
  try:
    return st.secrets["YOUTUBE_API_KEY"]
  except Exception:
    return None


def ambil_kategori_channel_youtube(video_url, api_key):
  if not api_key:
    return None
  try:
    match_video_id = re.search(
        r"(?:v=|youtu\.be/|/live/)([A-Za-z0-9_-]{11})", video_url
    )
    if not match_video_id:
      return None
    video_id = match_video_id.group(1)

    resp_video = requests.get(
        "https://www.googleapis.com/youtube/v3/videos",
        params={"part": "snippet", "id": video_id, "key": api_key},
        timeout=10,
    )
    resp_video.raise_for_status()
    items_video = resp_video.json().get("items", [])
    if not items_video:
      return None
    channel_id = items_video[0]["snippet"]["channelId"]

    resp_channel = requests.get(
        "https://www.googleapis.com/youtube/v3/channels",
        params={"part": "topicDetails", "id": channel_id, "key": api_key},
        timeout=10,
    )
    resp_channel.raise_for_status()
    items_channel = resp_channel.json().get("items", [])
    if not items_channel:
      return None
    urls_topik = items_channel[0].get("topicDetails", {}).get(
        "topicCategories", []
    )
    hasil = [_sederhanakan_topic_category(u) for u in urls_topik]
    return hasil or None
  except Exception:
    return None


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

    .kategori-chip-wrap {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 10px;
    }
    .kategori-chip {
        display: inline-block;
        font-size: 0.8rem;
        font-weight: 600;
        background: rgba(255, 255, 255, 0.06);
        border: 1.5px solid rgba(99, 102, 241, 0.35);
        border-radius: 999px;
        padding: 5px 14px;
    }

    div[data-testid="stExpander"] { border-radius: 12px; border-color: rgba(255,255,255,0.08); }
    div[data-testid="stTabs"] button { font-weight: 600; }
    </style>
""",
    unsafe_allow_html=True,
)


def metric_card(col, title, value, sub=None, color=None):
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


def _pilih_file_dataset():
  files = [f for f in os.listdir(".") if f.endswith(".xlsx")]
  if not files:
    return None
  if "data_vtuber_labeled.xlsx" in files:
    return "data_vtuber_labeled.xlsx"
  elif "hasil_akhir_analisis_skripsi.xlsx" in files:
    return "hasil_akhir_analisis_skripsi.xlsx"
  else:
    return files[0]


@st.cache_data
def load_benchmark_data(target_file, _mtime_signature):
  if not target_file:
    return pd.DataFrame()
  df = pd.read_excel(target_file)
  df.columns = df.columns.astype(str).str.strip()
  return df


_target_file_dataset = _pilih_file_dataset()
_mtime_dataset = (
    os.path.getmtime(_target_file_dataset) if _target_file_dataset else 0
)
df_benchmark = load_benchmark_data(_target_file_dataset, _mtime_dataset)


def find_col(df, possible_names, default=None):
  if df.empty:
    return default
  for name in possible_names:
    for col in df.columns:
      if name.lower() in str(col).lower():
        return col
  return default


_KATEGORI_KOSONG_PATTERN = {
    "", "nan", "none", "null", "-", "tidak diketahui", "unknown",
    "tidak diketahui,", "kosong",
}


def explode_kategori_channel(df, kolom_kategori):
  if df is None or df.empty or not kolom_kategori or kolom_kategori not in df.columns:
    return df
  df_exp = df.copy()
  df_exp[kolom_kategori] = df_exp[kolom_kategori].astype(str).str.split(",")
  df_exp = df_exp.explode(kolom_kategori)
  df_exp[kolom_kategori] = df_exp[kolom_kategori].str.strip()
  df_exp = df_exp[
      ~df_exp[kolom_kategori].str.lower().isin(_KATEGORI_KOSONG_PATTERN)
  ]
  return df_exp


col_vtuber = find_col(
    df_benchmark, ["vtuber", "nama", "channel", "creator"], "VTuber Name"
)
col_stream = find_col(
    df_benchmark,
    [
        "kategori channel (youtube)", "kategori channel", "channel category",
        "official category", "kategori resmi", "topic categor",
        "kategori", "category", "stream", "type",
    ],
    "Stream Type",
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

col_text_bersih_app = None
if col_text_raw and col_text_raw in df_benchmark.columns:
  col_text_bersih_app = "Pesan Bersih (Stopword App Terbaru)"
  df_benchmark[col_text_bersih_app] = (
      df_benchmark[col_text_raw]
      .astype(str)
      .str.lower()
      .str.replace(r"[^a-z\s]", "", regex=True)
      .apply(bersihkan_ulang_dari_stopword)
  )

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

  mask_invalid = df_benchmark[col_topik].isna() | df_benchmark[
      col_topik
  ].astype(str).str.lower().isin(["general", "nan", "", "none", "null"])

  if mask_invalid.any():
    col_teks_untuk_klasifikasi = col_text_bersih_app or col_text_raw

    if col_teks_untuk_klasifikasi and col_teks_untuk_klasifikasi in df_benchmark.columns:
      t_series = df_benchmark[col_teks_untuk_klasifikasi].astype(str).str.lower()

      def _buat_pola(daftar_kata):
        return "|".join(rf"\b{re.escape(k)}\b" for k in daftar_kata)

      c1 = t_series.str.contains(_buat_pola(TOPIC_SIGNAL_KEYWORDS[1]), regex=True)
      c2 = t_series.str.contains(_buat_pola(TOPIC_SIGNAL_KEYWORDS[2]), regex=True)
      c3 = t_series.str.contains(_buat_pola(TOPIC_SIGNAL_KEYWORDS[3]), regex=True)
      c4 = t_series.str.contains(_buat_pola(TOPIC_SIGNAL_KEYWORDS[4]), regex=True)

      fallback = pd.Series(TOPIC_LABELS[5], index=df_benchmark.index)
      fallback[c4] = TOPIC_LABELS[4]
      fallback[c3] = TOPIC_LABELS[3]
      fallback[c2] = TOPIC_LABELS[2]
      fallback[c1] = TOPIC_LABELS[1]

      df_benchmark[col_topik] = df_benchmark[col_topik].where(
          ~mask_invalid, fallback
      )
    else:
      df_benchmark[col_topik] = df_benchmark[col_topik].replace(
          ["General", "nan", "", "none", "null"], TOPIC_LABELS[5]
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

KATEGORI_COLOR_PALETTE = (
    px.colors.qualitative.Alphabet
    + px.colors.qualitative.Dark24
    + px.colors.qualitative.Light24
)


def get_kategori_color_map(df, kolom_kategori):
  if df is None or df.empty or kolom_kategori not in df.columns:
    return {}
  daftar_kategori_unik = sorted(
      df[kolom_kategori].dropna().astype(str).unique().tolist()
  )
  return {
      kat: KATEGORI_COLOR_PALETTE[i % len(KATEGORI_COLOR_PALETTE)]
      for i, kat in enumerate(daftar_kategori_unik)
  }


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

  input_url = st.text_input(
      "URL Video YouTube Stream Replay",
      placeholder=(
          "https://www.youtube.com/live/... atau"
          " https://www.youtube.com/watch?v=..."
      ),
  )
  st.caption(
      "Kategori channel dideteksi otomatis dari kategori resmi YouTube."
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

      youtube_api_key = _dapatkan_youtube_api_key()
      hasil_auto = (
          ambil_kategori_channel_youtube(clean_url, youtube_api_key)
          if youtube_api_key
          else None
      )
      kategori_final = ", ".join(hasil_auto) if hasil_auto else "Tidak Diketahui"

      status_box = st.info(
          "Sedang mengambil live chat, membersihkan teksnya, lalu menjalankan"
          " prediksi sentimen dan pemetaan topik..."
      )

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
            teks_deteksi = normalize_untuk_deteksi(komentar_asli)
            sentiment = prediksi_sentimen_ml(clean_text, komentar_asli)
            topik_lda = deteksi_topik_realtime(teks_deteksi)

            extracted_rows.append({
                "Username": msg.get("user_display_name", "Anonymous"),
                "Chat Text": komentar_asli,
                "Timestamp": msg.get("datetime", ""),
                "Pesan Bersih (Sastrawi)": clean_text,
                "Prediksi Sentimen": sentiment,
                "Topik LDA": topik_lda,
                "Kategori Channel (YouTube)": kategori_final,
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
                "semuanya tanpa field 'comment' yang valid, kemungkinan "
                "format respons dari YouTube berubah dan library scraper "
                "perlu di-update ke versi terbaru."
            )
        else:
          df_res = pd.DataFrame(extracted_rows)
          st.session_state["real_extracted_data"] = df_res
          st.success(
              f"Berhasil mengekstrak dan menganalisis **{len(df_res):,} baris**"
              f" live chat. Kategori channel: **{kategori_final}**."
          )

      except Exception as e:
        status_box.empty()
        st.error(f"Terjadi kesalahan saat menarik data: {type(e).__name__}: {e}")
        with st.expander("Detail traceback (untuk debugging)"):
          st.code(traceback.format_exc())

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

    df_time = df_real.copy()
    df_time["Timestamp_parsed"] = pd.to_datetime(
        df_time["Timestamp"], errors="coerce"
    )
    has_valid_time = df_time["Timestamp_parsed"].notna().sum() >= 3

    st.markdown("---")
    st.subheader("Hasil Analisis Data Realtime")

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
    section_header("Isi Topik & Kategori Channel")

    c_topik1, c_topik2 = st.columns(2)
    with c_topik1:
      st.markdown("##### Isi Kata Kunci per Topik LDA (Chat yang Baru Diekstrak)")
      st.caption(
          "Ini buat ngecek beneran isi Topik 1-5 dari chat yang barusan"
          " diekstrak. Dihitung LANGSUNG dari kolom 'Pesan Bersih"
          " (Sastrawi)' di atas."
      )
      kata_kunci_realtime = top_words_per_topic(
          df_real, "Pesan Bersih (Sastrawi)", "Topik LDA", n_kata=6
      )
      rows_kata_topik = []
      for _topik, _kata_list in kata_kunci_realtime.items():
        for _kata, _freq in _kata_list:
          rows_kata_topik.append(
              {"Topik LDA": _topik, "Kata": _kata, "Frekuensi": _freq}
          )
      if rows_kata_topik:
        df_kata_topik = pd.DataFrame(rows_kata_topik)
        fig_kata_topik = px.bar(
            df_kata_topik,
            x="Frekuensi",
            y="Kata",
            color="Topik LDA",
            orientation="h",
            facet_col="Topik LDA",
            facet_col_wrap=1,
            height=max(280, 95 * max(len(kata_kunci_realtime), 1)),
            color_discrete_sequence=COLOR_THEME,
        )
        fig_kata_topik.update_yaxes(matches=None, showticklabels=True)
        fig_kata_topik.for_each_annotation(
            lambda a: a.update(text=a.text.split("=")[-1])
        )
        fig_kata_topik.update_layout(showlegend=False)
        st.plotly_chart(style_fig(fig_kata_topik), use_container_width=True)
        charts_for_pdf["Isi Kata Kunci per Topik LDA"] = fig_kata_topik
      else:
        st.info("Belum ada kata bersih yang cukup untuk dihitung per topik.")

    with c_topik2:
      st.markdown(
          "##### Sebaran Sentimen Berdasarkan Kategori Channel (YouTube API)"
      )
      st.caption(
          "Kategori diambil otomatis dari YouTube Data API saat ekstraksi"
          " (kolom 'Kategori Channel (YouTube)')."
          " Kalau sebuah channel punya lebih dari satu kategori resmi,"
          " setiap kategorinya dihitung terpisah di grafik ini."
      )
      if (df_real["Kategori Channel (YouTube)"] == "Tidak Diketahui").all():
        st.info(
            "Kategori channel belum terdeteksi untuk hasil ekstraksi ini."
            " Kemungkinan penyebab: YOUTUBE_API_KEY belum diisi di"
            " Streamlit Secrets, atau channel-nya memang tidak punya"
            " topicCategories di YouTube."
        )
      else:
        df_kategori_explode = df_real.copy()
        df_kategori_explode["Kategori Channel (YouTube)"] = (
            df_kategori_explode["Kategori Channel (YouTube)"]
            .astype(str)
            .str.split(",")
        )
        df_kategori_explode = df_kategori_explode.explode(
            "Kategori Channel (YouTube)"
        )
        df_kategori_explode["Kategori Channel (YouTube)"] = (
            df_kategori_explode["Kategori Channel (YouTube)"].str.strip()
        )
        fig_kat_sent = px.histogram(
            df_kategori_explode,
            x="Kategori Channel (YouTube)",
            color="Prediksi Sentimen",
            barmode="group",
            color_discrete_map={"Positif": COLOR_POS, "Negatif": COLOR_NEG},
        )
        st.plotly_chart(style_fig(fig_kat_sent), use_container_width=True)
        charts_for_pdf["Sebaran Sentimen per Kategori Channel"] = fig_kat_sent

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

else:
  if df_benchmark.empty:
    st.error("File dataset benchmark (.xlsx) tidak ditemukan di repositori.")
    st.info(
        "Pastikan berkas `data_vtuber_labeled.xlsx` atau"
        " `hasil_akhir_analisis_skripsi.xlsx` sudah tersedia di repo."
    )
  else:
    st.sidebar.markdown("### Filter Panel Global")
    with st.sidebar.expander("Info kolom terdeteksi (debug)", expanded=False):
      st.caption(
          f"Kolom VTuber: `{col_vtuber}`\n\n"
          f"Kolom kategori channel: `{col_stream}`\n\n"
          f"Kolom sentimen: `{col_sentimen}`\n\n"
          f"Kolom teks mentah: `{col_text_raw}`\n\n"
          f"Kolom teks bersih (app): `{col_text_bersih_app}`\n\n"
          "Kalau kolom kategori channel di atas masih nunjuk ke kolom"
          " observasi lama (bukan 'Kategori Channel (YouTube)'), berarti"
          " nama kolom di file Excel belum cocok/berubah, cek lagi nama"
          " kolomnya di file dataset."
      )
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
          "Analisis Kategori Channel & Korelasi",
      ])

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

        kata_kunci_topik_aktual = (
            top_words_per_topic(df_filtered, col_text_bersih_app, col_topik)
            if col_text_bersih_app and col_text_bersih_app in df_filtered.columns
            else {}
        )
        with st.expander(
            "Kata Kunci AKTUAL Tiap Topik LDA (dihitung dari dataset yang"
            " sedang dimuat, klik untuk membuka)",
            expanded=False,
        ):
          if kata_kunci_topik_aktual:
            for _label, _kata_list in kata_kunci_topik_aktual.items():
              _teks_kata = (
                  ", ".join([k for k, _ in _kata_list])
                  if _kata_list
                  else "(belum ada data bersih untuk topik ini)"
              )
              st.markdown(f"* **{_label}**: *{_teks_kata}*")
          else:
            st.warning(
                "Kolom teks ('Pesan Bersih'/'clean_text'/'Chat Text') tidak"
                " ditemukan di dataset ini, jadi kata kunci aktual belum"
                " bisa dihitung. Pastikan file dataset punya salah satu"
                " kolom teks tersebut. Tidak ada lagi teks contoh/hardcode"
                " yang ditampilkan sebagai pengganti, untuk menghindari"
                " kata yang sudah tidak relevan dengan data sebenarnya."
            )

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
          st.caption(
              "Kolom kategori di bawah ini otomatis diambil dari kolom"
              " kategori channel resmi YouTube yang ada di file dataset"
              " (data_vtuber_labeled.xlsx), BUKAN kategori observasi"
              " pribadi lagi. Kalau untuk VTuber tertentu belum ada"
              " kecocokan di dataset, ditandai jelas sebagai fallback."
          )
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

          def _normalisasi_nama(teks):
            return re.sub(r"[^a-z0-9]", "", str(teks).lower())

          def _cari_kategori_resmi(nama_saluran, username):
            if (
                df_benchmark.empty
                or col_vtuber not in df_benchmark.columns
                or col_stream not in df_benchmark.columns
            ):
              return None
            target_nama = _normalisasi_nama(nama_saluran)
            target_user = _normalisasi_nama(username)
            for nilai_vtuber in df_benchmark[col_vtuber].dropna().unique():
              kandidat = _normalisasi_nama(nilai_vtuber)
              if not kandidat:
                continue
              cocok = (
                  kandidat in target_nama
                  or target_nama in kandidat
                  or kandidat in target_user
                  or target_user in kandidat
              )
              if cocok:
                subset = df_benchmark.loc[
                    df_benchmark[col_vtuber] == nilai_vtuber, col_stream
                ]
                modus = subset.mode()
                if not modus.empty:
                  return modus.iloc[0]
            return None

          kategori_resmi_list = []
          for _row in data_20_vtuber:
            _hasil = _cari_kategori_resmi(
                _row["Nama Saluran"], _row["Username"]
            )
            if _hasil:
              kategori_resmi_list.append(_hasil)
            else:
              kategori_resmi_list.append(
                  f'{_row["Kategori Konten"]} (belum ada di dataset, fallback observasi)'
              )
          df_table_20["Kategori Channel (Resmi YouTube)"] = kategori_resmi_list
          df_table_20 = df_table_20.drop(columns=["Kategori Konten"])

          st.dataframe(df_table_20, use_container_width=True, height=350)
          if not df_benchmark.empty and col_stream not in df_benchmark.columns:
            st.warning(
                "File dataset yang sedang dimuat belum punya kolom kategori"
                " channel resmi YouTube, jadi seluruh baris di atas masih"
                " memakai fallback observasi. Timpa"
                " `data_vtuber_labeled.xlsx` di GitHub dengan versi yang"
                " sudah punya kolom 'Kategori Channel (YouTube)' hasil"
                " fetch_kategori_youtube.py."
            )

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
            st.markdown(f"##### Kategori Channel ({selected_single_vt})")
            if col_stream in df_single.columns:
              df_single_exploded = explode_kategori_channel(
                  df_single, col_stream
              )
              daftar_kategori_vt = (
                  df_single_exploded[col_stream]
                  .dropna()
                  .astype(str)
                  .unique()
                  .tolist()
              )
              warna_kategori_vt = get_kategori_color_map(
                  df_single_exploded, col_stream
              )
              if daftar_kategori_vt:
                chip_html = "".join(
                    f'<span class="kategori-chip" style="border-color:'
                    f'{warna_kategori_vt.get(kat, "#6366F1")}; color:'
                    f'{warna_kategori_vt.get(kat, "#6366F1")};">{kat}</span>'
                    for kat in daftar_kategori_vt
                )
                st.markdown(
                    f'<div class="kategori-chip-wrap">{chip_html}</div>',
                    unsafe_allow_html=True,
                )
                st.caption(
                    "Channel ini berlabel kategori resmi YouTube di atas."
                    " Bukan grafik proporsi/dominasi, karena YouTube tidak"
                    " memberi bobot antar kategori pada satu channel."
                )
              else:
                st.info("Kategori channel belum terdeteksi untuk VTuber ini.")

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
            st.markdown("##### Sebaran Kategori Channel per VTuber")
            if col_stream in df_filtered.columns:
              df_vt_cat_exploded = explode_kategori_channel(
                  df_filtered, col_stream
              )
              fig_vt_cat = px.histogram(
                  df_vt_cat_exploded,
                  x=col_vtuber,
                  color=col_stream,
                  barmode="stack",
                  color_discrete_map=get_kategori_color_map(
                      df_vt_cat_exploded, col_stream
                  ),
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

      with tab3:
        st.markdown("### Analisis Komparatif & Korelasi Kategori Channel")
        st.caption(
            "Membedah keterkaitan antara kategori channel dengan pola"
            " respon sentimen dan topik pembicaraan audiens. Kategori"
            " channel yang gabungan (mis. \"Role-playing video game,"
            " Video game culture\") dipecah jadi kategori individual"
            " supaya channel dengan kategori yang sama tetap satu"
            " kelompok di grafik."
        )

        df_kategori_exploded = explode_kategori_channel(df_filtered, col_stream)

        if (
            col_stream in df_filtered.columns
            and df_kategori_exploded is not None
            and not df_kategori_exploded.empty
        ):
          col_k1, col_k2 = st.columns(2)
          with col_k1:
            st.markdown("##### 1. Distribusi Sentimen per Kategori Channel")
            if col_sentimen in df_kategori_exploded.columns:
              fig_cat_sent = px.histogram(
                  df_kategori_exploded,
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
            st.markdown("##### 2. Distribusi Topik LDA per Kategori Channel")
            fig_cat_top = px.histogram(
                df_kategori_exploded,
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
                df_kategori_exploded.groupby([col_stream, col_sentimen])
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
                df_kategori_exploded.groupby([col_stream, col_topik])
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
              "##### Tabel Ringkasan Korelasi Kategori Channel & Sentimen"
          )
          cat_summary = (
              df_kategori_exploded.groupby(col_stream)
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
