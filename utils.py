"""Fungsi bersama untuk semua halaman: pemuatan data/model, tema, dan data referensi."""
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"

# ----------------------------------------------------------------------------- identitas
PEMBUAT = [
    ("Muhammad Ihsan Fawwazi", "241011403237"),
    ("Radja Putra Andhika", "241011403338"),
    ("Sapta Aqila Adnanta", "241011401850"),
]
DOSEN = "Farizi Ilham, M.Kom."
KAMPUS = "Program Studi Informatika, Fakultas Ilmu Komputer, Universitas Pamulang"

MODEL_NAMES = ["Logistic Regression", "KNN", "Decision Tree", "Random Forest", "Gradient Boosting"]

KAMUS = {
    "CustomerID": ("Identitas", "ID unik pelanggan"),
    "Count": ("Konstan", "Selalu bernilai 1"),
    "Country": ("Geografis", "Negara (selalu United States)"),
    "State": ("Geografis", "Negara bagian (selalu California)"),
    "City": ("Geografis", "Kota tempat tinggal pelanggan"),
    "Zip Code": ("Geografis", "Kode pos"),
    "Lat Long": ("Geografis", "Koordinat gabungan (teks)"),
    "Latitude": ("Geografis", "Garis lintang"),
    "Longitude": ("Geografis", "Garis bujur"),
    "Gender": ("Demografi", "Jenis kelamin pelanggan"),
    "Senior Citizen": ("Demografi", "Apakah pelanggan lansia (Yes/No)"),
    "Partner": ("Demografi", "Apakah memiliki pasangan"),
    "Dependents": ("Demografi", "Apakah memiliki tanggungan"),
    "Tenure Months": ("Layanan", "Lama berlangganan (bulan)"),
    "Phone Service": ("Layanan", "Berlangganan layanan telepon"),
    "Multiple Lines": ("Layanan", "Memiliki lebih dari satu jalur telepon"),
    "Internet Service": ("Layanan", "Jenis layanan internet (DSL / Fiber optic / No)"),
    "Online Security": ("Layanan", "Layanan keamanan online"),
    "Online Backup": ("Layanan", "Layanan cadangan online"),
    "Device Protection": ("Layanan", "Perlindungan perangkat"),
    "Tech Support": ("Layanan", "Dukungan teknis"),
    "Streaming TV": ("Layanan", "Layanan streaming TV"),
    "Streaming Movies": ("Layanan", "Layanan streaming film"),
    "Contract": ("Akun", "Jenis kontrak (Month-to-month / One year / Two year)"),
    "Paperless Billing": ("Akun", "Tagihan tanpa kertas"),
    "Payment Method": ("Akun", "Metode pembayaran"),
    "Monthly Charges": ("Akun", "Tagihan per bulan (USD)"),
    "Total Charges": ("Akun", "Total tagihan sepanjang berlangganan (USD)"),
    "Churn Label": ("Target", "Status churn dalam teks (Yes/No), duplikat target"),
    "Churn Value": ("Target", "TARGET: 1 = churn, 0 = bertahan"),
    "Churn Score": ("Bocoran", "Skor churn dihitung setelah status diketahui"),
    "CLTV": ("Bocoran", "Customer Lifetime Value, dihitung dengan memakai info churn"),
    "Churn Reason": ("Bocoran", "Alasan churn, hanya terisi untuk pelanggan yang sudah churn"),
}

BIRU, BIRU_MUDA, BIRU_TUA, KORAL, ABU = "#4F86E8", "#A9C7F7", "#2F5DA8", "#E58A8A", "#8A9BB0"


# ----------------------------------------------------------------------------- tema
CSS = """
<style>
.block-container {padding-top: 2.2rem; max-width: 1200px;}
h1, h2, h3 {color: #2F5DA8;}
div[data-testid="stMetric"] {background: #FFFFFF; border: 1px solid #D5E3F8; border-radius: 14px;
                             padding: 12px 16px; box-shadow: 0 1px 4px rgba(79,134,232,.10);}
div[data-testid="stVerticalBlockBorderWrapper"] {border-radius: 14px;}
.stTabs [data-baseweb="tab-list"] {gap: 6px;}
.stTabs [data-baseweb="tab"] {background: #E4EEFC; border-radius: 10px 10px 0 0; padding: 8px 16px;}
.stTabs [aria-selected="true"] {background: #FFFFFF;}
</style>
"""


def pasang_tema():
    st.markdown(CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------------- data & model
@st.cache_data
def load_data() -> pd.DataFrame:
    d = pd.read_csv(ROOT / "data" / "Telco_customer_churn.csv")
    d["Total Charges"] = pd.to_numeric(d["Total Charges"], errors="coerce")
    return d


@st.cache_data
def load_raw() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data" / "Telco_customer_churn.csv")


@st.cache_data
def load_hasil() -> dict:
    f = ART / "hasil.json"
    if not f.exists():
        st.error("File artifacts/hasil.json belum ada. Jalankan dulu:  python train_all.py")
        st.stop()
    return json.loads(f.read_text(encoding="utf-8"))


@st.cache_resource
def load_model(kode: str):
    return joblib.load(ART / "models" / f"{kode}.joblib")


def muat_model_aman(nama: str, hasil: dict):
    """Memuat model; jika gagal (versi scikit-learn berbeda), tampilkan petunjuk dan hentikan halaman."""
    try:
        return load_model(hasil["models"][nama]["kode"])
    except Exception as e:
        st.error("Model gagal dimuat (kemungkinan versi scikit-learn berbeda). Jalankan:  python train_all.py")
        st.caption(f"Detail: {e}")
        st.stop()


def daftar_fitur(hasil: dict) -> list:
    return hasil["features"]["num"] + hasil["features"]["cat"]


def rekomendasi(profil: dict, proba: float) -> list:
    """Saran tindakan retensi berdasarkan profil pelanggan dan probabilitas churn."""
    saran = []
    if profil["Contract"] == "Month-to-month":
        saran.append("Tawarkan promo upgrade ke kontrak 1 atau 2 tahun (kontrak bulanan adalah faktor churn terbesar).")
    if profil["Internet Service"] == "Fiber optic" and profil["Tech Support"] == "No":
        saran.append("Berikan paket Tech Support gratis/diskon; pengguna Fiber tanpa dukungan teknis cenderung churn.")
    if profil["Online Security"] == "No" and profil["Internet Service"] != "No":
        saran.append("Tawarkan layanan Online Security sebagai bundling.")
    if profil["Payment Method"] == "Electronic check":
        saran.append("Dorong pindah ke pembayaran otomatis (kartu/transfer) dengan insentif kecil.")
    if profil["Tenure Months"] < 12:
        saran.append("Masukkan ke program onboarding 12 bulan pertama (pelanggan baru paling rentan).")
    if not saran:
        saran.append("Pertahankan layanan dan lakukan survei kepuasan berkala.")
    if proba >= 0.7:
        saran.insert(0, "Prioritas tinggi: hubungi langsung (telepon/CS) dalam minggu ini.")
    return saran
