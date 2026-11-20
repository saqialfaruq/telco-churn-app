import pandas as pd
import streamlit as st
from utils import KAMUS, load_data, load_raw
import charts

df, raw = load_data(), load_raw()
st.title("Tentang Dataset")
st.caption("Sumber: Kaggle - IBM Telco Customer Churn (format CSV)")

tab1, tab2, tab3, tab4 = st.tabs(["Ringkasan", "Kamus Kolom", "Lihat Data", "Kualitas Data"])

with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Baris", f"{df.shape[0]:,}")
    c2.metric("Kolom", df.shape[1])
    c3.metric("Pelanggan churn", f"{int(df['Churn Value'].sum()):,}")
    c4.metric("Rasio kelas", f"{(df['Churn Value'] == 0).sum() / (df['Churn Value'] == 1).sum():.2f} : 1")
    st.markdown(
        "- **Jenis masalah:** klasifikasi biner (supervised learning)\n"
        "- **Target:** `Churn Value` (1 = berhenti berlangganan, 0 = bertahan)\n"
        "- **Tujuan:** memprediksi pelanggan yang berpotensi berhenti\n"
        "- **Catatan:** kelas tidak seimbang, jadi akurasi saja tidak cukup")
    charts.tampil(charts.distribusi_kelas(df))

with tab2:
    kamus = pd.DataFrame([(k, v[0], v[1], str(raw[k].dtype)) for k, v in KAMUS.items()],
                         columns=["Kolom", "Kelompok", "Keterangan", "Tipe data"])
    pilih = st.multiselect("Filter kelompok", sorted(kamus["Kelompok"].unique()), default=sorted(kamus["Kelompok"].unique()))
    st.dataframe(kamus[kamus["Kelompok"].isin(pilih)], hide_index=True)

with tab3:
    kiri, kanan = st.columns(2)
    kontrak = kiri.multiselect("Jenis kontrak", df["Contract"].unique().tolist(), default=df["Contract"].unique().tolist())
    status = kanan.radio("Status", ["Semua", "Churn", "Tidak churn"], horizontal=True)
    sub = df[df["Contract"].isin(kontrak)]
    if status == "Churn":
        sub = sub[sub["Churn Value"] == 1]
    elif status == "Tidak churn":
        sub = sub[sub["Churn Value"] == 0]
    st.write(f"Menampilkan {len(sub):,} baris")
    st.dataframe(sub.head(500))
    st.download_button("Download data terfilter (CSV)", sub.to_csv(index=False).encode("utf-8"),
                       file_name="telco_terfilter.csv", mime="text/csv")

with tab4:
    kosong_tc = int((raw["Total Charges"].astype(str).str.strip() == "").sum())
    st.subheader("Data hilang dan tipe data")
    c1, c2 = st.columns(2)
    c1.metric("Total Charges kosong", kosong_tc, help="Semuanya pelanggan baru (tenure = 0); terbaca sebagai teks.")
    c2.metric("Churn Reason kosong", int(raw["Churn Reason"].isna().sum()), help="Hanya terisi untuk pelanggan churn.")
    st.subheader("Kolom yang tidak dipakai sebagai fitur")
    buang = pd.DataFrame([(k, v[0], v[1]) for k, v in KAMUS.items() if v[0] in ("Identitas", "Konstan", "Geografis", "Bocoran", "Target")],
                         columns=["Kolom", "Alasan dibuang", "Keterangan"])
    st.dataframe(buang, hide_index=True)
    st.warning("Kolom **Churn Score, CLTV, Churn Reason** adalah bocoran (data leakage): nilainya baru diketahui "
               "setelah status churn ada, sehingga tidak tersedia saat prediksi nyata.")
