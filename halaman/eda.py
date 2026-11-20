import pandas as pd
import streamlit as st
from utils import load_data
import charts

df = load_data()
st.title("Eksplorasi Data (EDA)")
st.caption("Pola apa yang membedakan pelanggan yang churn dan yang bertahan?")

# filter global di sidebar (2.7 layout)
with st.sidebar:
    st.header("Filter EDA")
    kontrak = st.multiselect("Jenis kontrak", df["Contract"].unique().tolist(), default=df["Contract"].unique().tolist())
    tmin, tmax = st.slider("Rentang tenure (bulan)", 0, 72, (0, 72))
dff = df[df["Contract"].isin(kontrak) & df["Tenure Months"].between(tmin, tmax)]
if dff.empty:
    st.warning("Tidak ada data untuk filter ini. Ubah filter di sidebar.")
    st.stop()

c1, c2, c3 = st.columns(3)
base = df["Churn Value"].mean() * 100
rate = dff["Churn Value"].mean() * 100
c1.metric("Pelanggan terfilter", f"{len(dff):,}")
c2.metric("Churn rate terfilter", f"{rate:.1f}%", f"{rate - base:+.1f} poin vs keseluruhan", delta_color="inverse")
c3.metric("Rata-rata tagihan", f"${dff['Monthly Charges'].mean():.2f}")

t1, t2, t3, t4, t5 = st.tabs(["Distribusi Churn", "Fitur Kategorikal", "Fitur Numerik", "Korelasi", "Sebaran Lokasi"])
with t1:
    charts.tampil(charts.distribusi_kelas(dff))
    st.info("Kelas tidak seimbang: model yang menebak semua pelanggan 'tidak churn' sudah akurat ~73,5% tetapi "
            "recall churn-nya 0. Karena itu yang dipantau adalah Recall, F1, dan ROC-AUC.")
with t2:
    kat = ["Contract", "Internet Service", "Payment Method", "Tech Support", "Online Security", "Paperless Billing",
           "Senior Citizen", "Dependents", "Partner", "Gender", "Phone Service", "Multiple Lines",
           "Online Backup", "Device Protection", "Streaming TV", "Streaming Movies"]
    pilih = st.selectbox("Pilih fitur", kat)
    charts.tampil(charts.rate_kategori(dff, pilih))
    with st.expander("Lihat tabel angka"):
        tb = dff.groupby(pilih)["Churn Value"].agg(Jumlah="count", Churn="sum", Churn_rate="mean")
        tb["Churn_rate"] = (tb["Churn_rate"] * 100).round(1)
        st.dataframe(tb.sort_values("Churn_rate", ascending=False))
with t3:
    pilih = st.radio("Pilih fitur numerik", ["Tenure Months", "Monthly Charges", "Total Charges"], horizontal=True)
    charts.tampil(charts.hist_numerik(dff, pilih))
    st.caption("Pelanggan churn cenderung punya tenure pendek dan tagihan bulanan lebih tinggi.")
with t4:
    kiri, kanan = st.columns([1, 1])
    with kiri:
        charts.tampil(charts.korelasi(dff))
    with kanan:
        st.markdown("**Cara membaca:** nilai mendekati 1/-1 berarti hubungan kuat.\n\n"
                    "- `Tenure Months` berkorelasi negatif dengan churn (makin lama berlangganan, makin kecil kemungkinan churn).\n"
                    "- `Total Charges` kira-kira = Tenure x Monthly Charges, sehingga ketiganya saling berkorelasi (multikolinearitas).")
with t5:
    peta = dff[dff["Churn Value"] == 1].rename(columns={"Latitude": "latitude", "Longitude": "longitude"})
    st.write(f"Sebaran {len(peta):,} pelanggan churn di California (maks. 1.000 titik)")
    st.map(peta[["latitude", "longitude"]].sample(min(1000, len(peta)), random_state=1), color="#4F86E8")

with st.expander("Ringkasan temuan utama"):
    g = lambda col, val: df.loc[df[col] == val, "Churn Value"].mean() * 100
    st.markdown(
        f"- Kontrak **Month-to-month**: churn rate {g('Contract', 'Month-to-month'):.1f}% "
        f"vs Two year {g('Contract', 'Two year'):.1f}%.\n"
        f"- Layanan **Fiber optic**: {g('Internet Service', 'Fiber optic'):.1f}% vs DSL {g('Internet Service', 'DSL'):.1f}%.\n"
        f"- Pembayaran **Electronic check**: {g('Payment Method', 'Electronic check'):.1f}%.\n"
        f"- Tanpa **Tech Support**: {g('Tech Support', 'No'):.1f}% vs dengan Tech Support {g('Tech Support', 'Yes'):.1f}%.")
