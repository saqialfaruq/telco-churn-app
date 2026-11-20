import streamlit as st
from utils import DOSEN, KAMPUS, PEMBUAT, load_data, load_hasil

df, hasil = load_data(), load_hasil()
best = hasil["best_model"]
mb = hasil["models"][best]["metrics"]

st.title("Prediksi Churn Pelanggan Telco")
st.caption("Implementasi Machine Learning berbasis web dengan Streamlit")

st.markdown(
    "Website ini memaparkan seluruh tahapan proyek **prediksi pelanggan yang akan berhenti berlangganan (churn)** "
    "pada dataset *Telco Customer Churn*: mulai dari pengenalan data, eksplorasi, preprocessing, reduksi dimensi (PCA), "
    "perbandingan lima algoritma klasifikasi, hingga **prediksi langsung** yang bisa dicoba pengguna."
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Jumlah pelanggan", f"{len(df):,}")
c2.metric("Churn rate", f"{df['Churn Value'].mean() * 100:.1f}%")
c3.metric("Model terbaik (F1)", best)
c4.metric("ROC-AUC model terbaik", f"{mb['roc_auc']:.3f}")

st.divider()
st.subheader("Alur pengerjaan")
a, b, c, d = st.columns(4)
for kol, (jd, isi) in zip([a, b, c, d], [
    ("1. Dataset", "Karakteristik data: 7.043 baris, 33 kolom, klasifikasi biner."),
    ("2. EDA & Preprocessing", "Menangani data hilang, data leakage, ketidakseimbangan kelas, dan encoding."),
    ("3. Reduksi PCA", "46 fitur dipadatkan menjadi 15-18 komponen tanpa penurunan kinerja berarti."),
    ("4. Modelling", "Lima algoritma dibandingkan, di-tuning, lalu dipakai untuk prediksi."),
]):
    with kol.container(border=True):
        st.markdown(f"**{jd}**")
        st.write(isi)

st.divider()
st.subheader("Anggota Kelompok 3")
cols = st.columns(3)
for kol, (nama, nim) in zip(cols, PEMBUAT):
    with kol.container(border=True):
        st.markdown(f"**{nama}**")
        st.caption(f"NIM {nim}")
st.write(f"Dosen pengampu: **{DOSEN}**")
st.caption(KAMPUS)

with st.expander("Mengapa memprediksi churn penting?"):
    st.markdown(
        "- Biaya mempertahankan pelanggan jauh lebih rendah daripada mencari pelanggan baru.\n"
        "- Dengan prediksi dini, tim retensi dapat menargetkan promo hanya pada pelanggan berisiko.\n"
        "- Kelas churn hanya ~26,5% dari data, sehingga model dinilai dengan **Recall, F1, dan ROC-AUC**, bukan akurasi saja."
    )
