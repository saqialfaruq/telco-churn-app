import pandas as pd
import streamlit as st
from utils import MODEL_NAMES, load_data, load_hasil, muat_model_aman, daftar_fitur
import charts

hasil = load_hasil()
FITUR = daftar_fitur(hasil)
st.title("Prediksi Massal (CSV)")
st.caption("Unggah file CSV berisi banyak pelanggan, lalu unduh hasil prediksinya.")

a, b = st.columns(2)
nama_model = a.selectbox("Model", MODEL_NAMES, index=MODEL_NAMES.index(hasil["best_model"]))
ambang = b.slider("Threshold churn", 0.1, 0.9, 0.5, 0.05)

with st.expander("Kolom yang dibutuhkan"):
    st.write(", ".join(f"`{c}`" for c in FITUR))
    st.caption("Kolom lain (mis. CustomerID) boleh ada dan akan dibawa ke hasil. Format sama dengan Telco_customer_churn.csv.")

file = st.file_uploader("Unggah file CSV", type=["csv"])
pakai_contoh = st.checkbox("Tidak punya file? Pakai 300 baris contoh dari dataset")

data = None
if file is not None:
    data = pd.read_csv(file)
elif pakai_contoh:
    data = load_data().sample(300, random_state=7).reset_index(drop=True)

if data is None:
    st.info("Unggah file CSV atau centang contoh data untuk memulai.")
    st.stop()

hilang = [c for c in FITUR if c not in data.columns]
if hilang:
    st.error("Kolom berikut tidak ada di file: " + ", ".join(hilang))
    st.stop()

data["Total Charges"] = pd.to_numeric(data["Total Charges"], errors="coerce")
model = muat_model_aman(nama_model, hasil)
proba = model.predict_proba(data[FITUR])[:, 1]
label = (proba >= ambang).astype(int)
out = data.copy()
out["Probabilitas Churn"] = proba.round(4)
out["Prediksi"] = pd.Series(label, index=out.index).map({1: "Churn", 0: "Bertahan"})

c1, c2, c3 = st.columns(3)
c1.metric("Jumlah pelanggan", f"{len(out):,}")
c2.metric("Diprediksi churn", f"{int(label.sum()):,}")
c3.metric("Persentase churn", f"{label.mean() * 100:.1f}%")

if "Churn Value" in data.columns:
    from sklearn.metrics import accuracy_score, f1_score, recall_score
    yt = data["Churn Value"]
    st.subheader("Kecocokan dengan label asli (kolom Churn Value ada di file)")
    d1, d2, d3 = st.columns(3)
    d1.metric("Accuracy", f"{accuracy_score(yt, label):.3f}")
    d2.metric("Recall churn", f"{recall_score(yt, label):.3f}")
    d3.metric("F1", f"{f1_score(yt, label):.3f}")
    st.caption("Jika file berisi data yang dipakai melatih model, angka ini lebih optimistis daripada hasil pada data uji.")

t1, t2 = st.tabs(["Tabel hasil", "Sebaran probabilitas"])
with t1:
    awal = [c for c in ["CustomerID"] if c in out.columns]
    st.dataframe(out[awal + ["Probabilitas Churn", "Prediksi"] + FITUR[:8]].sort_values("Probabilitas Churn", ascending=False).head(200))
    st.download_button("Download hasil prediksi (CSV)", out.to_csv(index=False).encode("utf-8"),
                       file_name="hasil_prediksi_churn.csv", mime="text/csv")
with t2:
    charts.tampil(charts.proba_hist(proba, ambang))
