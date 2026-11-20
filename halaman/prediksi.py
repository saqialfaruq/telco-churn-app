import pandas as pd
import streamlit as st
from utils import MODEL_NAMES, load_data, load_hasil, muat_model_aman, rekomendasi, daftar_fitur

df, hasil = load_data(), load_hasil()
FITUR = daftar_fitur(hasil)
NUM = hasil["features"]["num"]
opsi = {c: sorted(df[c].dropna().unique().tolist()) for c in FITUR if c not in NUM}

st.title("Prediksi Churn Satu Pelanggan")
st.caption("Pilih model, isi profil pelanggan, lalu lihat probabilitas churn dan saran tindakan retensi.")

# profil contoh: mengganti pilihan akan membuat ulang semua widget dengan nilai awal baru (lewat key)
CONTOH = {
    "Isi manual": {},
    "Contoh: risiko tinggi": {"Contract": "Month-to-month", "Internet Service": "Fiber optic", "Payment Method": "Electronic check",
                              "Tech Support": "No", "Online Security": "No", "Tenure Months": 2, "Monthly Charges": 85.0},
    "Contoh: risiko rendah": {"Contract": "Two year", "Internet Service": "DSL", "Payment Method": "Credit card (automatic)",
                              "Tech Support": "Yes", "Online Security": "Yes", "Tenure Months": 60, "Monthly Charges": 55.0},
}

a, b = st.columns(2)
nama_model = a.selectbox("Model yang dipakai", MODEL_NAMES, index=MODEL_NAMES.index(hasil["best_model"]),
                         help="Model terbaik berdasarkan F1 dipilih secara bawaan.")
contoh = b.selectbox("Profil awal", list(CONTOH))
d = CONTOH[contoh]
k = contoh.replace(" ", "_")


def sel(label, kolom, kol):
    nilai = d.get(kolom)
    idx = opsi[kolom].index(nilai) if nilai in opsi[kolom] else 0
    return kol.selectbox(label, opsi[kolom], index=idx, key=f"{k}_{kolom}")


with st.form("form_prediksi"):
    c1, c2, c3 = st.columns(3)
    gender = sel("Gender", "Gender", c1)
    senior = sel("Senior Citizen", "Senior Citizen", c1)
    partner = sel("Partner", "Partner", c1)
    dependents = sel("Dependents", "Dependents", c1)
    tenure = c1.slider("Tenure (bulan)", 0, 72, int(d.get("Tenure Months", 12)), key=f"{k}_tenure")
    phone = sel("Phone Service", "Phone Service", c1)
    multi = sel("Multiple Lines", "Multiple Lines", c2)
    inet = sel("Internet Service", "Internet Service", c2)
    osec = sel("Online Security", "Online Security", c2)
    obak = sel("Online Backup", "Online Backup", c2)
    dprot = sel("Device Protection", "Device Protection", c2)
    tsup = sel("Tech Support", "Tech Support", c2)
    stv = sel("Streaming TV", "Streaming TV", c3)
    smov = sel("Streaming Movies", "Streaming Movies", c3)
    kontrak = sel("Contract", "Contract", c3)
    paperless = sel("Paperless Billing", "Paperless Billing", c3)
    bayar = sel("Payment Method", "Payment Method", c3)
    monthly = c3.number_input("Monthly Charges ($)", 0.0, 200.0, float(d.get("Monthly Charges", 70.0)), 0.5, key=f"{k}_monthly")
    ambang = st.slider("Threshold churn", 0.1, 0.9, 0.5, 0.05,
                       help="Turunkan threshold agar lebih banyak calon churn terjaring (Recall naik, Precision turun).")
    kirim = st.form_submit_button("Prediksi", type="primary")

if kirim:
    profil = {"Gender": gender, "Senior Citizen": senior, "Partner": partner, "Dependents": dependents,
              "Tenure Months": tenure, "Phone Service": phone, "Multiple Lines": multi, "Internet Service": inet,
              "Online Security": osec, "Online Backup": obak, "Device Protection": dprot, "Tech Support": tsup,
              "Streaming TV": stv, "Streaming Movies": smov, "Contract": kontrak, "Paperless Billing": paperless,
              "Payment Method": bayar, "Monthly Charges": monthly, "Total Charges": round(tenure * monthly, 2)}
    baris = pd.DataFrame([profil])[FITUR]
    model = muat_model_aman(nama_model, hasil)
    proba = float(model.predict_proba(baris)[0, 1])

    st.divider()
    st.subheader(f"Hasil prediksi - {nama_model}")
    m1, m2, m3 = st.columns(3)
    m1.metric("Probabilitas churn", f"{proba * 100:.1f}%")
    m2.metric("Threshold", f"{ambang:.2f}")
    m3.metric("Keputusan", "CHURN" if proba >= ambang else "BERTAHAN")
    st.progress(min(max(proba, 0.0), 1.0), text="Tingkat risiko churn")
    if proba >= ambang:
        st.error("Pelanggan ini diprediksi **churn**. Prioritaskan untuk program retensi.")
    else:
        st.success("Pelanggan ini diprediksi **bertahan**.")

    kiri, kanan = st.columns(2)
    with kiri:
        st.markdown("**Saran tindakan**")
        for s in rekomendasi(profil, proba):
            st.write("- " + s)
    with kanan:
        with st.expander("Bandingkan dengan semua model", expanded=True):
            rows = []
            for n in MODEL_NAMES:
                pr = float(muat_model_aman(n, hasil).predict_proba(baris)[0, 1])
                rows.append({"Model": n, "Probabilitas churn (%)": round(pr * 100, 1), "Prediksi": "Churn" if pr >= ambang else "Bertahan"})
            st.dataframe(pd.DataFrame(rows).set_index("Model"))
        with st.expander("Data yang dikirim ke model"):
            st.dataframe(baris.T.rename(columns={0: "Nilai"}).astype(str))
    st.caption("Total Charges diperkirakan = Tenure x Monthly Charges. Hasil prediksi adalah estimasi statistik, bukan kepastian.")
