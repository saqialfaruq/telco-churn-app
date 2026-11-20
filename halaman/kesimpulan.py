import pandas as pd
import streamlit as st
from utils import MODEL_NAMES, load_data, load_hasil

df, hasil = load_data(), load_hasil()
M, P = hasil["models"], hasil["pca"]
best = hasil["best_model"]
mb = M[best]["metrics"]
top = list(M["Random Forest"]["importance"])[:5]

st.title("Kesimpulan & Saran")

st.subheader("Kesimpulan")
st.markdown(
    f"1. **Model terbaik** berdasarkan F1 adalah **{best}** dengan F1 {mb['f1']:.3f}, Recall {mb['recall']:.3f}, "
    f"Precision {mb['precision']:.3f}, dan ROC-AUC {mb['roc_auc']:.3f}.\n"
    f"2. Akurasi ({mb['accuracy']:.3f}) tampak wajar tetapi tidak cukup karena kelas tidak seimbang; **Recall kelas churn** "
    f"menunjukkan model menangkap {mb['recall'] * 100:.0f}% pelanggan yang benar-benar churn.\n"
    f"3. Setelah dituning, Random Forest naik F1-nya dari {hasil['baseline_tuning']['Random Forest']['f1']:.3f} menjadi {M['Random Forest']['metrics']['f1']:.3f}.\n"
    f"4. **PCA** memadatkan {P['n_features']} fitur menjadi {P['k90']}-{P['k95']} komponen (90-95% varians) dengan penurunan AUC yang kecil, "
    f"namun tidak meningkatkan akurasi dan menyulitkan interpretasi. Karena itu model prediksi memakai fitur asli.\n"
    f"5. Fitur paling berpengaruh (Random Forest): {', '.join(f'`{t}`' for t in top)}.\n"
    "6. Menurunkan *threshold* meningkatkan Recall tetapi menurunkan Precision, sehingga pilihannya bergantung pada biaya kampanye retensi.")

st.subheader("Perbandingan ringkas lima model")
tb = pd.DataFrame({n: M[n]["metrics"] for n in MODEL_NAMES}).T[["recall", "f1", "roc_auc"]]
tb.columns = ["Recall", "F1-Score", "ROC-AUC"]
st.dataframe(tb.sort_values("F1-Score", ascending=False).style.format("{:.3f}"))

st.subheader("Saran bisnis")
g = lambda c, v: df.loc[df[c] == v, "Churn Value"].mean() * 100
st.markdown(
    f"- **Kontrak bulanan** (churn {g('Contract', 'Month-to-month'):.0f}%) dan **tenure rendah** adalah target utama retensi: tawarkan promo upgrade ke kontrak 1-2 tahun.\n"
    f"- Pengguna **Fiber optic** ({g('Internet Service', 'Fiber optic'):.0f}%) tanpa **Tech Support**: berikan dukungan teknis atau bundling keamanan online.\n"
    f"- Pembayaran **Electronic check** ({g('Payment Method', 'Electronic check'):.0f}%): dorong pembayaran otomatis dengan insentif.\n"
    "- Gunakan threshold lebih rendah (mis. 0,4) bila biaya kehilangan pelanggan jauh lebih besar daripada biaya promo.")

st.subheader("Keterbatasan")
st.markdown(
    "- Data hanya dari satu wilayah (California) dan satu periode, sehingga belum tentu berlaku untuk wilayah lain.\n"
    "- Model memprediksi **kemungkinan**, bukan sebab-akibat; faktor seperti harga pesaing tidak ada di data.\n"
    "- Hasil dapat sedikit berbeda jika model dilatih ulang dengan versi library atau random state yang berbeda.")
