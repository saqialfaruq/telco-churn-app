"""
Website Prediksi Churn Pelanggan Telco (Streamlit) - Tugas Machine Learning Kelompok 3.
Jalankan:  streamlit run app.py      lalu buka  http://localhost:8501
"""
import streamlit as st

st.set_page_config(page_title="Prediksi Churn Telco", page_icon=":telephone_receiver:", layout="wide")

from utils import pasang_tema  # noqa: E402  (harus setelah set_page_config)

pasang_tema()

halaman = {
    "Pengantar": [
        st.Page("halaman/beranda.py", title="Beranda", icon=":material/home:", default=True),
        st.Page("halaman/dataset.py", title="Tentang Dataset", icon=":material/database:"),
        st.Page("halaman/eda.py", title="Eksplorasi Data (EDA)", icon=":material/bar_chart:"),
    ],
    "Pengolahan & Model": [
        st.Page("halaman/preprocessing.py", title="Preprocessing", icon=":material/tune:"),
        st.Page("halaman/pca.py", title="Reduksi Dimensi (PCA)", icon=":material/scatter_plot:"),
        st.Page("halaman/modelling.py", title="Modelling & Evaluasi", icon=":material/model_training:"),
    ],
    "Penerapan": [
        st.Page("halaman/prediksi.py", title="Prediksi Satu Pelanggan", icon=":material/person_search:"),
        st.Page("halaman/massal.py", title="Prediksi Massal (CSV)", icon=":material/upload_file:"),
        st.Page("halaman/kesimpulan.py", title="Kesimpulan & Saran", icon=":material/task_alt:"),
    ],
}
nav = st.navigation(halaman)
with st.sidebar:
    st.caption("Tugas Machine Learning - Kelompok 3  \nInformatika, Universitas Pamulang")
nav.run()
