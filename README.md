# Website Prediksi Churn Pelanggan Telco (Streamlit)
Tugas Machine Learning - Kelompok 3, Informatika, Universitas Pamulang. Dosen: Farizi Ilham, M.Kom.

Website multi-halaman yang menerapkan dataset Telco Customer Churn beserta hasil preprocessing, PCA, modelling, dan evaluasi dari notebook kelompok, memakai elemen Streamlit dari modul Bab 1-2.

## Menjalankan di localhost

Buka terminal PowerShell, lalu jalankan perintah berikut secara berurutan:

cd "D:\BACKUP_SEBELUM_INSTALL_ULANG\Dokumen-Dokumen\Dokumen Kuliah\Dokumen Kuliah\semester 5\ML\telco churn web\telco_churn_web"
python -m pip install -r requirements.txt
python -m streamlit run app.py

Model sudah disertakan di `artifacts/`. Jika muncul pesan "Model gagal dimuat" (karena perbedaan versi scikit-learn), latih ulang model terlebih dahulu (sekitar 5-8 menit):
python train_all.py

## Halaman
| Halaman | Isi |
|---|---|
| Beranda | Ringkasan proyek, alur kerja, anggota kelompok |
| Tentang Dataset | Ringkasan, kamus kolom, lihat/filter/unduh data, kualitas data |
| Eksplorasi Data (EDA) | Distribusi churn, fitur kategorikal/numerik, korelasi, peta; filter di sidebar |
| Preprocessing | Cleaning, pencegahan leakage (dengan demo), split, pipeline, penanganan imbalance |
| Reduksi Dimensi (PCA) | Scree plot interaktif, interpretasi komponen, benchmark dengan/tanpa PCA |
| Modelling & Evaluasi | 5 algoritma: metrik, confusion matrix, ROC/PR, cross-validation, tuning, threshold, fitur penting |
| Prediksi Satu Pelanggan | Pilih 1 dari 5 model, isi profil, probabilitas churn, saran retensi, bandingkan semua model |
| Prediksi Massal | Unggah CSV, hasil prediksi, unduh |
| Kesimpulan & Saran | Kesimpulan otomatis dari hasil, saran bisnis, keterbatasan |

## Struktur
app.py            # pintu masuk + navigasi
utils.py charts.py
halaman/          # satu file per halaman
train_all.py      # melatih 5 model + hitung hasil PCA -> artifacts/
artifacts/        # models/*.joblib dan hasil.json (dibaca website)
data/Telco_customer_churn.csv
.streamlit/config.toml   # tema biru soft
