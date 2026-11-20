import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from utils import KAMUS, load_data, load_hasil, load_raw

df, raw, hasil = load_data(), load_raw(), load_hasil()
NUM, CAT = hasil["features"]["num"], hasil["features"]["cat"]

st.title("Preprocessing")
st.caption("Menyiapkan data agar siap dilatih dan bebas dari kebocoran informasi (data leakage).")


def prep():
    return ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), NUM),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CAT)])


@st.cache_data
def ukuran_setelah_prep():
    Xtr, Xts, ytr, _ = train_test_split(df[NUM + CAT], df["Churn Value"], test_size=0.2, stratify=df["Churn Value"], random_state=42)
    p = prep().fit(Xtr)
    return Xtr.shape, p.transform(Xtr).shape, len(Xts)


@st.cache_data
def demo_leakage():
    y = df["Churn Value"]
    out = {}
    for nama, kolom in [("Tanpa Churn Score (benar)", NUM + CAT), ("Dengan Churn Score (bocor)", NUM + CAT + ["Churn Score"])]:
        Xtr, Xts, ytr, yts = train_test_split(df[kolom], y, test_size=0.2, stratify=y, random_state=42)
        nm = NUM + (["Churn Score"] if "Churn Score" in kolom else [])
        p = Pipeline([("prep", ColumnTransformer([
            ("num", Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]), nm),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT)])), ("m", LogisticRegression(max_iter=1000, class_weight="balanced"))])
        p.fit(Xtr, ytr)
        out[nama] = roc_auc_score(yts, p.predict_proba(Xts)[:, 1])
    return out


st.progress(1.0, text="Alur: Cleaning -> Pencegahan leakage -> Split -> Imputasi -> Scaling -> One-Hot Encoding")

with st.expander("1. Data cleaning: Total Charges", expanded=True):
    kosong = raw["Total Charges"].astype(str).str.strip() == ""
    c1, c2 = st.columns(2)
    c1.metric("Tipe awal", str(raw["Total Charges"].dtype))
    c2.metric("Baris kosong (spasi)", int(kosong.sum()))
    st.write("`Total Charges` terbaca sebagai teks karena ada 11 nilai kosong, seluruhnya pelanggan baru (tenure = 0). "
             "Kolom diubah ke numerik dengan `pd.to_numeric(errors='coerce')`, lalu nilai kosong diisi **median data latih** di dalam pipeline.")
    st.dataframe(raw.loc[kosong, ["CustomerID", "Tenure Months", "Monthly Charges", "Total Charges", "Churn Label"]])

with st.expander("2. Pencegahan data leakage", expanded=True):
    buang = [k for k, v in KAMUS.items() if v[0] in ("Bocoran", "Target")]
    st.write("Kolom yang dibuang karena bocoran/duplikat target:", ", ".join(f"`{k}`" for k in buang))
    st.write("Kolom identitas, konstan, dan lokasi (`CustomerID, Count, Country, State, City, Zip Code, Lat Long, Latitude, Longitude`) "
             "juga dibuang karena tidak informatif.")
    st.markdown("**Demo:** latih Logistic Regression dengan dan tanpa `Churn Score`.")
    if st.button("Jalankan demo leakage"):
        h = demo_leakage()
        a, b = st.columns(2)
        k = list(h)
        a.metric(k[0], f"AUC {h[k[0]]:.3f}")
        b.metric(k[1], f"AUC {h[k[1]]:.3f}", f"+{h[k[1]] - h[k[0]]:.3f} (tidak wajar)", delta_color="inverse")
        st.error("AUC melonjak tidak wajar karena `Churn Score` dihitung dari status churn. Model akan gagal di dunia nyata.")

with st.expander("3. Train-test split"):
    shape_tr, shape_p, n_test = ukuran_setelah_prep()
    c1, c2, c3 = st.columns(3)
    c1.metric("Data latih", f"{shape_tr[0]:,}")
    c2.metric("Data uji", f"{n_test:,}")
    c3.metric("Proporsi churn", f"{hasil['split']['churn_train'] * 100:.1f}% / {hasil['split']['churn_test'] * 100:.1f}%")
    st.write("Pembagian 80:20 dengan `stratify` agar proporsi churn sama di data latih dan uji. "
             "Split dilakukan **sebelum** transformasi agar tidak ada kebocoran dari scaling/imputasi.")

with st.expander("4. Pipeline: imputasi, scaling, dan encoding"):
    st.markdown(f"- Fitur numerik ({len(NUM)}): {', '.join(NUM)} -> imputasi median -> `StandardScaler`\n"
                f"- Fitur kategorikal ({len(CAT)}): imputasi modus -> `OneHotEncoder`\n"
                f"- Hasil: **{shape_tr[1]} kolom** awal menjadi **{shape_p[1]} kolom** numerik")
    st.code('''num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")),
                     ("scaler", StandardScaler())])
cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                     ("ohe", OneHotEncoder(handle_unknown="ignore"))])
preprocess = ColumnTransformer([("num", num_pipe, num_cols),
                                ("cat", cat_pipe, cat_cols)])''', language="python")

with st.expander("5. Penanganan ketidakseimbangan kelas"):
    st.write("Rasio kelas 2,77 : 1. Penanganan: `stratify` saat split, `class_weight='balanced'` pada Logistic Regression, "
             "Decision Tree, dan Random Forest, serta evaluasi dengan Recall, F1, ROC-AUC, dan PR-AUC.")
