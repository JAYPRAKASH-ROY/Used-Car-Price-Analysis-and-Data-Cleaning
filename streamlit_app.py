
import streamlit as st, pandas as pd, numpy as np, joblib, json
from pathlib import Path

st.set_page_config(page_title="Quikr Car Price Estimator", page_icon="🚗", layout="wide")

DATA_PATH = Path("Cleaned_Car_data.csv")
MODEL_PATH = Path("model_joblib.pkl")
METRICS_PATH = Path("metrics.json")

# ---------- Helpers (must mirror training) ----------
def parse_price(x):
    if pd.isna(x): return np.nan
    s = str(x)
    if "ask" in s.lower(): return np.nan
    s = s.replace(",", "").strip()
    try: return float(s)
    except: return np.nan

def parse_kms(x):
    if pd.isna(x): return np.nan
    s = str(x).lower().replace(",", "").replace("kms", "").replace("km", "").strip()
    try: return float(s)
    except: return np.nan

def parse_year(x):
    try: return int(str(x).strip())
    except: return np.nan

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["price_num"] = df["Price"].apply(parse_price)
    df["kms_num"] = df["kms_driven"].apply(parse_kms)
    df["year_num"] = df["year"].apply(parse_year)
    return df

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

def load_metrics():
    try:
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None

st.title("🚗 Quikr Car Price Estimator (Streamlit)")
st.caption("Project-specific app for Cleaned_Car_data.csv with columns: name, company, year, Price, kms_driven, fuel_type.")

df = load_data()
pipe = load_model()
metrics = load_metrics()

tab1, tab2, tab3, tab4 = st.tabs(["🔎 Explore", "🧹 Clean Overview", "🧮 Predict (Single)", "📦 Batch Predict"])

with tab1:
    st.subheader("Raw dataset preview")
    st.dataframe(df.head(30), use_container_width=True)
    st.write("Shape:", df.shape)
    st.write("Columns:", list(df.columns))

with tab2:
    st.subheader("Cleaning summary")
    dfc = df.copy()
    dfc = dfc.dropna(subset=["price_num","kms_num","year_num","company","fuel_type"])
    dfc = dfc[(dfc["price_num"]>10000) & (dfc["price_num"]<1.5e7)]
    dfc = dfc[(dfc["year_num"]>=1995) & (dfc["year_num"]<=2025)]
    dfc = dfc[(dfc["kms_num"]>=0) & (dfc["kms_num"]<=500000)]
    st.write("Rows after cleaning:", len(dfc))
    st.write("Price (₹) summary after cleaning:")
    st.dataframe(dfc["price_num"].describe().to_frame().T)

with tab3:
    st.subheader("Single prediction")
    c1,c2 = st.columns(2)
    with c1:
        company = st.selectbox("Company", sorted(df["company"].dropna().astype(str).unique().tolist()))
        fuel = st.selectbox("Fuel type", sorted(df["fuel_type"].dropna().astype(str).unique().tolist()))
    with c2:
        year = st.number_input("Year", min_value=1990, max_value=2026, value=2015, step=1)
        kms = st.number_input("Kms driven", min_value=0, max_value=1000000, value=50000, step=1000)
    if st.button("Estimate price", use_container_width=True):
        X = pd.DataFrame([{"company": company, "fuel_type": fuel, "year_num": year, "kms_num": kms}])
        try:
            y = float(pipe.predict(X)[0])
            st.metric("Estimated price", f"₹ {y:,.0f}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

with tab4:
    st.subheader("Batch predict from CSV")
    st.write("Upload a CSV with at least the columns: company, fuel_type, year or year_num, kms_driven or kms_num.")
    up = st.file_uploader("Upload CSV", type=["csv"])
    if up:
        data = pd.read_csv(up)
        # Map/clean columns
        if "year_num" not in data.columns and "year" in data.columns:
            data["year_num"] = data["year"].apply(parse_year)
        if "kms_num" not in data.columns and "kms_driven" in data.columns:
            data["kms_num"] = data["kms_driven"].apply(parse_kms)
        needed = ["company","fuel_type","year_num","kms_num"]
        missing = [c for c in needed if c not in data.columns]
        if missing:
            st.error(f"Missing columns after cleaning: {missing}")
        else:
            preds = pipe.predict(data[needed])
            out = data.copy()
            out["predicted_price"] = preds
            st.dataframe(out.head(30), use_container_width=True)
            st.download_button("Download predictions CSV",
                               out.to_csv(index=False).encode("utf-8"),
                               file_name="car_price_predictions.csv",
                               mime="text/csv")

st.divider()
st.subheader("Model metrics")
if metrics:
    st.json(metrics)
else:
    st.caption("No metrics available.")
