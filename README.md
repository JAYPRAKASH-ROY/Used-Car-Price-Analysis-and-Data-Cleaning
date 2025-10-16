# Quikr Car Price Estimator — Streamlit (Deploy Ready)

Deploy-ready Streamlit app tailored to **Cleaned_Car_data.csv** with columns:
`name, company, year, Price, kms_driven, fuel_type`.

## What it does
- Cleans the Quikr data (parses `Price`, `kms_driven`, `year`)
- Trains a Linear Regression pipeline with OneHotEncoder on (`company`, `fuel_type`)
- Predicts car prices from (`company`, `fuel_type`, `year`, `kms`)
- Single prediction + batch CSV predictions
- Shows cleaning and metrics

## Run locally
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Deploy (Streamlit Community Cloud)
1. Push: `Cleaned_Car_data.csv`, `streamlit_app.py`, `requirements.txt`, `model_joblib.pkl`, `metrics.json`
2. Set main file to `streamlit_app.py` and deploy.

## Notes
- The app expects dataset column names exactly as in the CSV.
- You can extend features by including `name` or other engineered features.
