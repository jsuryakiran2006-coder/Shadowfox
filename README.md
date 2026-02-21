# Boston House Price Streamlit App

This repository contains a Streamlit app (`boston.py`) that loads a pre-trained model (`boston_house_model.joblib`) and an optional scaler (`scaler.joblib`) to predict Boston house prices.

Quick start

1. (Optional) Create and activate a virtual environment.

2. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

3. Run the Streamlit app:

```bash
streamlit run boston.py
```

Notes

- Ensure `boston_house_model.joblib` is present in the same folder as `boston.py`.
- If `scaler.joblib` is present the app will attempt to scale inputs before prediction.
- The app shows inputs in the sidebar and a `Predict Price` button to compute a result.

If you want, I can run a quick local sanity test to verify model and scaler load (if you allow it).