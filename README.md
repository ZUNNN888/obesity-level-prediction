# Obesity Level Prediction — Streamlit Prototype

BMDS2003 Data Science deployment prototype for multiclass obesity-level prediction using demographic, physical, eating-habit and lifestyle inputs.

## Files
- `app.py` — Streamlit application and Soft Voting champion-model training pipeline
- `ObesityDataSet_cleaned.csv` — cleaned UCI Obesity Levels dataset (2,087 unique records)
- `model_comparison.csv` — evaluation metrics for five machine-learning models
- `feature_importance.csv` — Random Forest feature-importance summary
- `model_metadata.json` — project/model metadata and tuned parameters
- `requirements.txt` — Python dependencies for Streamlit Community Cloud

The app trains the deployment model from the bundled cleaned dataset when the app starts. This avoids binary-model compatibility problems on Streamlit Community Cloud.

## Deploy on Streamlit Community Cloud
1. Upload every file in this folder to the root of your GitHub repository.
2. Open https://share.streamlit.io/ and create a new app.
3. Select your repository and `main` branch.
4. Set **Main file path** to `app.py`.
5. Deploy.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Disclaimer
This is an academic/educational prototype. It does not provide a medical diagnosis or replace assessment by a qualified healthcare professional.
