# Student Performance Analytics

A Streamlit learning demo for exploring synthetic academic data, visualizing patterns, evaluating a Random Forest classifier, and trying example predictions.

## Features
- Interactive attendance and outcome filters
- KPI cards and outcome/score charts
- Feature relationship scatter plots and correlation heatmap
- Train/test split, accuracy, confusion matrix, and feature importance
- Interactive prediction form with class probabilities

## Important
The app generates **synthetic demo data**. It does not use real student records, and model results should not be used to make real educational decisions.

## Run locally
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Publish to GitHub and Streamlit Community Cloud
1. Create a new public GitHub repository named `student-performance-analytics`.
2. Upload `app.py`, `requirements.txt`, and `README.md` (or push this folder with Git).
3. Open https://share.streamlit.io/ and sign in with GitHub.
4. Choose the repository, branch `main`, and main file path `app.py`, then deploy.
5. Copy the real repository and app URLs into your presentation/report.

No GitHub or Streamlit deployment URLs are included here because publishing must happen from an authenticated account; never use a guessed URL.
