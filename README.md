# SaaS Customer Churn Analysis

> **A full-stack Data Science project** — from raw CSV ingestion to a live, interactive retention dashboard — built to help SaaS businesses identify and prevent customer churn.

---

## 🏗️ Project Structure

```
SaaS-Churn-Analysis/
├── app.py                 # Main entry point (Navigation & Layout)
├── modules/               # Dashboard Tab Logic
│   ├── overview.py        # "Overview & Trends" tab
│   ├── individual.py      # "Individual Lookup" tab (SHAP Explainer)
│   └── simulator.py       # "What-If Simulator" tab
├── src/                   # Backend Data Engineering
│   ├── __init__.py
│   ├── data_loader.py     # CSV loading logic
│   ├── preprocessing.py   # Data cleaning & feature encoding
│   └── train_model.py     # ML training, evaluation & chart exports
├── data/
│   ├── raw/               # Original source files (Git-ignored)
│   └── processed/         # Cleaned "Gold" dataset
├── sql_queries/           # SQL scripts for DB setup & analysis
├── models/                # Saved model files (Git-ignored)
├── reports/               # Auto-generated charts (feature importance, confusion matrix)
├── main.py                # CLI Pipeline runner (load → clean → train → export)
├── .gitignore
├── README.md
└── requirements.txt
```

---

## 🚀 Quickstart

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Full ML Pipeline
This loads data, trains the model, generates report charts, and exports `dashboard_data.csv`:
```bash
python main.py
```

### 3. Launch the Interactive Dashboard
```bash
streamlit run app.py
```
Then open your browser at `http://localhost:8501`.

---

## 📊 Dashboard Features

| Tab | Description |
|-----|-------------|
| **Overview & Trends** | Color-coded risk histogram, revenue-at-risk by plan, feature importance, and satisfaction heatmap |
| **Individual Lookup** | Select any high-risk account to see their risk score and top 3 SHAP-driven reasons for churn |
| **What-If Simulator** | Adjust sliders to simulate how improving satisfaction or resolution time reduces predicted churn probability |

---

## 🤖 Model Details

- **Algorithm:** Random Forest Classifier (200 estimators)
- **Probability Estimation:** Out-Of-Bag (OOB) predictions for unbiased, realistic risk scores
- **Performance:** ~85% Accuracy | 76% Precision | 63% Recall
- **Explainability:** SHAP TreeExplainer for per-account feature attribution

---

## 📋 Key Findings

- **Platform Engagement** (Active Weekly Minutes) is the single strongest predictor of retention
- **Support Resolution Time** — accounts with slow ticket resolution churn significantly faster
- **Low Satisfaction + High Risk** quadrant is the "Priority Intervention Zone" for the Customer Success team

---

## 🛠️ Tech Stack

`Python` · `Scikit-Learn` · `SHAP` · `Streamlit` · `Pandas` · `Matplotlib` · `Seaborn`
