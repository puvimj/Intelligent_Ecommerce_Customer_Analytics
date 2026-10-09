# Intelligent E-Commerce Customer Analytics Platform

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Data

Download the **Brazilian E-Commerce Public Dataset by Olist** from Kaggle and put all 9 CSVs directly into `data/raw/`.

## Workflow (run notebooks in order)

1. `01_data_integration.ipynb` — merge 9 tables
2. `02_data_cleaning.ipynb` — clean the merge
3. `03_eda.ipynb` — exploratory analysis
4. `04_feature_engineering.ipynb` — customer / product / seller / time features
5. `05_customer_segmentation.ipynb` — K-Means, Agglomerative, DBSCAN, 5 personas
6. `06_purchase_prediction.ipynb` — 90-day repeat-purchase classifiers
7. `07_clv_prediction.ipynb` — 90-day future-revenue regressors
8. `08_recommendation.ipynb` — (optional) recommender systems
9. `09_explainability.ipynb` — SHAP + feature importance
10. `app.py` — Streamlit dashboard

## Running the dashboard

Requires Phases 4, 5, 6 and 7 to have been run (Phase 8 is optional).

```bash
streamlit run app.py
```

