# backend/app/model.py

import os
import yaml
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler

# ─── 1. Load config & model ─────────────────────────────────────────────────────
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
cfg_path = os.path.join(base_dir, "config", "config.yaml")
with open(cfg_path) as f:
    cfg = yaml.safe_load(f)

model_path = os.path.join(base_dir, cfg["model"]["path"])
xgb_model  = joblib.load(model_path)

# ─── 2. Load processed training data columns for reindexing ──────────────────────
#    We assume your processed CSV has a 'Risk' column you can drop.
proc_csv = os.path.abspath(os.path.join(base_dir, "..", "data", "processed",
                                        "german_credit_data_processed.csv"))
train_df = pd.read_csv(proc_csv)
FEATURE_COLS = [c for c in train_df.columns if c != "Risk"]

# ─── 3. Preprocessing & feature‐engineering ──────────────────────────────────────

def preprocess_input(record: dict) -> pd.DataFrame:
    """
    Takes one JSON record with the 9 raw fields and returns
    a 1×31 DataFrame matching the training features.
    """
    df = pd.DataFrame([record])
    
    # 3.1 Numeric features: scale credit_amount and duration
    #     Using robust scaling as in your notebook
    scaler = RobustScaler()
    df[["Credit_amount", "Duration"]] = scaler.fit_transform(
        df[["Credit_amount", "Duration"]]
    )
    
    # 3.2 Sex → int (assuming training used 0=male,1=female or vice versa)
    #     If you encoded differently, swap the mapping here:
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    
    # 3.3 Age → Age_cat bins & dummies
    bins  = [18, 25, 35, 60, 120]
    cats  = ["Student", "Young", "Adult", "Senior"]
    df["Age_cat"] = pd.cut(df["Age"], bins=bins, labels=cats)
    df = pd.get_dummies(
        df,
        columns=["Age_cat"],
        prefix="Age_cat",
        drop_first=True  # drops 'Age_cat_Student'
    )
    df.drop(columns="Age", inplace=True)
    
    # 3.4 Duration → Year_cat bins & dummies
    year_bins  = [0, 12, 24, 36, 48, 60, 72]
    year_labels = [
        "0-12 months", "Year_1-2 year", "Year_2-3 year",
        "Year_3-4 year", "Year_4-5 year", "Year_5-6 year"
    ]
    df["Year_cat"] = pd.cut(
        df["Duration"], bins=year_bins, labels=year_labels, right=False
    )
    df = pd.get_dummies(
        df,
        columns=["Year_cat"],
        prefix="Year",
        drop_first=True  # drops 'Year_cat_0-12 months'
    )
    
    # 3.5 One-hot encode the multi-class fields
    to_dummy = [
        "Job", "Housing",
        "Saving_accounts", "Checking_account",
        "Purpose"
    ]
    df = pd.get_dummies(
        df,
        columns=to_dummy,
        drop_first=True,  # drops the first category in each
        prefix_sep="_"
    )
    
    # 3.6 Reindex to ensure all training‐time features are present
    #     Missing columns → 0, Extra columns → dropped
    df = df.reindex(columns=FEATURE_COLS, fill_value=0)
    
    return df

def predict_risk(data: dict) -> tuple[str, float]:
    """
    data: JSON payload converted to dict from CreditRiskRequest.
    Returns: (label, probability).
    """
    processed = preprocess_input(data)
    probs = xgb_model.predict_proba(processed.values)[0]
    # probs[0] = P(bad), probs[1] = P(good)
    label = "Good credit" if probs[1] >= probs[0] else "Bad credit"
    return label, float(max(probs))
