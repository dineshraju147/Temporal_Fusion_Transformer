"""
Feature Configuration for M5 Forecasting
=========================================
(Adapted from Kaggle TFT-Lite Notebook)

Centralized feature definitions to ensure consistency.
"""

# --- Features from Notebook Cell 6 ---

# Continuous features (past-observed)
CONT_FEATS = [
    'sell_price',
    'lag_1',
    'lag_7',
    'lag_28',
    'rolling_7',
    'rolling_28'
]

# Categorical features (past and future-known)
CAT_FEATS  = [
    'wday',
    'month',
    'event_name_1',
    'store_id',
    'dept_id',
    'cat_id'
]

# --- Config for Preprocessor ---
LAG_CONFIG = {
    "lags": [1, 7, 28],
    "rolling_windows": [7, 28],
}


def get_feature_counts(df):
    """Gets the embedding sizes for categorical features"""
    cat_maps = {}
    for c in CAT_FEATS:
        cat_maps[c] = {v:i for i,v in enumerate(df[c].astype(str).unique())}
    
    n_cat_maps = [len(cat_maps[c]) for c in CAT_FEATS]
    n_cont = len(CONT_FEATS)
    return n_cat_maps, n_cont

def validate_features(df):
    """Validates that all required features exist in the dataframe."""
    missing_cont = [col for col in CONT_FEATS if col not in df.columns]
    missing_cat = [col for col in CAT_FEATS if col not in df.columns]
    missing = missing_cont + missing_cat
    return (len(missing) == 0, missing)