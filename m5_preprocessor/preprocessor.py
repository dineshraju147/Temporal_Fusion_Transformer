"""
M5 Dataset Preprocessor
=======================
(Adapted from Kaggle TFT-Lite Notebook)

Loads raw M5 data and performs feature engineering 
to create a daily-level dataset.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from config.feature_config import LAG_CONFIG

def add_lags_rollings(g, lags, windows):
    """
    Applies lags and rollings from notebook (Cell 5)
    Note: shift(1) is used for rollings to prevent leakage
    """
    g = g.copy()
    for lag in lags:
        g[f'lag_{lag}'] = g['sales'].shift(lag)
    
    for window in windows:
        g[f'rolling_{window}'] = g['sales'].shift(1).rolling(window).mean()
        
    return g

def load_and_preprocess_daily(config_dataset):
    """
    Loads raw M5 data and processes it into a daily feature DataFrame.
    Matches logic from Kaggle notebook Cells 4 & 5.
    """
    
    print("="*70)
    print("🚀 M5 PREPROCESSOR (Daily TFT-Lite Mode)")
    print("="*70)

    # === 1. Load Base Data ===
    DATA_DIR = config_dataset['dataset']['sales_path'].replace('sales_train_validation.csv', '')
    sample_items = config_dataset['dataset'].get('sample_items')
    
    calendar = pd.read_csv(f"{DATA_DIR}/calendar.csv")
    prices = pd.read_csv(f"{DATA_DIR}/sell_prices.csv")
    
    if sample_items:
        print(f"Loading first {sample_items} items (rows) for sampling...")
        sales = pd.read_csv(f"{DATA_DIR}/sales_train_validation.csv", nrows=sample_items)
    else:
        print("Loading all items...")
        sales = pd.read_csv(f"{DATA_DIR}/sales_train_validation.csv")
        
    print(f"✓ Raw sales: {sales.shape}, Prices: {prices.shape}, Calendar: {calendar.shape}")

    # --- Parameters from Notebook (Cell 4) ---
    id_cols = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    d_cols = [c for c in sales.columns if c.startswith('d_')]
    
    # === 2. Melt to Long Format (Notebook Cell 5) ===
    print("Melting sales data to long format...")
    df_long = sales.melt(id_vars=id_cols, value_vars=d_cols, var_name='d', value_name='sales')

    # === 3. Merge with Calendar & Prices (Notebook Cell 5) ===
    print("Merging with calendar and prices...")
    calendar['date'] = pd.to_datetime(calendar['date'])
    
    df = df_long.merge(calendar[['d','date','wm_yr_wk','wday','month','event_name_1']],
                       how='left', on='d')
    df = df.merge(prices, on=['store_id','item_id','wm_yr_wk'], how='left')

    # === 4. Feature Engineering (Notebook Cell 5) ===
    print("Running minimal feature engineering...")
    df['wday'] = df['wday'].astype(int)
    df['month'] = df['month'].astype(int)
    df['event_name_1'] = df['event_name_1'].fillna('None')

    # Sort and create lags/rollings per (id)
    df = df.sort_values(['id','date']).reset_index(drop=True)

    print("Adding lagged features and rolling windows...")
    lags = LAG_CONFIG['lags']
    windows = LAG_CONFIG['rolling_windows']
    
    # Use .apply() for parallel processing if possible, or just groupby
    # Using groupby().apply() as in the notebook
    df = df.groupby('id', group_keys=False).apply(
        lambda g: add_lags_rollings(g, lags, windows)
    )

    # --- CRITICAL: Handle NaNs ---
    # The notebook drops NaNs implicitly by starting the sequence loop
    # later. For a robust preprocessor, we fill NaNs from lags/rolling
    # so we can still scale them.
    
    # Fill price NaNs (e.g., ffill or with mean)
    df['sell_price'] = df.groupby('id')['sell_price'].ffill().bfill()
    df['sell_price'] = df['sell_price'].fillna(df['sell_price'].mean())
    
    # Fill lag/rolling NaNs (with 0, as they represent no past data)
    lag_cols = [col for col in df.columns if 'lag_' in col or 'rolling_' in col]
    print(f"Filling NaNs in {len(lag_cols)} lagged columns with 0...")
    df[lag_cols] = df[lag_cols].fillna(0)
    
    # Drop any remaining NaNs (e.g., from calendar merges, though unlikely)
    df = df.dropna().reset_index(drop=True)
    
    print(f"\n✅ Preprocessing complete. Final shape: {df.shape}")
    print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"   Unique time series: {df['id'].nunique()}")
    print("="*70 + "\n")
    
    return df