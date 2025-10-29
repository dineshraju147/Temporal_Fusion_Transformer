# ---------------------------------------------------------
# train_m5.py (TFT-Lite Version)
# Trains the TFT-Lite model using PyTorch Lightning
# ---------------------------------------------------------

import os
import yaml
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

# --- IMPORTS ---
from m5_preprocessor.preprocessor import load_and_preprocess_daily
from models.tft_lite_model import TFTLite
from config.feature_config import (
    CONT_FEATS, CAT_FEATS, get_feature_counts, validate_features
)
# ------------------------

class TSDS(Dataset):
    """Time Series Dataset (from notebook Cell 7)"""
    def __init__(self, Xec, Xea, Xdc, Y):
        self.Xec = torch.from_numpy(Xec)
        self.Xea = torch.from_numpy(Xea)
        self.Xdc = torch.from_numpy(Xdc)
        self.Y   = torch.from_numpy(Y)
    def __len__(self):
        return len(self.Y)
    def __getitem__(self, i):
        return self.Xec[i], self.Xea[i], self.Xdc[i], self.Y[i]

def build_sequences(df, enc_len, pred_days):
    """Builds sequences from the dataframe (notebook Cell 6)"""
    
    # 1. Encode categories to ints
    print("Encoding categorical features...")
    cat_maps = {}
    for c in CAT_FEATS:
        cat_maps[c] = {v:i for i,v in enumerate(df[c].astype(str).unique())}
        df[c+'_idx'] = df[c].astype(str).map(cat_maps[c])
    CAT_IDX = [c+'_idx' for c in CAT_FEATS]
    
    # 2. Scale continuous features
    print("Scaling continuous features...")
    scaler = StandardScaler()
    df[CONT_FEATS] = scaler.fit_transform(df[CONT_FEATS])

    # 3. Build per-id time series tensors
    print("Building sequences for each time series...")
    def build_id_windows(g):
        X_enc_cont, X_enc_cat, X_dec_cat, Y = [], [], [], []
        g = g.sort_values('date')
        arr_cont = g[CONT_FEATS].values
        arr_cat  = g[CAT_IDX].values
        y = g['sales'].values
        
        # Start at enc_len to avoid NaNs from lags
        for t in range(enc_len, len(g)-pred_days):
            enc_cont = arr_cont[t-enc_len:t]
            enc_cat  = arr_cat[t-enc_len:t]
            dec_cat  = arr_cat[t:t+pred_days]  # future known cats
            target   = y[t:t+pred_days]
            
            X_enc_cont.append(enc_cont)
            X_enc_cat.append(enc_cat)
            X_dec_cat.append(dec_cat)
            Y.append(target)
            
        return (
            np.array(X_enc_cont, dtype=np.float32),
            np.array(X_enc_cat, dtype=np.int64),
            np.array(X_dec_cat, dtype=np.int64),
            np.array(Y, dtype=np.float32)
        )

    # Run for all groups
    data = [build_id_windows(g) for _,g in df.groupby('id')]
    if not data: 
        raise RuntimeError('No sequences produced. Check ENC_LEN or sample_items.')

    # 4. Concatenate all sequences
    Xec = np.concatenate([d[0] for d in data])
    Xea = np.concatenate([d[1] for d in data])
    Xdc = np.concatenate([d[2] for d in data])
    Y   = np.concatenate([d[3] for d in data])
    
    print(f"\nTotal sequences created: {len(Y):,}")
    print(f"  Encoder Cont shape: {Xec.shape}")
    print(f"  Encoder Cat shape:  {Xea.shape}")
    print(f"  Decoder Cat shape:  {Xdc.shape}")
    print(f"  Target shape:       {Y.shape}")
    
    return Xec, Xea, Xdc, Y, scaler, cat_maps


def train_m5(config_dataset, config_model):
    pl.seed_everything(config_model["SEED"])

    print("\n" + "="*70)
    print("🚀 M5 TFT-LITE MODEL - TRAINING PIPELINE")
    print("="*70 + "\n")

    # === STEP 1: LOAD AND PREPROCESS DATA ===
    df = load_and_preprocess_daily(config_dataset)

    # === STEP 2: VALIDATE FEATURES ===
    features_ok, missing = validate_features(df)
    if not features_ok:
        raise ValueError(f"Missing required features: {missing}")
    print("✅ All required features present.")
    
    # Get categorical embedding sizes
    n_cat_maps, n_cont = get_feature_counts(df)
    print(f"Feature counts: {n_cont} continuous, {len(n_cat_maps)} categorical.")

    # === STEP 3: BUILD SEQUENCES ===
    ENC_LEN = config_model["FEATURE_PARAMS"]["ENC_LEN"]
    PRED_DAYS = config_model["FEATURE_PARAMS"]["PRED_DAYS"]
    
    Xec, Xea, Xdc, Y, scaler, cat_maps = build_sequences(df, ENC_LEN, PRED_DAYS)
    
    # === STEP 4: TRAIN/VAL SPLIT ===
    print("\nSplitting sequences into train/validation...")
    Xec_tr, Xec_va, Xea_tr, Xea_va, Xdc_tr, Xdc_va, Y_tr, Y_va = train_test_split(
        Xec, Xea, Xdc, Y, 
        test_size=config_dataset["dataset"]["val_split_pct"], 
        random_state=config_model["SEED"],
        shuffle=True # Shuffle sequences
    )
    print(f"Train windows: {len(Y_tr):,}")
    print(f"Valid windows: {len(Y_va):,}")

    # === STEP 5: CREATE DATALOADERS ===
    train_ds = TSDS(Xec_tr, Xea_tr, Xdc_tr, Y_tr)
    valid_ds = TSDS(Xec_va, Xea_va, Xdc_va, Y_va)
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=config_model["TRAIN_PARAMS"]["BATCH_SIZE"], 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    valid_loader = DataLoader(
        valid_ds, 
        batch_size=config_model["TRAIN_PARAMS"]["BATCH_SIZE"] * 2, # Larger val batch
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )

    # === STEP 6: BUILD MODEL ===
    print("\nBuilding TFT-Lite model...")
    model = TFTLite(
        n_cat_maps=n_cat_maps,
        n_cont=n_cont,
        enc_len=ENC_LEN,
        pred_days=PRED_DAYS,
        d_model=config_model["MODEL_PARAMS"]["d_model"],
        nhead=config_model["MODEL_PARAMS"]["nhead"],
        dim_ff=config_model["MODEL_PARAMS"]["dim_feedforward"],
        num_enc_layers=config_model["MODEL_PARAMS"]["num_encoder_layers"],
        num_dec_layers=config_model["MODEL_PARAMS"]["num_decoder_layers"],
        p=config_model["MODEL_PARAMS"]["dropout"],
        lr=config_model["TRAIN_PARAMS"]["LEARNING_RATE"]
    )

    # === STEP 7: SETUP TRAINER ===
    checkpoint_path = config_model["CHECKPOINT_PATH"]
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    ckpt_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=os.path.dirname(checkpoint_path),
        filename=os.path.basename(checkpoint_path).replace('.ckpt', ''),
        save_top_k=1,
        mode='min'
    )
    
    es_callback = EarlyStopping(
        monitor='val_loss', 
        patience=config_model["TRAIN_PARAMS"]["EARLY_STOPPING_PATIENCE"], 
        mode='min'
    )
    
    trainer = pl.Trainer(
        max_epochs=config_model["TRAIN_PARAMS"]["EPOCHS"],
        accelerator='auto', # 'gpu' or 'cpu'
        callbacks=[ckpt_callback, es_callback],
        log_every_n_steps=20,
        enable_progress_bar=True
    )

    # === STEP 8: TRAIN ===
    print("\n" + "="*70)
    print("STEP 6: Training model...")
    print("="*70 + "\n")
    
    trainer.fit(model, train_loader, valid_loader)
    
    print("\n" + "="*70)
    print("🎉 TRAINING COMPLETE!")
    print("="*70)
    print(f"✅ Best model saved to: {ckpt_callback.best_model_path}")

if __name__ == "__main__":
    try:
        with open("config/config_m5.yaml", "r") as f1, \
             open("config/model_config_m5.yaml", "r") as f2:
            config_dataset = yaml.safe_load(f1)
            config_model = yaml.safe_load(f2)
            
        train_m5(config_dataset, config_model)
        
    except FileNotFoundError as e:
        print(f"🚨 CONFIG FILES NOT FOUND: {e}")
    except Exception as e:
        print(f"\n🚨 ERROR DURING TRAINING: {e}")
        import traceback
        traceback.print_exc()