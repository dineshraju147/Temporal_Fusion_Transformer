# ---------------------------------------------------------
# predict_m5.py (TFT-Lite Version)
#
# UPDATED:
# - Calculates metrics across the *entire* validation set
# - Adds Error Histogram and Actual vs. Predicted plots
# ---------------------------------------------------------

import os
import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

# --- IMPORTS ---
from m5_preprocessor.preprocessor import load_and_preprocess_daily
from models.tft_lite_model import TFTLite
from config.feature_config import CONT_FEATS, CAT_FEATS, validate_features
from train_m5 import TSDS, build_sequences # Reuse from train script
# ------------------------


@torch.no_grad()
def predict_and_visualize(config_dataset, config_model, num_samples=3):
    
    print("\n" + "="*70)
    print("🔮 M5 TFT-LITE - PREDICTION PIPELINE")
    print("="*70 + "\n")
    
    # === STEP 1: LOAD AND PREPROCESS DATA ===
    df = load_and_preprocess_daily(config_dataset)

    # === STEP 2: BUILD SEQUENCES ===
    ENC_LEN = config_model["FEATURE_PARAMS"]["ENC_LEN"]
    PRED_DAYS = config_model["FEATURE_PARAMS"]["PRED_DAYS"]
    
    Xec, Xea, Xdc, Y, _, _ = build_sequences(df, ENC_LEN, PRED_DAYS)
    
    # === STEP 3: GET VALIDATION SET ===
    print("\nSplitting sequences to get validation set...")
    _, Xec_va, _, Xea_va, _, Xdc_va, _, Y_va = train_test_split(
        Xec, Xea, Xdc, Y, 
        test_size=config_dataset["dataset"]["val_split_pct"], 
        random_state=config_model["SEED"],
        shuffle=True # Must use same shuffle as training
    )
    print(f"Validation windows available: {len(Y_va):,}")

    # === STEP 4: CREATE VALIDATION DATALOADER ===
    valid_ds = TSDS(Xec_va, Xea_va, Xdc_va, Y_va)
    valid_loader = DataLoader(
        valid_ds, 
        batch_size=config_model["TRAIN_PARAMS"]["BATCH_SIZE"] * 2, # Use larger batch for pred
        shuffle=False, 
        num_workers=0
    )

    # === STEP 5: LOAD MODEL ===
    print("\n" + "="*70)
    print("STEP 5: Loading trained model...")
    print("-" * 70)
    
    checkpoint_path = config_model["CHECKPOINT_PATH"]
    try:
        model = TFTLite.load_from_checkpoint(checkpoint_path)
        model.eval()
        
        device = torch.device(
            "cuda" if torch.cuda.is_available() 
            else "mps" if torch.backends.mps.is_available() 
            else "cpu"
        )
        model.to(device)
        print(f"✅ Model loaded from: {checkpoint_path}")
        print(f"   Running on device: {device}")
        
    except FileNotFoundError:
        print(f"🚨 ERROR: Model checkpoint not found at {checkpoint_path}")
        print("   Run: python train_m5.py")
        return
    except Exception as e:
        print(f"🚨 ERROR loading model: {e}")
        return

    # === STEP 6: GENERATE PREDICTIONS (ENTIRE VALIDATION SET) ===
    print("\nGenerating predictions for entire validation set...")
    
    all_preds_q = []
    all_y_true = []
    
    for batch in valid_loader:
        enc_cont, enc_cat, dec_cat, y_true = batch
        
        # Move batch to device
        enc_cont = enc_cont.to(device)
        enc_cat = enc_cat.to(device)
        dec_cat = dec_cat.to(device)
        
        # Predict
        preds_q = model(enc_cont, enc_cat, dec_cat).cpu().numpy() # (B, H, 3)
        
        all_preds_q.append(preds_q)
        all_y_true.append(y_true.numpy())

    # Concatenate all batches
    all_preds_q = np.concatenate(all_preds_q, axis=0)
    all_y_true = np.concatenate(all_y_true, axis=0)
    
    print(f"Total prediction shape: {all_preds_q.shape}") # (Total_Samples, Horizon, 3)
    print(f"Total true shape: {all_y_true.shape}")     # (Total_Samples, Horizon)

    # === STEP 7: CALCULATE DATASET-WIDE METRICS ===
    print("\n" + "="*70)
    print("STEP 7: Calculating Performance Metrics...")
    print("="*70)

    # Get P50 (median) predictions and ensure non-negative
    p10_preds = np.maximum(0, all_preds_q[:, :, 0])
    p50_preds = np.maximum(0, all_preds_q[:, :, 1])
    p90_preds = np.maximum(0, all_preds_q[:, :, 2])

    # Flatten for overall metrics
    y_flat = all_y_true.flatten()
    p50_flat = p50_preds.flatten()
    p10_flat = p10_preds.flatten()
    p90_flat = p90_preds.flatten()

    # 1. MAE (Mean Absolute Error) on P50
    mae = np.mean(np.abs(y_flat - p50_flat))
    print(f"  • P50 MAE (Median):    {mae:.4f}")

    # 2. MAPE (Mean Absolute Percentage Error) on P50
    # Use 1e-3 to avoid division by zero
    mape = np.mean(np.abs((y_flat - p50_flat) / np.maximum(y_flat, 1e-3))) * 100
    print(f"  • P50 MAPE (Median):   {mape:.2f}%")
    
    # 3. P10-P90 Coverage
    in_range = (y_flat >= p10_flat) & (y_flat <= p90_flat)
    coverage = np.mean(in_range) * 100
    print(f"  • P10-P90 Coverage:  {coverage:.2f}% (Target: 80%)")

    # 4. Quantile Loss (for reference, should match val_loss)
    q_loss = model.quantile_loss( # <--- Use the loaded 'model' instance
        torch.tensor(all_preds_q).to(device), # <--- Move tensor to model's device
        torch.tensor(all_y_true).to(device)   # <--- Move tensor to model's device
    ).item()
    print(f"  • Mean Quantile Loss: {q_loss:.4f} (Matches val_loss)")
    
    print("\nInterpretation:")
    print(f"  → Your median (P50) forecast is, on average, {mae:.2f} units off.")
    if coverage < 70 or coverage > 90:
        print(f"  → Your coverage ({coverage:.2f}%) is far from 80%. The model's uncertainty estimates are not well-calibrated.")
    else:
        print(f"  → Your coverage ({coverage:.2f}%) is good. The model has a reasonable sense of uncertainty.")


    # === STEP 8: VISUALIZE (Fan Charts) ===
    print("\n" + "="*70)
    print(f"STEP 8: Plotting {num_samples} random 'fan-charts'...")
    print("="*70)
    
    fig, axes = plt.subplots(num_samples, 1, figsize=(14, 5 * num_samples), squeeze=False)
    axes = axes.flatten()
    
    # Get random samples from the *entire* validation set
    plot_indices = np.random.choice(len(all_y_true), num_samples, replace=False)

    for i, b_idx in enumerate(plot_indices):
        ax = axes[i]
        
        # Get quantiles for this sample
        p10, p50, p90 = p10_preds[b_idx], p50_preds[b_idx], p90_preds[b_idx]
        truth = all_y_true[b_idx]
        
        # Plot
        ax.plot(truth, label='True Sales', color='blue', marker='.')
        ax.plot(p50, label='Forecast (P50)', color='red', linestyle='--')
        ax.fill_between(
            np.arange(len(p50)), p10, p90, 
            alpha=0.2, color='red', label='P10-P90 Range'
        )
        
        ax.set_title(f"Sample {b_idx}: {PRED_DAYS}-Day Forecast")
        ax.set_xlabel(f"Forecast Day (T+{PRED_DAYS})")
        ax.set_ylabel("Sales")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    output_path_fan = "m5_tft_forecast_visualization.png"
    plt.savefig(output_path_fan, dpi=150)
    print(f"✅ Fan-chart visualization saved to: {output_path_fan}")
    plt.close(fig)


    # === STEP 9: NEW PLOTS (Histogram & Scatter) ===
    print("\n" + "="*70)
    print(f"STEP 9: Plotting overall performance graphs...")
    print("="*70)
    
    fig_perf, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # 1. Histogram of P50 Errors
    errors = y_flat - p50_flat
    ax1.hist(errors, bins=100, alpha=0.7, label='P50 Error (Actual - Pred)')
    ax1.axvline(errors.mean(), color='red', linestyle='--', label=f'Mean Error: {errors.mean():.2f}')
    ax1.set_title('Histogram of P50 Forecast Errors')
    ax1.set_xlabel('Error')
    ax1.set_ylabel('Frequency')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.5)

    # 2. Scatter Plot: Actual vs. Predicted
    # Use a subset of points for performance if the dataset is too large
    plot_sample_size = min(50000, len(y_flat))
    scatter_indices = np.random.choice(len(y_flat), plot_sample_size, replace=False)
    
    ax2.scatter(
        y_flat[scatter_indices], 
        p50_flat[scatter_indices], 
        alpha=0.1, 
        label='Actual vs. P50'
    )
    # Add y=x line
    lims = [
        np.min([ax2.get_xlim(), ax2.get_ylim()]),
        np.max([ax2.get_xlim(), ax2.get_ylim()]),
    ]
    ax2.plot(lims, lims, 'r--', alpha=0.75, zorder=5, label='Perfect Forecast (y=x)')
    ax2.set_xlabel('Actual Sales')
    ax2.set_ylabel('Predicted Sales (P50)')
    ax2.set_title('Actual vs. Predicted (P50)')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    output_path_perf = "m5_tft_performance_plots.png"
    plt.savefig(output_path_perf, dpi=150)
    print(f"✅ Performance plots saved to: {output_path_perf}")
    plt.close(fig_perf)

    print("\n" + "="*70)
    print("🔮 PREDICTION PIPELINE COMPLETE!")
    print("="*70 + "\n")


if __name__ == "__main__":
    try:
        with open("config/config_m5.yaml", "r") as f1, \
             open("config/model_config_m5.yaml", "r") as f2:
            config_dataset = yaml.safe_load(f1)
            config_model = yaml.safe_load(f2)
            
        predict_and_visualize(config_dataset, config_model, num_samples=3) 

    except FileNotFoundError as e:
        print(f"🚨 CONFIG FILES NOT FOUND: {e}")
    except Exception as e:
        print(f"\n🚨 ERROR DURING PREDICTION: {e}")
        import traceback
        traceback.print_exc()