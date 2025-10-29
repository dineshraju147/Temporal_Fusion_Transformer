"""
TFT-Lite Model Architecture
===========================
(Adapted from Kaggle TFT-Lite Notebook Cell 7)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import math

# --- Helper Modules ---

class GatedLinear(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.fc = nn.Linear(d_in, d_out)
        self.gate = nn.Linear(d_in, d_out)
        self.sig = nn.Sigmoid()
    def forward(self, x):
        return self.fc(x) * self.sig(self.gate(x))

class VariableSelector(nn.Module):
    """
    Simple variable selection: per-feature gate to weight cont feats
    """
    def __init__(self, n_feats, d_model):
        super().__init__()
        self.proj = nn.ModuleList([GatedLinear(1, d_model) for _ in range(n_feats)])
        self.softmax = nn.Softmax(dim=2)
    def forward(self, x):  # (B,T,F)
        B,T,F = x.size()
        outs = []
        for i in range(F):
            outs.append(self.proj[i](x[:,:,i:i+1]))
        H = torch.stack(outs, dim=2)  # (B,T,F,d)
        weights = self.softmax(H.mean(dim=1))  # (B,F,d)
        Hw = H * weights.unsqueeze(1)
        return Hw.sum(dim=2)  # (B,T,d)

class SimpleTransformerBlock(nn.Module):
    def __init__(self, d_model=128, nhead=4, dim_ff=256, p=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=p, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff), nn.ReLU(), nn.Dropout(p), nn.Linear(dim_ff, d_model)
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(p)
    def forward(self, x, key_padding_mask=None):
        # Note: The notebook didn't use padding, but it's good practice
        h,_ = self.attn(x,x,x, key_padding_mask=key_padding_mask)
        x = self.ln1(x + self.drop(h))
        h = self.ff(x)
        x = self.ln2(x + self.drop(h))
        return x

# --- Main TFT-Lite Model ---

class TFTLite(pl.LightningModule):
    def __init__(self, n_cat_maps, n_cont, enc_len, pred_days, 
                 d_model=128, nhead=4, dim_ff=256, 
                 num_enc_layers=2, num_dec_layers=1, 
                 p=0.1, lr=2e-3):
        
        super().__init__()
        self.save_hyperparameters()
        
        # Quantile loss setup
        self.quantiles = torch.tensor([0.1,0.5,0.9], dtype=torch.float32)

        # === Embeddings & Projections ===
        # 1. Categorical embeddings (one for each cat feature)
        self.cat_embs = nn.ModuleList(
            [nn.Embedding(num_classes, d_model) for num_classes in n_cat_maps]
        )
        # 2. Continuous feature projection (using VariableSelector)
        self.varsel = VariableSelector(n_cont, d_model)

        # === Encoder ===
        self.enc = nn.Sequential(
            *[SimpleTransformerBlock(d_model, nhead, dim_ff, p) for _ in range(num_enc_layers)]
        )
        
        # === Decoder ===
        self.dec = nn.Sequential(
            *[SimpleTransformerBlock(d_model, nhead, dim_ff, p) for _ in range(num_dec_layers)]
        )
        
        # === Head ===
        # Output is 3 quantiles (P10, P50, P90)
        self.head = nn.Linear(d_model, 3)

    def _embed_cats(self, arr):
        """Helper to sum embeddings for an array of categorical indices"""
        # arr: (B, L, F_cat)
        embs = [emb(arr[:,:,i]) for i,emb in enumerate(self.cat_embs)]
        return torch.stack(embs, dim=0).sum(0)

    def quantile_loss(self, preds, target):
        """Pinball/Quantile loss"""
        # preds: (B,H,3) target: (B,H)
        q = self.quantiles.to(preds.device).view(1,1,3)
        e = target.unsqueeze(-1) - preds
        return torch.mean(torch.max(q*e, (q-1)*e))

    def forward(self, enc_cont, enc_cat, dec_cat):
        # enc_cont: (B,T,Fc), enc_cat: (B,T,Fk), dec_cat: (B,H,Fk)
        B, T, Fc = enc_cont.size()
        H = dec_cat.size(1)

        # 1. Process continuous features -> (B,T,d)
        h_cont = self.varsel(enc_cont)
        
        # 2. Process categorical features -> (B,T,d)
        h_enc_cat = self._embed_cats(enc_cat)

        # 3. Create Encoder input (sum cont + cat) -> (B,T,d)
        h_enc = h_cont + h_enc_cat
        h_enc = self.enc(h_enc) # (B,T,d)

        # 4. Create Decoder input (future cats) -> (B,H,d)
        h_dec = self._embed_cats(dec_cat)

        # 5. Cross-attend: Concat last encoder step with all decoder steps
        # [B, 1, d] + [B, H, d] -> [B, 1+H, d]
        enc_context = h_enc[:,-1:].repeat(1,H,1) # Use last encoder step as context
        
        # Simple concat-based cross-attention
        # Note: The notebook's implementation is slightly different but this is clearer
        # h = self.dec(torch.cat([h_enc[:,-1:].repeat(1,H,1), h_dec], dim=1))[:, -H:, :]
        
        # A more standard Transformer Decoder approach (using context)
        # We'll stick to the notebook's simpler method:
        # It concatenates last encoder state with decoder inputs
        # and passes it *all* through a standard *encoder* block (SimpleTransformerBlock)
        
        # This is what the notebook did:
        decoder_input = torch.cat([enc_context, h_dec], dim=1) # (B, H+H, d) ?? No, (B, H, d)
        
        # Rereading notebook: h = self.dec(torch.cat([h_enc[:,-1:].repeat(1,H,1), h_dec], dim=1))[:, -H:, :]
        # This is (B, H, d) cat (B, H, d) -> (B, 2*H, d)? No, dim=1 is time.
        # It's (B, 1, d).repeat(1,H,1) -> (B, H, d)
        # Cat with (B, H, d) on dim=1 -> (B, 2*H, d).
        # This seems wrong or I am misreading.
        
        # Ah, notebook: torch.cat([h_enc[:,-1:].repeat(1,H,1), h_dec], dim=1)
        # h_enc[:,-1:] is (B, 1, d)
        # .repeat(1,H,1) is (B, H, d)
        # h_dec is (B, H, d)
        # Concat on dim=1 -> (B, 2*H, d)
        # Then [:, -H:, :] takes the *last H steps*, which are just the decoder outputs!
        
        # Let's try what the notebook *likely* intended:
        # Use last encoder step as memory for decoder
        
        # Re-reading again.
        # h_enc[:,-1:].repeat(1,H,1) -> (B, H, d)
        # h_dec -> (B, H, d)
        # torch.cat(..., dim=1) -> (B, 2*H, d)
        # self.dec(...) -> (B, 2*H, d)
        # [:, -H:, :] -> (B, H, d)
        # This means the decoder input is the context *concatenated with* the future cats,
        # and we only take the outputs corresponding to the future cat positions.
        # This is an unusual but valid way to do it. Let's stick to the notebook.
        
        full_decoder_input = torch.cat([h_enc[:,-1:].repeat(1,H,1), h_dec], dim=1)
        
        # Pass through decoder block(s)
        h = self.dec(full_decoder_input)
        
        # Take only the last H timesteps (corresponding to h_dec)
        h_out = h[:, -H:, :]
        
        # 6. Final Head
        out = self.head(h_out) # (B,H,3)
        return out

    def training_step(self, batch, batch_idx):
        enc_cont, enc_cat, dec_cat, y = batch
        qhat = self(enc_cont, enc_cat, dec_cat)
        loss = self.quantile_loss(qhat, y)
        self.log('train_loss', loss, prog_bar=True, batch_size=y.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        enc_cont, enc_cat, dec_cat, y = batch
        qhat = self(enc_cont, enc_cat, dec_cat)
        loss = self.quantile_loss(qhat, y)
        self.log('val_loss', loss, prog_bar=True, batch_size=y.size(0))
        
        # Log P50 MAE for interpretability
        mae = F.l1_loss(qhat[:, :, 1], y) # P50 is index 1
        self.log('val_mae_p50', mae, prog_bar=True, batch_size=y.size(0))

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }