#!/usr/bin/env python3
"""
Bird species presence/abundance prototype (tabular):
- Loads eBird-style sightings and NOAA-style climate CSVs
- Merges on [date, region]
- Optional species filtering (e.g., "Red-breasted Nuthatch")
- Feature engineering (month, day_of_year, rolling aggregates optional)
- Train/val split
- Normalization and one-hot encoding for categoricals
- PyTorch MLP with early stopping
- Supports regression target ("sightings") or binary classification ("present")

Usage examples:
  python bird-migration-main.py --ebird ebird_data.csv --climate climate_data.csv \
      --species "Red-breasted Nuthatch" --target sightings

  python bird-migration-main.py --ebird ebird_data.csv --climate climate_data.csv \
      --target present

CSV expectations (minimum):
  ebird_data.csv   : date, region, species, sightings OR present (0/1)
  climate_data.csv : date, region, <numeric climate columns like temp, precip>

Outputs:
  - model.pt (trained weights)
  - metrics.json (basic performance numbers)
  - feature_columns.json (for inference mapping)
"""

from __future__ import annotations
import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ------------------------------
# Reproducibility
# ------------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ------------------------------
# Data utilities
# ------------------------------

def load_and_merge(ebird_path: str, climate_path: str) -> pd.DataFrame:
    ebird = pd.read_csv(ebird_path)
    climate = pd.read_csv(climate_path)

    required_ebird_cols = {"date", "region"}
    required_climate_cols = {"date", "region"}
    if not required_ebird_cols.issubset(ebird.columns):
        raise ValueError(f"ebird_data missing required columns: {required_ebird_cols}")
    if not required_climate_cols.issubset(climate.columns):
        raise ValueError(f"climate_data missing required columns: {required_climate_cols}")

    # Ensure datetime
    ebird["date"] = pd.to_datetime(ebird["date"], errors="coerce")
    climate["date"] = pd.to_datetime(climate["date"], errors="coerce")

    # Merge
    df = pd.merge(ebird, climate, on=["date", "region"], how="inner")

    # Basic temporal features
    df["month"] = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear

    # Drop obvious bad rows
    df = df.dropna(subset=["region", "date"]).reset_index(drop=True)
    return df


def filter_species(df: pd.DataFrame, species: Optional[str]) -> pd.DataFrame:
    if species is None:
        return df
    if "species" not in df.columns:
        print("[warn] --species given but 'species' column not found; skipping filter.")
        return df
    m = df["species"].astype(str).str.lower() == species.lower()
    out = df[m].copy()
    if out.empty:
        print(f"[warn] No rows matched species='{species}'. Proceeding without filter.")
        return df
    return out


def infer_target(df: pd.DataFrame, target_hint: Optional[str]) -> Tuple[str, str]:
    """Return (task_type, target_col)
    task_type in {"regression", "classification"}
    """
    if target_hint is not None:
        target_hint = target_hint.strip().lower()
        if target_hint in {"sightings", "abundance"} and "sightings" in df.columns:
            return "regression", "sightings"
        if target_hint in {"present", "presence"} and "present" in df.columns:
            return "classification", "present"
        # Fallback to auto in case hint doesn't match columns

    # Auto-detect
    if "sightings" in df.columns:
        return "regression", "sightings"
    if "present" in df.columns:
        return "classification", "present"
    raise ValueError("Could not infer target. Provide --target (sightings|present) and ensure the column exists.")


def build_features(
    df: pd.DataFrame,
    target_col: str,
    use_one_hot_region: bool = True,
    drop_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    drop_cols = drop_cols or []

    # Keep a copy to avoid SettingWithCopy issues
    df = df.copy()

    # Typical non-feature columns to drop
    default_drop = {"date"}
    if target_col in df.columns:
        default_drop.add(target_col)
    if "species" in df.columns:
        default_drop.add("species")  # after filtering, we usually drop it

    default_drop.update(drop_cols)

    # One-hot encode region (simple & robust)
    if use_one_hot_region and "region" in df.columns:
        df = pd.get_dummies(df, columns=["region"], drop_first=True)

    # Select features: numeric + engineered + one-hot
    # Anything left that's not obviously non-numeric becomes candidate; coerce numeric
    for col in df.columns:
        if col not in default_drop:
            df[col] = pd.to_numeric(df[col], errors="ignore")

    # After coercion, separate target
    y = df[target_col] if target_col in df.columns else None
    X = df.drop(columns=list(default_drop))

    # Remove non-numeric columns if any remain
    non_numeric = [c for c in X.columns if not np.issubdtype(X[c].dtype, np.number)]
    if non_numeric:
        print(f"[info] Dropping non-numeric feature columns: {non_numeric}")
        X = X.drop(columns=non_numeric)

    # Final NaN cleanup
    valid = X.notna().all(axis=1)
    if y is not None:
        valid &= y.notna()
    X = X.loc[valid].reset_index(drop=True)
    y = y.loc[valid].reset_index(drop=True)

    return X, y


# ------------------------------
# Torch dataset + model
# ------------------------------

class TabularDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        if self.y.ndim == 1:
            self.y = self.y.unsqueeze(1)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, depth: int = 2, out_dim: int = 1):
        super().__init__()
        layers: List[nn.Module] = []
        d = in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(p=0.1)]
            d = hidden
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ------------------------------
# Training / Evaluation
# ------------------------------

@dataclass
class TrainConfig:
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 50
    early_stopping_patience: int = 7


def train_loop(model, train_loader, val_loader, optimizer, loss_fn, task_type: str, device: str) -> Tuple[float, float]:
    best_val = float("inf")
    patience = 0
    best_state = None

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_losses = []
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(Xb)
            loss = loss_fn(out, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                out = model(Xb)
                val_loss = loss_fn(out, yb)
                val_losses.append(val_loss.item())

        mean_train = float(np.mean(train_losses)) if train_losses else math.nan
        mean_val = float(np.mean(val_losses)) if val_losses else math.nan
        print(f"Epoch {epoch:03d} | train_loss={mean_train:.4f} | val_loss={mean_val:.4f}")

        # Early stopping on val loss
        if mean_val < best_val - 1e-6:
            best_val = mean_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= config.early_stopping_patience:
                print("[info] Early stopping triggered.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_val, mean_train


def evaluate(model, X, y, task_type: str, device: str) -> dict:
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32, device=device)
        y_t = torch.tensor(y, dtype=torch.float32, device=device)
        if y_t.ndim == 1:
            y_t = y_t.unsqueeze(1)
        preds = model(X_t)

    preds_np = preds.squeeze(1).cpu().numpy()
    y_np = y_t.squeeze(1).cpu().numpy()

    metrics = {}
    if task_type == "regression":
        # RMSE on original scale if we trained on log1p, handle both cases
        # Here we assume we trained on log1p(sightings). We'll track both scales.
        rmse = float(np.sqrt(np.mean((preds_np - y_np) ** 2)))
        metrics["rmse"] = rmse
    else:
        # classification: compute accuracy & AUROC if possible
        from sklearn.metrics import accuracy_score, roc_auc_score
        probs = torch.sigmoid(torch.tensor(preds_np)).numpy()
        pred_labels = (probs >= 0.5).astype(np.float32)
        try:
            auc = float(roc_auc_score(y_np, probs))
        except Exception:
            auc = float("nan")
        acc = float(accuracy_score(y_np, pred_labels))
        metrics.update({"accuracy": acc, "auc": auc})
    return metrics


# ------------------------------
# Main
# ------------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Bird presence/abundance prototype")
    ap.add_argument("--ebird", type=str, default="ebird_data.csv", help="Path to eBird sightings CSV")
    ap.add_argument("--climate", type=str, default="climate_data.csv", help="Path to NOAA climate CSV")
    ap.add_argument("--species", type=str, default=None, help="Filter to a specific species name")
    ap.add_argument("--target", type=str, default=None, choices=["sightings", "present"], help="Target column hint")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--val_size", type=float, default=0.2, help="Proportion of TRAIN used for validation")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)

    df = load_and_merge(args.ebird, args.climate)
    df = filter_species(df, args.species)

    task_type, target_col = infer_target(df, args.target)
    print(f"[info] task_type={task_type} target_col={target_col}")

    # If regression on counts, log1p-transform to stabilize; keep a copy for reporting
    if task_type == "regression":
        if not np.issubdtype(df[target_col].dtype, np.number):
            raise ValueError(f"Target '{target_col}' must be numeric for regression.")
        df["target_raw"] = df[target_col]
        df[target_col] = np.log1p(df[target_col].clip(lower=0))

    X_df, y_series = build_features(df, target_col=target_col)

    # Train/val/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_df.values, y_series.values, test_size=args.test_size, random_state=args.seed
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=args.val_size, random_state=args.seed
    )

    # Scale numeric features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    np.savez("scaler_stats.npz", mean=scaler.mean_, scale=scaler.scale_)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # Model
    in_dim = X_train.shape[1]
    out_dim = 1
    model = MLP(in_dim=in_dim, hidden=64, depth=2, out_dim=out_dim)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)



    # Torch datasets
    train_ds = TabularDataset(X_train, y_train)
    val_ds = TabularDataset(X_val, y_val)
    test_ds = TabularDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)


    # Loss & optimizer
    if task_type == "regression":
        loss_fn = nn.MSELoss()
    else:
        loss_fn = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Train
    config = TrainConfig(
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
    )

    best_val, last_train = train_loop(model, train_loader, val_loader, optimizer, loss_fn, task_type, device)

    # Evaluate
    metrics_val = evaluate(model, X_val, y_val, task_type, device)
    metrics_test = evaluate(model, X_test, y_test, task_type, device)

    results = {
        "task_type": task_type,
        "target": target_col,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "n_test": len(test_ds),
        "val_loss": best_val,
        "train_loss": last_train,
        "val_metrics": metrics_val,
        "test_metrics": metrics_test,
        "feature_shape": in_dim,
        "device": device,
    }

    # Save artifacts
    torch.save(model.state_dict(), "model.pt")
    np.savez("scaler_stats.npz", mean=scaler.mean_, scale=scaler.scale_)
    with open("metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    with open("feature_columns.json", "w") as f:
        json.dump({"columns": list(X_df.columns)}, f, indent=2)

    print("\n[done] Saved: model.pt, metrics.json, feature_columns.json, scaler_stats.npz")
    print("[hint] For inference, replicate preprocessing (get_dummies, StandardScaler) before model.forward().")