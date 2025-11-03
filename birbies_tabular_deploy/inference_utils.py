\
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, depth: int = 2, out_dim: int = 1):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(p=0.1)]
            d = hidden
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

def load_artifacts(model_path="model.pt",
                   feature_columns_path="feature_columns.json",
                   scaler_stats_path="scaler_stats.npz",
                   task_type="classification"):
    # columns
    with open(feature_columns_path, "r") as f:
        feat = json.load(f)
    columns = feat["columns"]
    # scaler
    stats = np.load(scaler_stats_path)
    mean = stats["mean"]
    scale = stats["scale"]
    # model
    in_dim = len(columns)
    out_dim = 1
    model = MLP(in_dim=in_dim, hidden=64, depth=2, out_dim=out_dim)
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.eval()
    return model, columns, mean, scale

def preprocess(df: pd.DataFrame, columns, mean, scale) -> np.ndarray:
    # Align to training one-hots and numeric columns
    # Missing columns -> add zeros; extra columns -> drop
    X = df.copy()
    for col in columns:
        if col not in X.columns:
            X[col] = 0
    X = X[columns]
    # Coerce numeric (non-numeric become NaN -> fill 0)
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.fillna(0.0)
    # Standardize with saved stats
    X = (X.values - mean) / np.where(scale == 0, 1.0, scale)
    return X.astype(np.float32)

def predict_df(model, X: np.ndarray, task_type="classification") -> np.ndarray:
    with torch.no_grad():
        xt = torch.tensor(X, dtype=torch.float32)
        logits = model(xt).squeeze(1).numpy()
    if task_type == "classification":
        # Convert logits -> probability via sigmoid
        import numpy as np
        probs = 1.0 / (1.0 + np.exp(-logits))
        return probs
    else:
        # regression: output is on transformed scale if trained that way
        return logits
