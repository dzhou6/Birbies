# Birbies — Tabular DL Inference (Gradio / Hugging Face Spaces)

This app serves your **bird presence/abundance** model trained on merged eBird + climate features.
It expects the artifacts produced by your training script:

- `model.pt` — trained PyTorch weights
- `feature_columns.json` — column order used during training (`{"columns": [...]}`)
- `scaler_stats.npz` — StandardScaler stats (`mean`, `scale`) saved from training

## Quick start (local)

```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

In the UI, upload a CSV of rows to score (same schema as training before target),
and download predictions as a CSV.

## Hugging Face Spaces

1. Create a new Space using the **Gradio** template.
2. Upload all files from this folder.
3. Also upload `model.pt`, `feature_columns.json`, and `scaler_stats.npz` into the Space.
4. The Space will auto-build and launch `app.py`.
