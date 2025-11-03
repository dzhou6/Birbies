import gradio as gr
import pandas as pd
from inference_utils import load_artifacts, preprocess, predict_df
import logging, sys

# ---- Logging setup ----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("app.log", encoding="utf-8")
    ]
)
log = logging.getLogger(__name__)

# ---- Config ----
TASK_TYPE = "regression"   # use "classification" if you trained on present=0/1
model = None
columns = None
mean = None
scale = None


# ---- Load & Predict ----
def load_on_start():
    global model, columns, mean, scale
    log.info("Loading artifacts...")
    model, columns, mean, scale = load_artifacts(
        model_path="model.pt",
        feature_columns_path="feature_columns.json",
        scaler_stats_path="scaler_stats.npz",
        task_type=TASK_TYPE
    )
    log.info("Loaded model; %d feature columns.", len(columns))
    return "Artifacts loaded."


def score_csv(file):
    if file is None:
        log.warning("No file uploaded.")
        return None
    log.info("Received file: %s", file.name)
    try:
        df = pd.read_csv(file.name)
        log.info("Input shape: %s", df.shape)
        for drop in ["date", "species", "present", "sightings", "target_raw"]:
            if drop in df.columns:
                df = df.drop(columns=[drop])
        X = preprocess(df, columns, mean, scale)
        preds = predict_df(model, X, task_type=TASK_TYPE)
        out = df.copy()
        out["prediction"] = preds
        log.info("Predicted %d rows.", len(out))
        return out
    except Exception:
        log.exception("Scoring failed.")
        raise


# ---- UI ----
_ = load_on_start()  # eager-load when script starts

with gr.Blocks(title="Birbies — Tabular DL Inference") as demo:
    gr.Markdown("# 🐦 Birbies — Bird Presence/Abundance (Tabular DL)")
    gr.Markdown("Upload a CSV with the same features you used for training (before target).")
    in_file = gr.File(file_count="single", file_types=[".csv"], label="Upload CSV")
    btn = gr.Button("Predict")
    out_df = gr.Dataframe(label="Predictions") 
    btn.click(fn=score_csv, inputs=[in_file], outputs=[out_df])
