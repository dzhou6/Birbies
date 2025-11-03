# 🐦 Birbies — Bird Migration Predictor (Deep Learning Project)

**Birbies** is a deep learning project that predicts bird migration and seasonal sighting trends using combined **eBird observation data** and **climate metrics**.  
The model learns from historical sightings and climate conditions (temperature, precipitation, etc.) to estimate future migration intensity by region.

---

## 🚀 Features

- 🧠 **PyTorch deep learning model** trained on eBird + climate datasets  
- 🧩 **Feature scaling and normalization** with saved `scaler_stats.npz`  
- 📊 **Automated CSV merging and preprocessing** using `bird_migration_main.py`  
- 🌤️ **Gradio web app interface** (`app.py`) for easy upload + prediction  
- 📁 **Modular deployment structure** (`birbies_tabular_deploy/`)

---

## 🧰 Tech Stack

| Category | Tools |
|-----------|-------|
| **Language** | Python 3.10+ |
| **Frameworks** | PyTorch, scikit-learn, Gradio |
| **Data Tools** | pandas, NumPy |
| **Deployment** | Local Gradio UI / Hugging Face Spaces ready |

---

## 📦 Folder Structure
birbies/
├── bird_migration_main.py # Model training and dataset merging
├── birbies_tabular_deploy/
│ ├── app.py # Gradio web interface for inference
│ ├── inference_utils.py # Prediction + preprocessing helpers
│ ├── requirements.txt
│ ├── model.pt # (place trained model here)
│ ├── scaler_stats.npz # (place scaler stats here)
│ └── feature_columns.json # (optional: feature ordering)
├── deepbirb.py # Core training logic (older module)
└── README.md
