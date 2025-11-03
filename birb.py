import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
# eBird sightings
ebird = pd.read_csv("ebird_data.csv")
# NOAA climate data
climate = pd.read_csv("climate_data.csv")

# Merge on date and region
df = pd.merge(ebird, climate, on=["date","region"])

# Feature engineering
df["month"] = pd.to_datetime(df["date"]).dt.month
df = df.dropna()
