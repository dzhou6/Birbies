import torch, torch.nn as nn
import matplotlib.pyplot as plt
import geopandas as gpd
class MigrationNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x): return self.layers(x)

# Train with PyTorch
