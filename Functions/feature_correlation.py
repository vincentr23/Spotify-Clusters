import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from yaml import safe_load

# Path setup
data_dir = "../Data"
third_party_dir = "../ThirdParty"
output_dir = "../Out"
file_path = os.path.join(data_dir, "spotify_dataset.csv")

# Load the latest version
df = pd.read_csv(file_path)

# TODO: create yaml and populate with these config
features = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness",
    "valence", "tempo", "duration_ms", "popularity",
    "mode", "time_signature"
]

def feature_scaling(X):
  X["explicit"] = df["explicit"].astype(int)
  scaler = StandardScaler()
  X_scaled = scaler.fit_transform(X)
  return X_scaled

def feature_correlation(X):
  X_scaled = feature_scaling(X)
  # Find correlation between audio features
  corr = pd.DataFrame(X_scaled, columns=X.columns).corr()
  # ---
  return corr

def plot_correlation_matrix(corr):
  fig, ax = plt.subplots(figsize=(10, 8))
  im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
  ax.set_xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
  ax.set_yticks(range(len(corr.columns)), corr.columns)
  for i in range(len(corr.columns)):
      for j in range(len(corr.columns)):
          ax.text(j, i, f"{corr.iat[i, j]:.2f}", ha="center", va="center", fontsize=7)
  fig.colorbar(im, ax=ax)
  ax.set_title("Audio-feature correlation matrix")
  plt.tight_layout()
  plt.show()

def inv_cov_matrix(X):
  # Finding the inverse covariance of the scaled features
  X_scaled = feature_scaling(X)
  cov = np.cov(X_scaled, rowvar=False)
  VI = np.linalg.pinv(cov)
  return VI

feature_correlation(df[features].copy())