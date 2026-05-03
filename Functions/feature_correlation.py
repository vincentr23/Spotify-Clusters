import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

def feature_scaling(X):
  scaler = StandardScaler()
  X_scaled = scaler.fit_transform(X)
  return X_scaled

def feature_correlation(X):
  X_scaled = feature_scaling(X)
  # Find correlation between audio features
  return pd.DataFrame(X_scaled, columns=X.columns).corr()

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
