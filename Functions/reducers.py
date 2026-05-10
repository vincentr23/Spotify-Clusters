from sklearn.manifold import trustworthiness
import numpy as np

import cudf, cuml
from cuml.manifold import TSNE, UMAP
import plotly.express as px
import plotly.io as pio
import hdbscan

def reduce(X, y, combo, id_data):
    n_neighbors, min_dist = combo
    reducer = UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="euclidean",
        random_state=42
    )

    embedding = reducer.fit_transform(X)

    clusterer = hdbscan.HDBSCAN(min_cluster_size=15)
    labels = clusterer.fit_predict(embedding)

    labels_plot = labels.copy()
    labels_plot = labels_plot.astype(str)
    labels_plot[labels == -1] = "noise"

    gray_out_noise = {"noise": "lightgray"}

    fig = px.scatter(
        x=embedding[:, 0],
        y=embedding[:, 1],
        color=labels_plot,
        color_discrete_map=gray_out_noise,
        hover_data={
            "name": id_data["name"],
            "artists": id_data["artists"],
            "popularity": y
        }
    )

    fig.update_layout(
        title=f"UMAP with n_neighbors={n_neighbors} and min_dist={min_dist}",
        xaxis_title="UMAP Dimension 1",
        yaxis_title="UMAP Dimension 2",
        legend_title="Cluster"
    )

    return embedding,fig

# evaluates trustworthiness and returns value
# to make more efficient, takes random sample of n_samples
def eval_trustworthiness(X, X_embedded, n_neighbors, n_samples=25000, random_state=42):
    rng = np.random.default_rng(random_state)
    idx = rng.choice(len(X), size=min(n_samples, len(X)), replace=False)

    X_sub = X[idx]
    X_emb_sub = X_embedded[idx]

    return trustworthiness(X_sub, X_emb_sub, n_neighbors=n_neighbors)