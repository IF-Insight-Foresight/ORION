import os
import argparse
import pickle
import pandas as pd
import nltk
import re
import numpy as np
from sentence_transformers import SentenceTransformer
import umap
from sklearn.neighbors import NearestNeighbors
import networkx as nx
import community as community_louvain
from keybert import KeyBERT
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

# ---------------------
# Excel input handling
# ---------------------
# The path to the Excel file can be provided either via the command
# line argument ``--excel`` or the environment variable
# ``ORION_EXCEL_PATH``.  If neither is supplied, the script looks for a
# file named ``ORION_Scanning_DB_Updated.xlsx`` relative to this
# repository.

parser = argparse.ArgumentParser(description="Preprocess ORION data")
parser.add_argument(
    "--excel",
    help="Path to the Excel file containing raw data",
)
args = parser.parse_args()

excel_path = (
    args.excel
    or os.environ.get("ORION_EXCEL_PATH")
    or "ORION_Scanning_DB_Updated.xlsx"
)

# 1) Load your raw data from Excel
data = pd.read_excel(excel_path, engine="openpyxl")

# --- Handle “Data Created” column ------------------------------
if "Data Created" in data.columns:
    data["Data Created"] = pd.to_datetime(data["Data Created"], errors="coerce")
    data["DaysSinceCreated"] = (
        pd.Timestamp.now() - data["Data Created"]
    ).dt.days.fillna(-1).astype(int)
    data["YearCreated"] = data["Data Created"].dt.year.fillna(-1).astype(int)
else:
    data["DaysSinceCreated"] = -1
    data["YearCreated"] = -1

# 2) Prepare text preprocessing
nltk.download("stopwords", quiet=True)
# Build the set of stopwords
text_stopwords = set(stopwords.words("english"))
# Convert to list for scikit‑learn
text_stopwords = list(text_stopwords)

def preprocess_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    tokens = text.split()
    tokens = [w for w in tokens if w not in text_stopwords and len(w) > 2]
    return " ".join(tokens)

# 3) Build the “PreprocessedText” column exactly as in orion2.py
data["PreprocessedText"] = (
    data["Title"].fillna("") + " " +
    data["Description"].fillna("") + " " +
    data["Tags"].fillna("")
).apply(preprocess_text)

# 4) Sentence‑Transformer embeddings
embedder = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedder.encode(
    data["PreprocessedText"].tolist(),
    show_progress_bar=True
)

# 5) UMAP to 3‑D coordinates (layout for Dash)
umap_3d = umap.UMAP(n_components=3, metric="cosine", random_state=42)
coords_3d = umap_3d.fit_transform(embeddings)

print("UMAP 3‑D coords min/max:")
print("x:", np.min(coords_3d[:, 0]), np.max(coords_3d[:, 0]))
print("y:", np.min(coords_3d[:, 1]), np.max(coords_3d[:, 1]))
print("z:", np.min(coords_3d[:, 2]), np.max(coords_3d[:, 2]))

# 6) Louvain community detection on a k-NN graph
knn = NearestNeighbors(n_neighbors=15, metric="cosine").fit(embeddings)
knn_graph = knn.kneighbors_graph(mode="connectivity")

# NetworkX ≥ 3.0 renamed the helper; fall back for older versions
if hasattr(nx, "from_scipy_sparse_array"):
    G = nx.from_scipy_sparse_array(knn_graph)
else:  # NetworkX ≤ 2.8
    G = nx.from_scipy_sparse_matrix(knn_graph)

partition = community_louvain.best_partition(G, resolution=1.2, random_state=42)
cluster_labels = np.array([partition[i] for i in range(len(data))])

# 7) Generate human‑readable cluster titles with KeyBERT
kw_model = KeyBERT(model=embedder)
cluster_titles = {}
for cid in np.unique(cluster_labels):
    docs = data.loc[cluster_labels == cid, "PreprocessedText"].tolist()
    joined = " ".join(docs[:10000])  # keep it lightweight
    keywords = kw_model.extract_keywords(joined, top_n=2, stop_words=text_stopwords)
    if keywords:
        cluster_titles[cid] = " & ".join([kw for kw, _ in keywords])
    else:
        cluster_titles[cid] = f"Cluster {cid}"

# 10) Compute silhouette (optional, but you can store it)
sil_scores = silhouette_score(embeddings, cluster_labels, metric="cosine")

# 11) Dump everything to pickle
features = {
    "tsne_x": coords_3d[:, 0].astype(float).tolist(),
    "tsne_y": coords_3d[:, 1].astype(float).tolist(),
    "tsne_z": coords_3d[:, 2].astype(float).tolist(),  # Added Z
    "cluster_labels": cluster_labels.tolist(),
    "cluster_titles": cluster_titles,
    "days_since_created": data["DaysSinceCreated"].tolist(),
    "year_created": data["YearCreated"].tolist(),
    "silhouette_score": float(silhouette_score(embeddings, cluster_labels, metric="cosine"))
}

with open("precomputed_features.pkl", "wb") as f:
    pickle.dump(features, f)

print("✅ Precomputed text‑based features saved to precomputed_features.pkl")