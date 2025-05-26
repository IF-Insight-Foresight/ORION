# preprocess.py
import pickle
import pandas as pd
import nltk
import re
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

# 1) Load your raw data from Excel
data = pd.read_excel(
    "/Users/soeir2/Desktop/ORION_Scanning_DB_Updated.xlsx",
    engine="openpyxl"
)

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

# 4) TF‑IDF vectorization on that cleaned text
vectorizer = TfidfVectorizer(
    max_features=2000,
    ngram_range=(1, 2),
    stop_words=text_stopwords
)
X_tfidf = vectorizer.fit_transform(data["PreprocessedText"])

# 5) PCA to 50 dimensions (or fewer if your TF‑IDF vocab is smaller)
n_features = X_tfidf.shape[1]
n_pca = min(50, n_features)
pca = PCA(n_components=n_pca, random_state=42)
pca_coords = pca.fit_transform(X_tfidf.toarray())

# 6) t‑SNE down to 3D for 3D network layout
tsne = TSNE(
    n_components=3,  # Now 3D
    init="random",
    random_state=42,
    learning_rate="auto"
)
tsne_coords = tsne.fit_transform(pca_coords)

print("t-SNE 3D coords min/max:")
print("x:", np.min(tsne_coords[:,0]), np.max(tsne_coords[:,0]))
print("y:", np.min(tsne_coords[:,1]), np.max(tsne_coords[:,1]))
print("z:", np.min(tsne_coords[:,2]), np.max(tsne_coords[:,2]))

# 7) K‑Means + silhouette search for optimal k among {5,10,…,45}
silhouette_scores = []
K_candidates = list(range(5, 46, 5))
for k in K_candidates:
    km_test = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_test = km_test.fit_predict(pca_coords)
    silhouette_scores.append(silhouette_score(pca_coords, labels_test))

optimal_k = K_candidates[int(np.argmax(silhouette_scores))]
print(f"→ optimal_k = {optimal_k}")

# 8) Final K‑Means with that optimal_k
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(pca_coords)

# 9) Build short labels from top TF‑IDF terms in each cluster
feature_names = vectorizer.get_feature_names_out()
cluster_titles = {}
for cid in range(optimal_k):
    idxs = [i for i, lbl in enumerate(cluster_labels) if lbl == cid]
    if not idxs:
        cluster_titles[cid] = f"Cluster {cid}"
        continue
    mat = X_tfidf[idxs]
    scores = np.asarray(mat.sum(axis=0)).flatten()
    top5 = np.argsort(scores)[::-1][:5]
    terms = [feature_names[i] for i in top5 if scores[i] > 0]
    cluster_titles[cid] = " & ".join(terms[:2]) if terms else f"Cluster {cid}"
    
# 10) Compute silhouette (optional, but you can store it)
sil_scores = silhouette_score(pca_coords, cluster_labels, metric="euclidean")

# 11) Dump everything to pickle
features = {
    "tsne_x": tsne_coords[:, 0].astype(float).tolist(),
    "tsne_y": tsne_coords[:, 1].astype(float).tolist(),
    "tsne_z": tsne_coords[:, 2].astype(float).tolist(),  # Added Z
    "cluster_labels": cluster_labels.tolist(),
    "cluster_titles": cluster_titles,
    "silhouette_score": float(silhouette_score(pca_coords, cluster_labels))
}

with open("precomputed_features.pkl", "wb") as f:
    pickle.dump(features, f)

print("✅ Precomputed text‑based features saved to precomputed_features.pkl")