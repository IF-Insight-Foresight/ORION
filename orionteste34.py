import re

def rgb_to_hex(rgb_str):
    match = re.match(r"rgb\((\d+),\s*(\d+),\s*(\d+)\)", rgb_str)
    if match:
        r, g, b = map(int, match.groups())
        return '#{:02x}{:02x}{:02x}'.format(r, g, b)
    else:
        return rgb_str
import os
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv
from pathlib import Path
import base64, mimetypes
import uuid
import pickle
import io
import sys
import logging

# Initialize logging before first use
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import re
import json
from datetime import datetime

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd

from dash import dcc, html, dash_table, Input, Output, State, ALL
from dash import ctx
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import plotly.graph_objects as go

from flask import Flask, session, redirect, request

from dash.dcc import send_bytes

import networkx as nx
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist, squareform
from functools import lru_cache
from plotly.colors import hex_to_rgb, qualitative
import nltk
from nltk.corpus import stopwords



# --- PROJECTS/DASHBOARD STATE PERSISTENCE LOGIC ---
import threading
PROJECTS_FILE = "projects.json"
projects_lock = threading.Lock()
def default_project_state():
    return {
        "chips": [],
        "logic": "AND",
        "search": "",
        "cluster": -1,
        "driving_forces": ["(All)"],
        "explorer_selected_rows": [],
        "selected_nodes": []
    }
def _gather_current_state(search, chips, cluster,
                          driving_forces, explorer_rows):
    return {
        "search":                search or "",
        "chips":                 chips or [],
        "cluster":               cluster,
        "driving_forces":        driving_forces,
        "explorer_selected_rows": explorer_rows or [],
    }
def load_projects():
    try:
        with projects_lock, open(PROJECTS_FILE, 'r') as f:
            raw = json.load(f)
            # Ensure all default keys for every project
            for v in raw.values():
                for k, default_val in default_project_state().items():
                    if k not in v:
                        v[k] = default_val
            return raw
    except Exception:
        return {"Default Project": default_project_state()}

def save_projects(data):
    with projects_lock, open(PROJECTS_FILE, 'w') as f:
        json.dump(data, f, indent=2)

# --- Global variable holding all projects state ---
projects = load_projects()


from dash.exceptions import PreventUpdate
APPLE_FONT = "'SF Pro Display', 'Helvetica Neue', Arial, sans-serif"
APPLE_BG = "#000"
APPLE_PANEL = "#111"
APPLE_ACCENT = "#222"
APPLE_RADIUS = "18px"
APPLE_SHADOW = "0 4px 24px rgba(0,0,0,0.18)"
APPLE_BUTTON = {
    "backgroundColor": "#007aff",
    "color": "#fff",
    "border": "none",
    "borderRadius": APPLE_RADIUS,
    "fontWeight": "600",
    "fontFamily": APPLE_FONT,
    "fontSize": "16px",
    "padding": "10px 0",
    "boxShadow": "0 2px 6px rgba(0,0,0,0.14)",
    "transition": "background 0.2s"
}
APPLE_BUTTON_SECONDARY = {
    **APPLE_BUTTON,
    "backgroundColor": "#333",
    "color": "#fff"
}
APPLE_INPUT = {
    "backgroundColor": "#1a1a1a",
    "color": "#fff",
    "border": "1px solid #333",
    "borderRadius": APPLE_RADIUS,
    "fontFamily": APPLE_FONT,
    "fontSize": "15px",
    "padding": "8px"
}
APPLE_CARD = {
    "backgroundColor": APPLE_PANEL,
    "borderRadius": APPLE_RADIUS,
    "boxShadow": APPLE_SHADOW,
    "padding": "24px"
}

# --- ENV & SECRETS ---
load_dotenv(Path.home() / "Desktop" / ".env", override=True)
api_key      = os.getenv("OPENAI_API_KEY")
assistant_id = os.getenv("ASSISTANT_ID") or os.getenv("OPENAI_ASSISTANT_ID")

# Scanning-page chatbot can use a different assistant; falls back if unset
scanning_assistant_id = os.getenv("SCANNING_ASSISTANT_ID") or assistant_id

if not api_key:
    raise ValueError("🚨 Missing OPENAI_API_KEY in .env file.")
if not assistant_id:
    raise ValueError("🚨 Missing ASSISTANT_ID in .env file.")
import openai
from openai import OpenAI
openai.api_key = api_key
client = OpenAI(api_key=api_key)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- DASH APP INIT ---
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True
)
server = app.server

# --- Custom CSS for Copilot FAB hover effect ---
app.index_string += """
<style>
#show-scanning-copilot-btn:hover {
    background: rgba(38, 90, 220, 0.18) !important;
    box-shadow: 0 8px 48px #007aff66;
    transform: scale(1.08);
}
#show-scanning-copilot-btn:active {
    background: rgba(38, 90, 220, 0.28) !important;
    box-shadow: 0 8px 32px #007aff99;
    transform: scale(0.96);
}
</style>
"""



from flask_sqlalchemy import SQLAlchemy

server.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///orion_users.db'
server.config['SECRET_KEY'] = 'orion_secret_key_123'
db = SQLAlchemy(server)

# --- Flask-Login initialization and User model ---
from flask_login import LoginManager, UserMixin, login_user

login_manager = LoginManager()
login_manager.init_app(server)

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(128), nullable=False)

    def set_password(self, password):
        from werkzeug.security import generate_password_hash
        self.password = generate_password_hash(password)

    def check_password(self, password):
        from werkzeug.security import check_password_hash
        return check_password_hash(self.password, password)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


app.server.secret_key = 'orion_secret_key_123'

# --- DATA ---
nltk.download('stopwords')
nltk_stopwords = set(stopwords.words("english"))
custom_stopwords = {
    "analysis", "approach", "use", "based", "important", "development", "solution",
    "research", "year", "2024", "2025", "info", "search", "getty", "image",
    "study", "new", "company", "report", "researchers", "mintel", "women", "men",
    "man", "woman", "says", "said", "economic", "also", "like", "court", "users",
    "one", "first", "says", "say",
}
stop_words = nltk_stopwords.union(custom_stopwords)
combined_stopwords = list(nltk_stopwords.union(custom_stopwords))

workspace_dir = "workspaces"
os.makedirs(workspace_dir, exist_ok=True)
file_path = "ORION_Scanning_DB_Updated.xlsx"
try:
    data = pd.read_excel(file_path)
    data.columns = data.columns.str.strip()
    logger.info(f"Data loaded from {file_path}")
except Exception as e:
    logger.error(f"Error loading data: {e}")
    sys.exit("Error loading data")
required_columns = [
    "ID", "Title", "Description", "Driving Force",
    "Source", "Tags", "URLs"
]
for col in required_columns:
    if col not in data.columns:
        logger.error(f"Column missing: {col}")
        sys.exit(f"Column {col} missing")
def sanitize_color(c):
    if pd.isnull(c) or not isinstance(c, str) or not c.startswith("#") or len(c) < 7:
        return "#AAAAAA"
    return c
data["Color"] = data["Color"].apply(sanitize_color)
data.loc[data["Driving Force"].str.lower() == "signals", "Color"] = "#333"
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words and len(w) > 2]
    return " ".join(tokens)
def match_advanced_query(row_text, query):
    if not query or not query.strip():
        return False
    row_text_lower = row_text.lower()
    def contains_token(token):
        pattern = rf"\b{re.escape(token.lower())}\b"
        return bool(re.search(pattern, row_text_lower))
    phrase_pattern = r'"([^"]+)"'
    phrases_found = re.findall(phrase_pattern, query)
    placeholder_dict = {}
    def placeholder_name(i): return f"__PHRASE_{i}__"
    temp_query = query
    for i, ph in enumerate(phrases_found):
        p_name = placeholder_name(i)
        temp_query = temp_query.replace(f'"{ph}"', p_name)
        placeholder_dict[p_name] = ph
    or_clauses = re.split(r'\s+OR\s+', temp_query, flags=re.IGNORECASE)
    for or_clause in or_clauses:
        and_parts = re.split(r'\s+AND\s+', or_clause, flags=re.IGNORECASE)
        all_and_ok = True
        for part in and_parts:
            part = part.strip()
            if not part:
                continue
            if part in placeholder_dict:
                phrase_text = placeholder_dict[part].lower()
                if phrase_text not in row_text_lower:
                    all_and_ok = False
                    break
            else:
                if not contains_token(part):
                    all_and_ok = False
                    break
        if all_and_ok:
            return True
    return False
def custom_tokenizer(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    tokens = text.split()
    tokens = [token for token in tokens if len(token) > 2]
    return tokens
count_vectorizer = CountVectorizer(tokenizer=custom_tokenizer, stop_words=stop_words)
def get_cluster_label_from_docs(cluster_id, data, X_tfidf, vectorizer, top_n=5):
    indices = data.index[data["Cluster"] == cluster_id].tolist()
    if not indices:
        return f"Cluster {cluster_id} (no docs)"
    cluster_matrix = X_tfidf[indices]
    summed_tfidf = cluster_matrix.sum(axis=0)
    summed_tfidf = np.asarray(summed_tfidf).flatten()
    feature_names = vectorizer.get_feature_names_out()
    sorted_indices = np.argsort(summed_tfidf)[::-1][:top_n]
    top_terms = [feature_names[i] for i in sorted_indices if summed_tfidf[i] > 0]
    if top_terms:
        return " & ".join(top_terms[:2])
    return f"Cluster {cluster_id}"
def hex_to_rgba_custom(hex_color, alpha=0.7):
    try:
        r, g, b = hex_to_rgb(hex_color)
        return f'rgba({r},{g},{b},{alpha})'
    except ValueError:
        print(f"Invalid hex color encountered: {hex_color}. Using default light grey.")
        return f'rgba(200,200,200,{alpha})'
data["PreprocessedText"] = (
    data["Title"].fillna("") + " " +
    data["Description"].fillna("") + " " +
    data["Tags"].fillna("")
).apply(preprocess_text)
if 'PreprocessedText' not in data.columns:
    print("🚨 Warning: 'PreprocessedText' column missing! Creating it now...")
    data["PreprocessedText"] = (
        data["Title"].fillna("") + " " +
        data["Description"].fillna("") + " " +
        data["Tags"].fillna("")
    ).apply(preprocess_text)
data = data[data["PreprocessedText"].str.strip() != ""].reset_index(drop=True)
data = data.head(32000)

#
# --- Search suggestions extraction ---
# (Patch: Use relevant, short prompts for Copilot suggestions)
search_suggestions = [
    "Summarize the main drivers behind these results",
    "Suggest 3 opportunity spaces reflected by the current selection"
]
features_path = os.getenv("ORION_PRECOMPUTED_PATH", "precomputed_features.pkl")
if not os.path.exists(features_path):
    raise FileNotFoundError(f"Precomputed features not found at {features_path}. Run preprocessteste.py first.")
with open(features_path, "rb") as f:
    feats = pickle.load(f)
data["tsne_x"]     = feats["tsne_x"]
data["tsne_y"]     = feats["tsne_y"]
data["Cluster"]    = feats["cluster_labels"]
data["silhouette"] = feats.get("silhouette_scores", None)

# Properly load tsne_z if present in feats, otherwise fallback to synthetic
from sklearn.preprocessing import StandardScaler
if "tsne_z" in feats:
    data["tsne_z"] = feats["tsne_z"]
elif "tsne_z" not in data.columns:
    scaler = StandardScaler()
    data["tsne_z"] = scaler.fit_transform(data[["tsne_x"]])[:,0] * 2.2  # Only as fallback if no true 3D
cluster_ids = sorted(data["Cluster"].unique().tolist())
color_map = {
    cid: rgb_to_hex(qualitative.Safe[i % len(qualitative.Safe)])
    for i, cid in enumerate(cluster_ids)
}
vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1, 2))
X_tfidf = vectorizer.fit_transform(data["PreprocessedText"])
cluster_titles = {
    cid: f"Cluster {cid}: " + get_cluster_label_from_docs(cid, data, X_tfidf, vectorizer)
    for cid in cluster_ids
}
@lru_cache(maxsize=256)
def semantic_search(query: str, top_n: int = 100):
    if not query or not query.strip():
        return np.arange(len(data)), np.ones(len(data))
    query_vec = vectorizer.transform([preprocess_text(query)])
    sims = cosine_similarity(query_vec, X_tfidf).flatten()
    top_idx = np.argsort(sims)[::-1][:top_n]
    return top_idx, sims[top_idx]

# --- VISUALIZATION FUNCTIONS ---
def generate_tsne_plot(filtered_data, show_edges=False, edge_threshold=2.0, dimension='2d', show_titles=False):
    # Defensive: if filtered_data is empty, return empty fig
    if filtered_data is None or filtered_data.empty:
        logger.info("generate_tsne_plot: filtered_data is empty or None, returning empty Figure.")
        return go.Figure()
    colors = []
    border_colors = []
    opacities = []
    sizes = []
    border_widths = []
    for i, (_, row) in enumerate(filtered_data.iterrows()):
        df_value = str(row["Driving Force"]).strip().lower()
        try:
            cluster_id = int(row["Cluster"])
        except Exception:
            cluster_id = -999  # fallback/default
        if df_value == "signals":
            base_color = "rgba(55,55,55,0.92)"
        else:
            base_color = hex_to_rgba_custom(color_map.get(cluster_id, "#ccc"), 0.89)
        # Debug print for node color assignment
        print(f"Node {row['ID']} | DF: {df_value} | Cluster: {cluster_id} | Color: {base_color}")
        border_color = "rgba(100,100,100,0.96)" if df_value == "signals" else base_color
        border_width = 0.85 if df_value == "signals" else 1.7
        colors.append(base_color)
        border_colors.append(border_color)
        opacities.append(0.92)
        sizes.append(9)
        border_widths.append(border_width)
    # Insert debug prints for unique values and first 10 colors
    print("Unique Driving Force values:", set(filtered_data["Driving Force"].str.lower()))
    print("Unique Cluster values:", set(filtered_data["Cluster"]))
    print("First 10 colors:", colors[:10])
    fig = go.Figure()
    # Edges (connections) go first, behind the nodes
    if show_edges:
        pts = filtered_data[["tsne_x", "tsne_y"]].to_numpy() if len(filtered_data) > 0 else np.empty((0,2))
        if len(pts) > 1:
            from scipy.spatial.distance import pdist, squareform
            D = squareform(pdist(pts))
            edge_x, edge_y = [], []
            for i in range(len(pts)):
                for j in range(i + 1, len(pts)):
                    if D[i, j] <= edge_threshold:
                        edge_x += [pts[i,0], pts[j,0], None]
                        edge_y += [pts[i,1], pts[j,1], None]
            if edge_x:
                fig.add_trace(go.Scatter(
                    x=edge_x, y=edge_y,
                    mode="lines",
                    line=dict(width=0.3, color="rgba(120,120,120,0.35)"),
                    line_shape="spline",
                    hoverinfo="none",
                    showlegend=False
                ))
    if dimension == '2d':
        # Simulated node glow: add invisible slightly larger shadow marker trace below nodes (static, no highlight)
        # For halos, use same color array but with lower alpha for clusters, lighter for signals
        shadow_colors = []
        for i, (_, row) in enumerate(filtered_data.iterrows()):
            df_value = str(row["Driving Force"]).strip().lower()
            try:
                cluster_id = int(row["Cluster"])
            except Exception:
                cluster_id = -999
            if df_value == "signals":
                shadow_colors.append("rgba(120,120,120,0.17)")
            else:
                # Use cluster color with lower alpha
                shadow_colors.append(hex_to_rgba_custom(color_map.get(cluster_id, "#ccc"), 0.02))
        shadow_sizes = [sz + 2 for sz in sizes]
        shadow_x = filtered_data["tsne_x"].tolist() if len(filtered_data) > 0 else []
        shadow_y = filtered_data["tsne_y"].tolist() if len(filtered_data) > 0 else []
        fig.add_trace(
            go.Scatter(
                x=shadow_x,
                y=shadow_y,
                mode="markers",
                marker=dict(
                    size=shadow_sizes,
                    color=shadow_colors,
                    opacity=1.0,  # fully opaque, color alpha controls the halo
                    line=dict(width=0, color="rgba(0,0,0,0)"),
                ),
                hoverinfo="skip",
                showlegend=False
            )
        )
    # Main nodes
    try:
        if dimension == '3d':
            # --- PATCHED LOGIC FOR 3D CURATED/SIGNALS EXTRACTION AND PLOTTING ---
            curated_mask = filtered_data["Driving Force"].str.lower().isin(["megatrends", "trends", "weak signals", "wildcards"])
            signals_mask = filtered_data["Driving Force"].str.lower() == "signals"

            coords = filtered_data[["tsne_x", "tsne_y", "tsne_z"]].to_numpy() if len(filtered_data) > 0 else np.empty((0,3))
            if len(coords) == 0:
                # PATCH: add dummy point to force 3D scene
                coords = np.array([[0, 0, 0]])
                curated_x = [0]
                curated_y = [0]
                curated_z = [0]
                signals_x = []
                signals_y = []
                signals_z = []
                curated_colors = ["#bbb"]
                curated_sizes = [9]
                signals_colors = []
                signals_sizes = []
                curated_halo_colors = ["rgba(200,200,200,0.01)"]
                signals_halo_colors = []
                curated_halo_sizes = [13]
                signals_halo_sizes = []
            else:
                for i in range(3):
                    minv, maxv = coords[:, i].min(), coords[:, i].max()
                    if maxv > minv:
                        coords[:, i] = (coords[:, i] - minv) / (maxv - minv)
                coords = (coords * 0.99 + 0.005)*2.5

                curated_x = coords[curated_mask.values, 0] if np.sum(curated_mask.values) > 0 else []
                curated_y = coords[curated_mask.values, 1] if np.sum(curated_mask.values) > 0 else []
                curated_z = coords[curated_mask.values, 2] if np.sum(curated_mask.values) > 0 else []
                signals_x = coords[signals_mask.values, 0] if np.sum(signals_mask.values) > 0 else []
                signals_y = coords[signals_mask.values, 1] if np.sum(signals_mask.values) > 0 else []
                signals_z = coords[signals_mask.values, 2] if np.sum(signals_mask.values) > 0 else []

                # For curated nodes, ensure fully opaque color
                curated_colors = []
                if np.sum(curated_mask.values) > 0:
                    for idx, (_, row) in enumerate(filtered_data[curated_mask].iterrows()):
                        try:
                            cluster_id = int(row["Cluster"])
                        except Exception:
                            cluster_id = -999
                        curated_colors.append(hex_to_rgba_custom(color_map.get(cluster_id, "#ccc"), 1.0))
                signals_colors = ["rgba(120,120,120,0.16)"] * int(np.sum(signals_mask.values))
                curated_sizes = np.array(sizes)[curated_mask.values] if np.sum(curated_mask.values) > 0 else []
                signals_sizes = np.array(sizes)[signals_mask.values] if np.sum(signals_mask.values) > 0 else []

                # Halos for 3D: for each node, use cluster color with alpha=0.01 for clusters, and "rgba(120,120,120,0.01)" for signals
                # For curated
                curated_halo_colors = []
                curated_halo_sizes = []
                for idx, (_, row) in enumerate(filtered_data[curated_mask].iterrows()):
                    df_value = str(row["Driving Force"]).strip().lower()
                    try:
                        cluster_id = int(row["Cluster"])
                    except Exception:
                        cluster_id = -999
                    curated_halo_colors.append(hex_to_rgba_custom(color_map.get(cluster_id, "#ccc"), 0.01))
                    curated_halo_sizes.append(9 + 4)
                # For signals
                signals_halo_colors = ["rgba(120,120,120,0.01)"] * int(np.sum(signals_mask.values))
                signals_halo_sizes = [9 + 4] * int(np.sum(signals_mask.values))

            # --- Edges (connections) only if requested ---
            if show_edges:
                edge_x, edge_y, edge_z = [], [], []
                from scipy.spatial.distance import pdist, squareform
                D = squareform(pdist(coords))
                threshold = 0.11  # tune for density
                for i in range(len(coords)):
                    for j in range(i+1, len(coords)):
                        if D[i, j] < threshold:
                            edge_x += [coords[i,0], coords[j,0], None]
                            edge_y += [coords[i,1], coords[j,1], None]
                            edge_z += [coords[i,2], coords[j,2], None]
                if edge_x:
                    fig.add_trace(go.Scatter3d(
                        x=edge_x, y=edge_y, z=edge_z,
                        mode="lines",
                        line=dict(width=0.25, color="rgba(120,120,120,0.14)"),
                        hoverinfo="none",
                        showlegend=False
                    ))

            # Add trace for signals (even if empty, so scene is always 3D)
            fig.add_trace(
                go.Scatter3d(
                    x=signals_x if len(signals_x) > 0 else [],
                    y=signals_y if len(signals_y) > 0 else [],
                    z=signals_z if len(signals_z) > 0 else [],
                    mode="markers",
                    marker=dict(
                        size=signals_sizes if len(signals_sizes) > 0 else [],
                        color=signals_colors if len(signals_colors) > 0 else [],
                        opacity=0.13,
                        line=dict(
                            color=signals_colors if len(signals_colors) > 0 else [],
                            width=0.6
                        ),
                        symbol="circle"
                    ),
                    hoverinfo="skip",
                    showlegend=False
                )
            )
            # --- Halo for curated nodes ---
            fig.add_trace(
                go.Scatter3d(
                    x=curated_x,
                    y=curated_y,
                    z=curated_z,
                    mode="markers",
                    marker=dict(
                        size=curated_halo_sizes,
                        color=curated_halo_colors,
                        opacity=1.0,  # color alpha controls the halo
                        line=dict(width=0),
                        symbol="circle"
                    ),
                    hoverinfo="skip",
                    showlegend=False
                )
            )
            # --- Halo for signals nodes ---
            fig.add_trace(
                go.Scatter3d(
                    x=signals_x if len(signals_x) > 0 else [],
                    y=signals_y if len(signals_y) > 0 else [],
                    z=signals_z if len(signals_z) > 0 else [],
                    mode="markers",
                    marker=dict(
                        size=signals_halo_sizes if len(signals_halo_sizes) > 0 else [],
                        color=signals_halo_colors if len(signals_halo_colors) > 0 else [],
                        opacity=1.0,
                        line=dict(width=0),
                        symbol="circle"
                    ),
                    hoverinfo="skip",
                    showlegend=False
                )
            )
            # Add trace for curated
            # --- PATCH: Guard np.stack for customdata ---
            if np.sum(curated_mask.values) > 0:
                try:
                    curated_customdata = np.stack([
                        filtered_data[curated_mask]["ID"].fillna("N/A"),
                        filtered_data[curated_mask]["Title"].fillna("N/A"),
                        filtered_data[curated_mask]["Driving Force"].fillna("N/A"),
                        filtered_data[curated_mask]["Cluster"].fillna(-1),
                        filtered_data[curated_mask]["Source"].fillna("N/A")
                    ], axis=-1)
                except Exception as e:
                    logger.error(f"Error stacking curated customdata in 3D: {e}\n"
                                 f"Shapes: {[filtered_data[curated_mask][col].shape for col in ['ID','Title','Driving Force','Cluster','Source']]}")
                    curated_customdata = None
            else:
                curated_customdata = None
            fig.add_trace(
                go.Scatter3d(
                    x=curated_x if len(curated_x) > 0 else [],
                    y=curated_y if len(curated_y) > 0 else [],
                    z=curated_z if len(curated_z) > 0 else [],
                    mode="markers",
                    marker=dict(
                        size=curated_sizes if len(curated_sizes) > 0 else [],
                        color=curated_colors if len(curated_colors) > 0 else [],
                        opacity=0.95,
                        line=dict(
                            color=curated_colors if len(curated_colors) > 0 else [],
                            width=1.7
                        ),
                        symbol="circle"
                    ),
                    customdata=curated_customdata,
                    hovertemplate="<b>%{customdata[1]}</b><br>Type: %{customdata[2]}<br>Cluster: %{customdata[3]}<br><b>ID:</b> %{customdata[0]}<extra></extra>",
                    showlegend=False
                )
            )

            fig.update_layout(
                margin=dict(l=0, r=0, t=0, b=0),
                scene=dict(
                    bgcolor="#000000",          # ← black 3‑D background
                    xaxis=dict(
                        visible=False, showgrid=False, zeroline=False,
                        showticklabels=False
                    ),
                    yaxis=dict(
                        visible=False, showgrid=False, zeroline=False,
                        showticklabels=False
                    ),
                    zaxis=dict(
                        visible=False, showgrid=False, zeroline=False,
                        showticklabels=False
                    ),
                    aspectmode="cube",
                    aspectratio=dict(x=1.2, y=1.2, z=1.2),
                ),
                scene_camera=dict(eye=dict(x=1.6, y=1.6, z=1.6)),
                paper_bgcolor="#000000",
                plot_bgcolor="#000000",
                font=dict(color="white", family=APPLE_FONT),
                showlegend=False
            )
            return fig
        else:
            # --- PATCH: Guard np.stack for customdata ---
            if len(filtered_data) > 0:
                try:
                    customdata = np.stack([
                        filtered_data["ID"].fillna("N/A"),
                        filtered_data["Title"].fillna("N/A"),
                        filtered_data["Driving Force"].fillna("N/A"),
                        filtered_data["Cluster"].fillna(-1),
                        filtered_data["Source"].fillna("N/A")
                    ], axis=-1)
                except Exception as e:
                    logger.error(f"Error stacking customdata in 2D: {e}\n"
                                 f"Shapes: {[filtered_data[col].shape for col in ['ID','Title','Driving Force','Cluster','Source']]}")
                    customdata = None
            else:
                customdata = None
            # Defensive: ensure x/y arrays are always the correct length or empty if filtered_data is empty
            trace_x = filtered_data["tsne_x"].tolist() if len(filtered_data) > 0 else []
            trace_y = filtered_data["tsne_y"].tolist() if len(filtered_data) > 0 else []
            trace_sizes = sizes if len(filtered_data) > 0 else []
            trace_colors = colors if len(filtered_data) > 0 else []
            trace_opacities = opacities if len(filtered_data) > 0 else []
            trace_border_colors = border_colors if len(filtered_data) > 0 else []
            trace_border_widths = border_widths if len(filtered_data) > 0 else []
            trace_text = filtered_data["Title"] if (show_titles and len(filtered_data) > 0) else None
            fig.add_trace(
                go.Scatter(
                    x=trace_x,
                    y=trace_y,
                    mode="markers+text" if show_titles else "markers",
                    marker=dict(
                        size=trace_sizes,
                        color=trace_colors,
                        opacity=trace_opacities,
                        line=dict(
                            color=trace_border_colors,
                            width=trace_border_widths
                        ),
                        symbol="circle"
                    ),
                    text=trace_text,
                    customdata=customdata,
                    hovertemplate="<b>ID:</b> %{customdata[0]}<br>" +
                                  "<b>Title:</b> %{customdata[1]}<br>" +
                                  "<b>Type:</b> %{customdata[2]}<br>" +
                                  "<b>Cluster:</b> %{customdata[3]}<br>" +
                                  "<b>Source:</b> %{customdata[4]}<extra></extra>",
                    showlegend=False
                )
            )
            fig.update_layout(
                title="",
                xaxis=dict(visible=False, showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(visible=False, showgrid=False, zeroline=False, showticklabels=False),
                paper_bgcolor=APPLE_BG,
                plot_bgcolor=APPLE_BG,
                font=dict(color="white", family=APPLE_FONT),
                margin=dict(l=20, r=20, t=20, b=20),
                showlegend=False
            )
            return fig
    except Exception as exc:
        import traceback
        logger.error(f"Exception in generate_tsne_plot: {exc}\n{traceback.format_exc()}")
        return go.Figure()
    # (Remove unreachable legacy code block here)

def build_cooccurrence_graph(docs, max_terms=200, min_tfidf=0.05):
    tfidf_vectorizer = TfidfVectorizer(
        tokenizer=custom_tokenizer,
        stop_words=combined_stopwords,
        token_pattern=r"(?u)\b\w+\b"
    )
    X_tfidf = tfidf_vectorizer.fit_transform(docs)
    terms = tfidf_vectorizer.get_feature_names_out()
    tfidf_scores = np.array(X_tfidf.mean(axis=0)).flatten()
    important_indices = np.where(tfidf_scores > min_tfidf)[0]
    if len(important_indices) > max_terms:
        important_indices = important_indices[:max_terms]
    filtered_terms = [terms[i] for i in important_indices]
    X_tfidf_filtered = X_tfidf[:, important_indices]
    Xc = (X_tfidf_filtered.T * X_tfidf_filtered)
    Xc.setdiag(0)
    G = nx.Graph()
    for term in filtered_terms:
        G.add_node(term)
    for i in range(len(filtered_terms)):
        for j in range(i + 1, len(filtered_terms)):
            weight = Xc[i, j]
            if weight > 0:
                G.add_edge(filtered_terms[i], filtered_terms[j], weight=float(weight))
    return G

def generate_infranodus_style_graph(G):
    from networkx.algorithms import community
    centrality = nx.betweenness_centrality(G, weight='weight')
    communities = community.greedy_modularity_communities(G)
    node_community_map = {}
    for c_index, comm in enumerate(communities):
        for node in comm:
            node_community_map[node] = c_index
    pos = nx.spring_layout(G, seed=42)
    edge_x, edge_y = [], []
    for src, tgt in G.edges():
        x0, y0 = pos[src]
        x1, y1 = pos[tgt]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(f"{node} (Centrality={centrality[node]:.4f})")
        node_color.append(node_community_map[node])
        node_size.append(10 + centrality[node] * 40)
    # If node_color could be a color string, convert it with rgb_to_hex
    node_color = [rgb_to_hex(c) for c in node_color]
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            size=node_size,
            color=node_color,
            colorscale='Rainbow',
            showscale=False
        )
    )
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title="Word Co-occurrence Graph",
            showlegend=False,
            hovermode='closest',
            paper_bgcolor=APPLE_BG,
            plot_bgcolor=APPLE_BG,
            font=dict(color="#fff", family=APPLE_FONT),
            xaxis=dict(visible=False, showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(visible=False, showgrid=False, zeroline=False, showticklabels=False)
        )
    )
    return fig

def generate_word_cooccurrence_graph(filtered_data):
    docs = filtered_data["PreprocessedText"].tolist()
    G = build_cooccurrence_graph(docs)
    return generate_infranodus_style_graph(G)

# --- Apple-Inspired Navbar ---
def get_navbar(active_page=None):
    username = session.get('username', 'Guest')
    role = session.get('role', 'Unknown')
    nav_items = [
        dbc.NavItem(
            dbc.NavLink("Home", href="/", active=(active_page == "Home"), style={"color": "#fff"})
        ),
        dbc.NavItem(
            dbc.NavLink("Scanning", href="/scanning", active=(active_page == "Scanning"), style={"color": "#fff"})
        ),
        dbc.NavItem(
            dbc.NavLink("Copilot", href="/copilot", active=(active_page == "Copilot"), style={"color": "#fff"})
        ),
        dbc.NavItem(
            dbc.NavLink(f"{username} ({role})", href="#", disabled=True, style={"color": "#aaa"})
        ),
        dbc.NavItem(
            dbc.NavLink("Logout", href="/logout", style={"color": "#fff"})
        )
    ]
    return dbc.Navbar(
        dbc.Container([
            html.Div(
                style={"display": "flex", "alignItems": "center"},
                children=[
                    html.Img(
                        src="/static/ORION LOGO.png",
                        style={
                            "width": "40px",
                            "height": "auto",
                            "marginRight": "10px"
                        }
                    ),
                    dbc.NavbarBrand(
                        "ORION",
                        style={
                            "color": "white",
                            "fontFamily": APPLE_FONT,
                            "fontWeight": "700",
                            "fontSize": "22px"
                        }
                    )
                ]
            ),
            dbc.Nav(nav_items, className="ml-auto", style={
                "fontFamily": APPLE_FONT,
                "fontWeight": "500"
            })
        ], fluid=True),
        color="dark",
        dark=True,
        style={
            "borderBottom": "1px solid rgba(255,255,255,0.1)",
            "backgroundColor": APPLE_BG,
            "height": "60px",
            "boxShadow": "0 2px 8px rgba(0,0,0,0.15)"
        }
    )

#############################################
# HOME LAYOUT (Apple style, fullscreen)
#############################################
def home_layout():
    return html.Div(
        style={
            "width": "100vw",
            "height": "100vh",
            "backgroundColor": APPLE_BG,
            "display": "flex",
            "flexDirection": "column",
            "fontFamily": APPLE_FONT
        },
        children=[
            get_navbar("Home"),
            html.Div(
                style={
                    "flex": "1",
                    "display": "flex",
                    "flexDirection": "column",
                    "alignItems": "center",
                    "justifyContent": "center"
                },
                children=[
                    html.Img(
                        src="/static/ORION LOGO.png",
                        style={
                            "width": "150px",
                            "marginBottom": "32px",
                            "boxShadow": APPLE_SHADOW
                        }
                    ),
                    html.H1(
                        "Welcome to ORION",
                        style={
                            "fontSize": "62px",
                            "color": "#fff",
                            "fontFamily": APPLE_FONT,
                            "fontWeight": "700",
                            "letterSpacing": "0.7px",
                            "textAlign": "center",
                            "marginBottom": "10px"
                        }
                    ),
                    html.H3(
                        "Your Strategic Foresight and Innovation Platform",
                        style={
                            "fontSize": "22px",
                            "fontWeight": "400",
                            "color": "#eaeaea",
                            "marginBottom": "40px",
                            "maxWidth": "700px",
                            "textAlign": "center",
                            "fontFamily": APPLE_FONT
                        }
                    ),
                    dbc.Button(
                        "Get Started",
                        color="light",
                        style={
                            **APPLE_BUTTON,
                            "width": "200px",
                            "backgroundColor": "#000",
                            "border": "1.5px solid #444"
                        },
                        href="/scanning"
                    )
                ]
            )
        ]
    )

#############################################
# SCANNING LAYOUT (Apple style)
#############################################
def scanning_layout():
    # SCANNING page with Explorer Table in slide-in drawer overlay
    return html.Div(
        style={
            "width": "100vw",
            "height": "100vh",
            "backgroundColor": APPLE_BG,
            "display": "flex",
            "flexDirection": "column",
            "fontFamily": APPLE_FONT
        },
        children=[
            get_navbar("Scanning"),
            dcc.Store(id="info-panel-open", data=False),
            dcc.Store(id="explorer-selected-rows", data=[]),  # PATCH: store for explorer selected rows
            dcc.Store(id="scanning-copilot-open", data=False),
            dcc.Store(id="scanning-copilot-thread", data={}),
            dcc.Store(id="scanning-copilot-history", data=[]),
            dcc.Store(id="search-chips", data=[]),
            # In scanning_layout(), replace Copilot panel section with this block:

            html.Div(
                id="scanning-copilot-panel",
                style={
                    "display": "none",
                    "flexDirection": "column",
                    "position": "fixed",
                    "right": "110px",
                    "bottom": "30px",
                    "width": "420px",
                    "height": "600px",
                    "background": "#181818",
                    "boxShadow": "0 8px 48px #007aff44",
                    "borderRadius": "22px",
                    "zIndex": 2003,
                    "overflow": "hidden",
                    "transition": "all 0.25s cubic-bezier(.4,0,.2,1)"
                },
                children=[
                    html.Div("ORION Copilot", style={
                        "color": "#fff", "padding": "18px", "fontWeight": "bold", "fontSize": "19px"
                    }),
                    html.Div(
                        id="scanning-copilot-chatbox",
                        style={
                            "flex": "1 1 0%",                # ← grows inside the panel, fixed for scroll
                            "minHeight": 0,                  # ← ensures scroll area doesn’t push siblings
                            "overflowY": "auto",             # ← scrolls when long
                            "padding": "0 18px",
                            "display": "flex",
                            "flexDirection": "column",
                            "gap": "8px"
                        }
                    ),
                    html.Div(
                        id="scanning-copilot-suggestions",
                        children=[
                            *[
                                html.Button(
                                    suggestion,
                                    id={'type': 'copilot-suggestion', 'index': i},
                                    n_clicks=0,
                                    style={
                                        "background": "#292929",
                                        "color": "#fff",
                                        "border": "1px solid #444",
                                        "borderRadius": "10px",
                                        "padding": "7px 17px",
                                        "fontSize": "15px",
                                        "marginBottom": "6px",
                                        "cursor": "pointer"
                                    }
                                ) for i, suggestion in enumerate(search_suggestions[:8])
                            ]
                        ],
                        style={
                            "padding": "8px 18px 2px 18px",
                            "display": "flex",
                            "gap": "8px",
                            "flexWrap": "wrap",
                            "alignItems": "center"
                        }
                    ),
                    html.Div(
                        style={
                            "padding": "8px 18px 8px 18px",
                            "display": "flex",
                            "gap": "8px"
                        },
                        children=[
                            dcc.Input(
                                id="scanning-copilot-input",
                                type="text",
                                placeholder="Type your prompt…",
                                style={
                                    "flex": "1",
                                    "fontSize": "16px",
                                    "borderRadius": APPLE_RADIUS,
                                    "padding": "9px 15px",
                                    "background": "#232323",
                                    "color": "#fff",
                                    "border": "1.2px solid #333"
                                },
                                n_submit=0,
                                debounce=False,
                            ),
                            html.Button(
                                "Send",
                                id="scanning-copilot-send-btn",
                                n_clicks=0,
                                style={
                                    "background": "#007aff",
                                    "color": "#fff",
                                    "border": "none",
                                    "borderRadius": APPLE_RADIUS,
                                    "fontWeight": "600",
                                    "fontSize": "16px",
                                    "padding": "8px 20px",
                                    "marginLeft": "8px"
                                }
                            )
                        ]
                    ),
                ]
            ),
            html.Button(
                id="show-scanning-copilot-btn",
                n_clicks=0,
                title="Open Copilot",
                style={
                    "position": "fixed",
                    "bottom": "34px",
                    "right": "34px",
                    "width": "66px",
                    "height": "66px",
                    "borderRadius": "50%",
                    "backdropFilter": "blur(16px)",
                    "background": "rgba(30,30,30,0.7)",
                    "border": "1.5px solid #2c2c2c",
                    "boxShadow": "0 8px 32px rgba(0,122,255,0.28), 0 2px 8px #222a",
                    "display": "flex",
                    "alignItems": "center",
                    "justifyContent": "center",
                    "zIndex": 2004,
                    "cursor": "pointer",
                    "transition": "all 0.18s cubic-bezier(.4,0,.2,1)",
                    "outline": "none",
                    # Add this line:
                    "backgroundImage": "url(\"data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='38' height='38' viewBox='0 0 44 44' fill='none'><defs><radialGradient id='copilot-glow' cx='50%' cy='50%' r='60%' fx='50%' fy='50%'><stop offset='0%' stop-color='#7fdcff' stop-opacity='0.92'/><stop offset='100%' stop-color='#007aff' stop-opacity='0.82'/></radialGradient></defs><circle cx='22' cy='22' r='19' fill='url(%23copilot-glow)' stroke='#fff' stroke-width='2'/><rect x='14' y='20' width='16' height='3' rx='1.2' fill='#fff'/><rect x='14' y='26' width='9' height='3' rx='1.2' fill='#fff'/><circle cx='22' cy='15' r='2.1' fill='#fff'/></svg>\")",
                    "backgroundRepeat": "no-repeat",
                    "backgroundPosition": "center",
                    "backgroundSize": "38px 38px",
                },
                children=[]  # No children
            ),
            html.Div(
                style={
                    "flex": "1",
                    "display": "flex",
                    "flexDirection": "row",
                    "height": "calc(100vh - 60px)",
                    "overflow": "hidden",
                    "position": "relative"
                },
                children=[
                    # LEFT FILTER PANEL
                    html.Div(
                        style={
                            "width": "300px",
                            "backgroundColor": APPLE_PANEL,
                            "borderRight": "1px solid rgba(255,255,255,0.08)",
                            "boxShadow": "2px 0 6px rgba(0,0,0,0.13)",
                            "overflowY": "auto"
                        },
                        children=[
                            html.Div(
                                style={
                                    "padding": "18px 16px",
                                    "backgroundColor": APPLE_PANEL
                                },
                                children=[
                                    # --- Modern Keyword Search block ---
                                    html.Div(
                                        style={
                                            "backgroundColor": "#181818",
                                            "borderRadius": APPLE_RADIUS,
                                            "boxShadow": "0 2px 10px rgba(0,0,0,0.12)",
                                            "padding": "20px",
                                            "marginBottom": "32px"
                                        },
                                        children=[
                                            html.Div(
                                                style={
                                                    "display": "flex",
                                                    "alignItems": "center",
                                                    "justifyContent": "center",
                                                    "marginBottom": "10px"
                                                },
                                                children=[
                                                    html.H4(
                                                        "Keyword Search",
                                                        style={
                                                            "color": "#fff",
                                                            "fontWeight": "700",
                                                            "fontFamily": APPLE_FONT,
                                                            "margin": 0,
                                                            "flex": "1",
                                                            "fontSize": "20px",
                                                            "textAlign": "center",
                                                            "letterSpacing": "0.03em"
                                                        }
                                                    ),
                                                    html.Span(
                                                        "ⓘ",
                                                        id="search-info-icon",
                                                        style={
                                                            "marginLeft": "8px",
                                                            "fontSize": "20px",
                                                            "color": "#aaa",
                                                            "cursor": "pointer",
                                                            "userSelect": "none"
                                                        },
                                                        title="Type words, AND/OR, or phrases in quotes (e.g., AI AND \"future of work\"). Press Enter to search."
                                                    )
                                                ]
                                            ),
                                            dcc.Input(
                                                id='search-term',
                                                type='text',
                                                placeholder='Type keywords or phrases (e.g., AI AND "future of work")…',
                                                debounce=False,
                                                n_submit=0,
                                                style={
                                                    'width': '100%',
                                                    'fontSize': '19px',
                                                    'padding': '13px 18px',
                                                    'borderRadius': APPLE_RADIUS,
                                                    'marginBottom': '10px',
                                                    **APPLE_INPUT
                                                }
                                            ),
                                            dbc.Tooltip(
                                                "Use quotes for phrases, AND/OR for advanced queries. Press Enter to search.",
                                                target="search-info-icon",
                                                placement="right"
                                            ),
                                        ]
                                    ),
                                    # --- Cluster and Driving Force filters remain unchanged ---
                                    html.Label("Select Cluster:", style={"color": "#ccc", "marginTop": "20px"}),
                                    dcc.Dropdown(
                                        id='cluster-highlight',
                                        options=[{"label": "(None)", "value": -1}]
                                                + [{"label": cluster_titles[cid], "value": cid} for cid in cluster_ids],
                                        value=-1,
                                        style={
                                            "width": "100%",
                                            "backgroundColor": "#181818",
                                            "color": "#eee",
                                            "fontSize": "14px",
                                            "border": "1.5px solid #333",
                                            "fontFamily": APPLE_FONT,
                                            "boxShadow": "none",
                                            "borderRadius": APPLE_RADIUS,
                                        },
                                        clearable=False
                                    ),
                                    html.Label("Driving Force Filter:", style={"color": "#ccc", "marginTop": "20px"}),
                                    dcc.Dropdown(
                                        id='driving-force-filter',
                                        options=[{"label": "(All)", "value": "(All)"}] + [
                                            {"label": force, "value": force} 
                                            for force in ["Megatrends", "Trends", "Weak Signals", "Wildcards", "Signals"]
                                        ],
                                        value=["(All)"],
                                        multi=True,
                                        style={
                                            "width": "100%",
                                            "backgroundColor": "#181818",
                                            "color": "#fff",
                                            "fontSize": "14px",
                                            "border": "none",
                                            "fontFamily": APPLE_FONT,
                                            "boxShadow": "none",
                                            "borderRadius": APPLE_RADIUS,
                                        }
                                    ),
                                    # --- Connections: just a single checklist, no label ---
                                    html.Div(
                                        style={"marginTop": "20px", "marginBottom": "4px"},
                                        children=[
                                            dcc.Checklist(
                                                id='show-edges',
                                                options=[{'label': 'Show Connections', 'value': 'edges'}],
                                                value=[],
                                                inline=True,
                                                style={
                                                    "color": "#fff",
                                                    "fontFamily": APPLE_FONT,
                                                    "fontSize": "15px",
                                                    "marginTop": "2px",
                                                    "marginLeft": "4px"
                                                }
                                            )
                                        ]
                                    ),
                                    html.Div(
                                        style={"marginTop": "6px", "marginBottom": "4px"},
                                        children=[
                                            dcc.Checklist(
                                                id='show-node-titles',
                                                options=[{'label': 'Show Node Titles', 'value': 'titles'}],
                                                value=[],
                                                inline=True,
                                                style={
                                                    "color": "#fff",
                                                    "fontFamily": APPLE_FONT,
                                                    "fontSize": "15px",
                                                    "marginTop": "2px",
                                                    "marginLeft": "4px"
                                                }
                                            )
                                        ]
                                    ),
                                    html.Div(
                                        style={"display": "flex", "gap": "8px", "marginTop": "22px"},
                                        children=[
                                            dbc.Button(
                                                "Apply Filters",
                                                id="apply-filters-button",
                                                color="primary",
                                                style={**APPLE_BUTTON, "width": "50%"}
                                            ),
                                            dbc.Button(
                                                "Reset Filters",
                                                id="reset-filters-button",
                                                color="secondary",
                                                style={**APPLE_BUTTON_SECONDARY, "width": "50%"}
                                            ),
                                        ]
                                    ),
                                    html.Hr(),
                                    html.Div(
                                        style={
                                            "backgroundColor": "#181818",
                                            "borderRadius": APPLE_RADIUS,
                                            "boxShadow": "0 2px 10px rgba(0,0,0,0.12)",
                                            "padding": "20px",
                                            "marginBottom": "32px"
                                        },
                                        children=[
                                            html.H4(
                                                "Projects",
                                                style={
                                                    "color": "#fff",
                                                    "fontWeight": "700",
                                                    "fontFamily": APPLE_FONT,
                                                    "marginBottom": "18px",
                                                    "fontSize": "20px",
                                                    "textAlign": "center",
                                                    "letterSpacing": "0.03em"
                                                }
                                            ),
                                            dcc.Dropdown(
                                                id='project-selector',
                                                options=(
                                                    [{'label': html.Span([
                                                        html.I(className="fa fa-plus", style={"marginRight": "7px", "color": "#0af"}), "+ New Project…"
                                                    ]), 'value': "__new_project__"}] +
                                                    [{'label': html.Span([
                                                        html.Span("⭐", className="orion-project-star"), "Default Project"
                                                    ], className="orion-project-default"), 'value': "Default Project"}] +
                                                    [{'label': n, 'value': n} for n in projects if n != "Default Project"]
                                                ),
                                                value="Default Project",
                                                clearable=False,
                                                searchable=True,
                                                className="orion-project-dropdown",
                                                style={
                                                    "backgroundColor": "#222",
                                                    "color": "#eee",
                                                    "fontSize": "15px",
                                                    "border": "1.5px solid #333",
                                                    "fontFamily": APPLE_FONT,
                                                    "boxShadow": "none",
                                                    "borderRadius": APPLE_RADIUS,
                                                    "marginBottom": "14px"
                                                },
                                                optionHeight=38,
                                            ),
                                            html.Div(
                                                style={
                                                    "display": "flex",
                                                    "gap": "10px",
                                                    "justifyContent": "center"
                                                },
                                                children=[
                                                    html.Button(
                                                        html.I(className="fa fa-plus"),
                                                        id='create-project',
                                                        className="orion-icon-btn",
                                                        style={
                                                            **APPLE_BUTTON,
                                                            "padding": "7px 12px",
                                                            "fontSize": "18px",
                                                            "minWidth": "44px",
                                                            "minHeight": "44px"
                                                        },
                                                        title="Create Project",
                                                        tabIndex=0,
                                                        **{"aria-label": "Create Project"}
                                                    ),
                                                    html.Button(
                                                        html.I(className="fa fa-pencil-alt"),
                                                        id='rename-project',
                                                        className="orion-icon-btn",
                                                        style={
                                                            **APPLE_BUTTON_SECONDARY,
                                                            "padding": "7px 12px",
                                                            "fontSize": "18px",
                                                            "minWidth": "44px",
                                                            "minHeight": "44px"
                                                        },
                                                        title="Rename Project",
                                                        tabIndex=0,
                                                        **{"aria-label": "Rename Project"}
                                                    ),
                                                    html.Button(
                                                        html.I(className="fa fa-trash"),
                                                        id='delete-project',
                                                        className="orion-icon-btn",
                                                        style={
                                                            **APPLE_BUTTON_SECONDARY,
                                                            "padding": "7px 12px",
                                                            "fontSize": "18px",
                                                            "minWidth": "44px",
                                                            "minHeight": "44px"
                                                        },
                                                        title="Delete Project",
                                                        tabIndex=0,
                                                        **{"aria-label": "Delete Project"}
                                                    ),
                                                    html.Button(
                                                        html.I(className="fa fa-save"),
                                                        id='save-project',
                                                        className="orion-icon-btn",
                                                        style={
                                                            **APPLE_BUTTON_SECONDARY,
                                                            "padding": "7px 12px",
                                                            "fontSize": "18px",
                                                            "minWidth": "44px",
                                                            "minHeight": "44px"
                                                        },
                                                        title="Save Project",
                                                        tabIndex=0,
                                                        **{"aria-label": "Save Project"}
                                                    ),
                                                ]
                                            ),
                                            html.Div(
                                                id="new-project-input-row",
                                                style={"display": "flex", "alignItems": "center", "marginTop": "12px", "marginBottom": "0"},
                                                children=[
                                                    dcc.Input(
                                                        id='new-project-name',
                                                        type='text',
                                                        placeholder='Project name',
                                                        style={
                                                            "display": "none",
                                                            "width": "75%",
                                                            **APPLE_INPUT
                                                        }
                                                    ),
                                                    html.Button(
                                                        html.I(className="fa fa-check"),
                                                        id="confirm-new-project",
                                                        n_clicks=0,
                                                        className="orion-icon-btn",
                                                        style={
                                                            "display": "none",
                                                            "background": "none",
                                                            "border": "none",
                                                            "color": "#00d26a",
                                                            "fontSize": "22px",
                                                            "marginLeft": "8px",
                                                            "minWidth": "44px",
                                                            "minHeight": "44px"
                                                        },
                                                        title="Confirm New Project",
                                                        tabIndex=0,
                                                        **{"aria-label": "Confirm New Project"}
                                                    )
                                                ]
                                            ),
                                            html.Div(
                                                id="new-project-error",
                                                style={"color": "#ff4d4f", "fontSize": "14px", "marginTop": "4px", "display": "none"}
                                            ),
                                            html.Div(
                                                id="rename-project-input-row",
                                                style={"display": "flex", "alignItems": "center", "marginTop": "12px", "marginBottom": "0"},
                                                children=[
                                                    dcc.Input(
                                                        id='rename-project-name',
                                                        type='text',
                                                        placeholder='New name',
                                                        style={
                                                            "display": "none",
                                                            "width": "75%",
                                                            **APPLE_INPUT
                                                        }
                                                    ),
                                                    html.Button(
                                                        html.I(className="fa fa-check"),
                                                        id="confirm-rename-project",
                                                        n_clicks=0,
                                                        className="orion-icon-btn",
                                                        style={
                                                            "display": "none",
                                                            "background": "none",
                                                            "border": "none",
                                                            "color": "#00d26a",
                                                            "fontSize": "22px",
                                                            "marginLeft": "8px",
                                                            "minWidth": "44px",
                                                            "minHeight": "44px"
                                                        },
                                                        title="Confirm Rename",
                                                        tabIndex=0,
                                                        **{"aria-label": "Confirm Rename"}
                                                    )
                                                ]
                                            ),
                                            html.Div(
                                                id="rename-project-error",
                                                style={"color": "#ff4d4f", "fontSize": "14px", "marginTop": "4px", "display": "none"}
                                            ),
                                            dcc.ConfirmDialog(
                                                id='confirm-delete',
                                                message='Are you sure you want to delete this project?'
                                            ),
                                            html.Div(
                                                id='save-status',
                                                style={
                                                    'color': '#0af',
                                                    'fontSize': '14px',
                                                    'marginTop': '6px',
                                                    'textAlign': 'center'
                                                }
                                            )
                                        ]
                                    )
                                ]
                            )
                        ]
                    ),

                    # MIDDLE t-SNE/Word Graph PANEL
                    html.Div(
                        style={
                            "flex": "1",
                            "display": "flex",
                            "flexDirection": "column",
                            "backgroundColor": APPLE_BG,
                            "overflow": "hidden",
                            "position": "relative"
                        },
                        children=[
                            html.Div(
                                style={
                                    "flex": "1",
                                    "padding": "10px",
                                    "backgroundColor": APPLE_BG,
                                    "position": "relative"
                                },
                                children=[
                                    dcc.Loading(
                                        id="loading-tsne-plot",
                                        type="default",
                                        fullscreen=False,
                                        color="#aaa",
                                        style={"background": APPLE_BG},
                                        children=[
                                            html.Div(
                                                style={"position": "relative"},
                                                children=[
                                                    html.Div(
                                                        style={
                                                            "display": "flex",
                                                            "flexDirection": "row",
                                                            "alignItems": "center",
                                                            "gap": "20px",
                                                            "margin": "0 0 12px 0"
                                                        },
                                                        children=[
                                                            dcc.Checklist(
                                                                id="show-signals-toggle",
                                                                options=[{"label": " Show all signals (may slow browser)", "value": "show"}],
                                                                value=[],
                                                                style={"color": "#fff", "fontSize": "15px"}
                                                            ),
                                                            dcc.RadioItems(
                                                                id="network-dimension",
                                                                options=[
                                                                    {"label": "2D", "value": "2d"},
                                                                    {"label": "3D", "value": "3d"}
                                                                ],
                                                                value="3d",
                                                                inline=True,
                                                                style={
                                                                    "color": "#fff",
                                                                    "backgroundColor": "#181818",
                                                                    "borderRadius": APPLE_RADIUS,
                                                                    "fontFamily": APPLE_FONT,
                                                                    "fontWeight": "600"
                                                                }
                                                            ),
                                                        ]
                                                    ),
                                                    dcc.Graph(
                                                        id='tsne-plot',
                                                        style={
                                                            "width": "100vw",
                                                            "height": "98vh",
                                                            "backgroundColor": APPLE_BG,
                                                            "margin": "0",
                                                            "padding": "10px"
                                                        },
                                                        config={
                                                            "displayModeBar": False,
                                                            "displaylogo": False
                                                        }
                                                    ),
                                                    html.Div(
                                                        id="custom-loading-overlay",
                                                        style={
                                                            "display": "none",  # Will show only when loading
                                                            "position": "absolute", "zIndex": 1000,
                                                            "left": 0, "top": 0, "right": 0, "bottom": 0,
                                                            "background": "#000", "opacity": 0.96,
                                                            "justifyContent": "center", "alignItems": "center"
                                                        },
                                                        children=[
                                                            html.Div([
                                                                # Apple/network-inspired SVG animation (example: bouncing nodes)
                                                                html.Span("⦿", style={"fontSize": "58px", "animation": "bounce 1.2s infinite alternate", "color": "#0af"}),
                                                                html.Div("Building network…", style={"color": "#fff", "marginTop": "18px", "fontSize": "22px", "fontWeight": "600"})
                                                            ])
                                                        ]
                                                    ),
                                                    # Stores for focus/spotlight (hidden, but for callbacks)
                                                    # Floating "Show Info" button
                                                    html.Button(
                                                        ">>",
                                                        id="show-info-panel-btn",
                                                        n_clicks=0,
                                                        style={
                                                            "position": "absolute",
                                                            "top": "40px",
                                                            "right": "18px",
                                                            "zIndex": 1062,
                                                            "border": "none",
                                                            "borderRadius": "20px 0 0 20px",
                                                            "backgroundColor": "#333",
                                                            "color": "#fff",
                                                            "fontWeight": "700",
                                                            "fontSize": "21px",
                                                            "width": "54px",
                                                            "height": "54px",
                                                            "boxShadow": "0 2px 14px #0007",
                                                            "cursor": "pointer",
                                                            "transition": "background 0.18s"
                                                        },
                                                        title="Show Information Explorer"
                                                    ),
                                                    # Backdrop overlay (initially hidden)
                                                    html.Div(
                                                        id="info-backdrop",
                                                        style={
                                                            "display": "none",
                                                            "position": "fixed",
                                                            "top": 0,
                                                            "left": 0,
                                                            "width": "100vw",
                                                            "height": "100vh",
                                                            "background": "rgba(0,0,0,0.44)",
                                                            "zIndex": 1060,
                                                            "transition": "opacity 0.25s"
                                                        }
                                                    ),
                                                    # Slide-in drawer overlay for Information Box
                                                    html.Div(
                                                        id="info-panel",
                                                        style={
                                                            "display": "none",
                                                            "position": "fixed",
                                                            "top": "0px",
                                                            "right": "-600px",
                                                            "width": "600px",
                                                            "height": "100vh",
                                                            "backgroundColor": APPLE_PANEL,
                                                            "boxShadow": "0 0 32px 0 #000b",
                                                            "borderRadius": "28px 0 0 28px",
                                                            "padding": "0",
                                                            "zIndex": 1061,
                                                            "overflowY": "auto",
                                                            "transition": "right 0.33s cubic-bezier(.67,.09,.34,.97), opacity 0.33s"
                                                        },
                                                        children=[
                                                            html.Div(
                                                                style={
                                                                    "display": "flex",
                                                                    "flexDirection": "row",
                                                                    "alignItems": "center",
                                                                    "justifyContent": "space-between",
                                                                    "padding": "22px 28px 0 28px"
                                                                },
                                                                children=[
                                                                    html.H4(
                                                                        "Information Box",
                                                                        style={"color": "#fff", "fontWeight": "700", "marginBottom": "8px"}
                                                                    ),
                                                                    html.Button(
                                                                        "✕",
                                                                        id="close-info-panel-btn",
                                                                        n_clicks=0,
                                                                        style={
                                                                            "background": "none",
                                                                            "border": "none",
                                                                            "color": "#bbb",
                                                                            "fontSize": "30px",
                                                                            "fontWeight": "700",
                                                                            "cursor": "pointer",
                                                                            "marginLeft": "12px"
                                                                        },
                                                                        title="Close"
                                                                    )
                                                                ]
                                                            ),
                                                            html.Div(
                                                                style={"padding": "0 28px 28px 28px"},
                                                                children=[
                                                                    dash_table.DataTable(
                                                                        id="explorer-table",
                                                                        columns=[
                                                                            {"name": "ID", "id": "ID"},
                                                                            {"name": "Title", "id": "Title"},
                                                                            {"name": "Description", "id": "ShortDescription"},
                                                                            {"name": "Driving Force", "id": "Driving Force"},
                                                                            {"name": "Cluster", "id": "Cluster"},
                                                                        ],
                                                                        data=[],
                                                                        tooltip_data=[],
                                                                        tooltip_duration=None,
                                                                        css=[{
                                                                            "selector": ".dash-table-tooltip",
                                                                            "rule": "background-color: #222; color: #ddd; font-size: 12px;"
                                                                        }],
                                                                        row_selectable='multi',
                                                                        selected_rows=[],  # PATCH: allow external control of selected rows
                                                                        page_action='native',
                                                                        page_size=20,
                                                                        style_table={
                                                                            "overflowX": "auto",
                                                                            "backgroundColor": APPLE_PANEL,
                                                                            "border": "none",
                                                                            "borderRadius": APPLE_RADIUS,
                                                                            "boxShadow": APPLE_SHADOW,
                                                                            "margin": "0px auto",
                                                                            "width": "100%",
                                                                            "padding": "0 8px",
                                                                        },
                                                                        style_data={
                                                                            "backgroundColor": APPLE_PANEL,
                                                                            "color": "#fff",
                                                                            "border": "none",
                                                                            "fontFamily": APPLE_FONT,
                                                                            "fontSize": "15px",
                                                                            "padding": "18px 12px",
                                                                            "borderRadius": APPLE_RADIUS,
                                                                            "boxShadow": "none"
                                                                        },
                                                                        style_cell_conditional=[
                                                                            {"if": {"column_id": "ID"}, "width": "54px", "textAlign": "center"},
                                                                            {"if": {"column_id": "Driving Force"}, "width": "90px", "textAlign": "center"},
                                                                        ],
                                                                        style_data_conditional=[
                                                                            {"if": {"row_index": "odd"}, "backgroundColor": "#191919"},
                                                                            {"if": {"state": "active"}, "backgroundColor": "#0af", "color": "#fff"},
                                                                            {"if": {"state": "selected"}, "backgroundColor": "#222", "color": "#fff"},
                                                                        ],
                                                                        style_header={
                                                                            "backgroundColor": "#222",
                                                                            "color": "#fff",
                                                                            "fontWeight": "bold",
                                                                            "borderBottom": "2px solid #222",
                                                                            "fontFamily": APPLE_FONT,
                                                                            "fontSize": "18px",
                                                                            "textAlign": "center",
                                                                            "borderRadius": APPLE_RADIUS,
                                                                            "boxShadow": APPLE_SHADOW,
                                                                            "padding": "18px 8px"
                                                                        },
                                                                        style_cell={
                                                                            "textAlign": "left",
                                                                            "padding": "14px 10px",
                                                                            "whiteSpace": "pre-line",
                                                                            "height": "auto",
                                                                            "border": "none",
                                                                        },
                                                                    ),
                                                                    html.Div(
                                                                        style={"marginTop": "15px", "display": "flex", "gap": "12px"},
                                                                        children=[
                                                                            dbc.Button(
                                                                                "Select All",
                                                                                id="select-all-button",
                                                                                color="primary",
                                                                                size="sm",
                                                                                style={**APPLE_BUTTON, "width": "40%", "fontSize": "13px"}
                                                                            ),
                                                                            dbc.Button(
                                                                                "Deselect All",
                                                                                id="deselect-all-button",
                                                                                color="secondary",
                                                                                size="sm",
                                                                                style={**APPLE_BUTTON_SECONDARY, "width": "40%", "fontSize": "13px"}
                                                                            ),
                                                                            dbc.Button(
                                                                                "Export to PDF",
                                                                                id="export-explorer-pdf-button",
                                                                                color="info",
                                                                                size="sm",
                                                                                style={
                                                                                    **APPLE_BUTTON_SECONDARY,
                                                                                    "backgroundColor": "#333",
                                                                                    "border": "1px solid #555",
                                                                                    "color": "white",
                                                                                    "width": "40%",
                                                                                    "fontSize": "13px"
                                                                                }
                                                                            )
                                                                        ]
                                                                    ),
                                                                    dcc.Download(id="download-explorer-pdf"),
                                                                    dcc.Store(id="selected-ids-store", data=[]),
                                                                    dcc.Store(id="current-subset", data=[])
                                                                ]
                                                            ),
                                                        ]
                                                    ),
                                                ]
                                            )
                                        ]
                                    ),
                                ]
                            )
                        ]
                    ),
                ]
            )
        ]
    )



#############################################
# PROJECTS / COPILOT LAYOUT
#############################################
def _render_chat(messages):
    if not messages:
        return []
    bubbles = []
    for msg in messages:
        role = msg.get("role", "assistant")
        text = msg.get("text", "")
        img_url = msg.get("image_url")
        is_user = role == "user"
        # Support file message rendering
        file_info = msg.get("file")
        # Render image file inline if present, or download link for non-image
        if file_info and isinstance(file_info, dict):
            filetype = file_info.get("type", "")
            filename = file_info.get("name", "file")
            filesize = file_info.get("size", 0)
            file_data = file_info.get("content")
            preview = None
            # Show image preview if filetype image or image_url present
            if (filetype.startswith("image/") and file_data):
                preview = html.Img(
                    src=f"data:{filetype};base64,{file_data}",
                    style={
                        "maxWidth": "70%",
                        "alignSelf": "flex-start" if not is_user else "flex-end",
                        "margin": "4px 0",
                        "borderRadius": "12px",
                        "boxShadow": APPLE_SHADOW
                    }
                )
            else:
                if file_data:
                    href = f"data:{filetype};base64,{file_data}"
                else:
                    href = "#"
                preview = html.A(
                    "Download",
                    href=href,
                    download=filename,
                    target="_blank",
                    style={"color": "#0af", "textDecoration": "underline", "marginLeft": "8px"}
                )
            # File bubble
            bubbles.append(
                html.Div(
                    [
                        html.Strong(filename or "Untitled Chat", style={"color": "#fff"}),
                        html.Span(f" ({filesize//1024} KB)" if filesize else "", style={"color": "#bbb", "fontSize": "13px", "marginLeft": "7px"}),
                        html.Div(preview) if preview else None,
                    ],
                    style={
                        "maxWidth": "70%",
                        "alignSelf": "flex-end" if is_user else "flex-start",
                        "backgroundColor": "#222" if not is_user else "#444",
                        "color": "#fff",
                        "padding": "10px 16px",
                        "margin": "7px 0",
                        "borderRadius": "14px",
                        "whiteSpace": "pre-wrap",
                        "fontSize": "14px",
                        "fontFamily": APPLE_FONT,
                        "border": "1.3px solid #444",
                    },
                )
            )
        # Assistant-generated image (image_url) or user image_url
        if img_url or (file_info and isinstance(file_info, dict) and file_info.get("type", "").startswith("image/")):
            # If already handled above, skip, unless image_url present
            if img_url:
                bubbles.append(
                    html.Img(
                        src=img_url,
                        style={
                            "maxWidth": "70%",
                            "alignSelf": "flex-start",
                            "margin": "4px 0",
                            "borderRadius": "12px",
                            "boxShadow": APPLE_SHADOW
                        },
                    )
                )
        if text:
            bubbles.append(
                html.Div(
                    text,
                    style={
                        "maxWidth": "70%",
                        "alignSelf": "flex-end" if is_user else "flex-start",
                        "backgroundColor": "#007aff" if is_user else "#282828",
                        "color": "#fff",
                        "padding": "12px 18px",
                        "margin": "6px 0",
                        "borderRadius": "14px",
                        "whiteSpace": "pre-wrap",
                        "fontSize": "15px",
                        "fontFamily": APPLE_FONT
                    },
                )
            )
    return bubbles

def copilot_layout():
    return html.Div(
        style={
            "width": "100vw",
            "height": "100vh",
            "backgroundColor": APPLE_BG,
            "display": "flex",
            "flexDirection": "column",
            "fontFamily": APPLE_FONT
        },
        children=[
            get_navbar("Copilot"),
            html.Div(
                style={
                    "flex": "1",
                    "display": "flex",
                    "flexDirection": "row",
                    "height": "calc(100vh - 60px)"
                },
                children=[
                    html.Div(
                        style={
                            "width": "300px",
                            "backgroundColor": APPLE_PANEL,
                            "borderRight": "1px solid rgba(255,255,255,0.09)",
                            "display": "flex",
                            "flexDirection": "column"
                        },
                        children=[
                            dbc.Button(
                                "New Chat",
                                id="new-chat-button",
                                color="primary",
                                style={**APPLE_BUTTON, "width": "100%", "marginTop": "12px"}
                            ),
                            html.Hr(style={"borderTop": "1.5px solid #282828"}),
                            html.Div("Chats", style={"color": "#fff", "fontWeight": "700", "fontSize": "18px", "padding": "12px"}),
                            html.Ul(id="chat-list", style={"listStyleType": "none", "padding": "0 14px"})
                        ]
                    ),
                    html.Div(
                        style={
                            "flex": "1",
                            "overflowY": "auto",
                            "backgroundColor": APPLE_BG,
                            "display": "flex",
                            "flexDirection": "column"
                        },
                        children=[
                            dcc.Store(id='chat-threads', data=[]),
                            dcc.Store(id="copilot-conversation", data={'thread_id': None, 'messages': [], 'active': False}),
                            html.Div(
                                id="copilot-chat-container",
                                style={
                                    "flex": "1",
                                    "padding": "36px",
                                    "backgroundColor": APPLE_ACCENT,
                                    "overflowY": "auto",
                                    "borderLeft": "1px solid rgba(255,255,255,0.09)",
                                    "fontFamily": APPLE_FONT
                                }
                            ),
                            html.Div(
                                style={
                                    "padding": "18px",
                                    "borderTop": "1px solid #222",
                                    "backgroundColor": "#1a1a1a"
                                },
                                children=[
                                    html.Div("ORION Copilot", style={
                                        "color": "#fff",
                                        "fontSize": "18px",
                                        "fontWeight": "700",
                                        "marginBottom": "12px"
                                    }),
                                    dcc.Textarea(
                                        id='user-input-copilot',
                                        placeholder='Ask a question...',
                                        style={
                                            "width": "80%",
                                            "height": "80px",
                                            "backgroundColor": "#232323",
                                            "color": "#eee",
                                            "border": "1.5px solid #444",
                                            "borderRadius": APPLE_RADIUS,
                                            "padding": "12px",
                                            "fontSize": "16px",
                                            "fontFamily": APPLE_FONT
                                        }
                                    ),
                                    html.Div(
                                        style={"display": "flex", "gap": "12px", "alignItems": "center", "marginTop": "14px"},
                                        children=[
                                            dbc.Button(
                                                'Submit',
                                                id='submit-question-copilot',
                                                color='secondary',
                                                style={**APPLE_BUTTON, "width": "120px", "fontSize": "15px"}
                                            ),
                                            dcc.Upload(
                                                id="copilot-file-upload",
                                                children=html.Div("Add file", style={"textAlign": "center"}),
                                                multiple=False,
                                                style={
                                                    **APPLE_BUTTON_SECONDARY,
                                                    "width": "120px",
                                                    "textAlign": "center",
                                                    "backgroundColor": "#333"
                                                },
                                            )
                                        ]
                                    ),
                                    dcc.Store(id="current-copilot-store", data={}),
                                    dcc.Store(id="chat-rename-modal-store", data={"show": False, "chat_id": None}),
                                    dbc.Modal(
                                        [
                                            dbc.ModalHeader("Rename Chat"),
                                            dbc.ModalBody(
                                                dcc.Input(
                                                    id="chat-rename-input",
                                                    type="text",
                                                    value="",
                                                    maxLength=80,
                                                    debounce=True,
                                                    style={
                                                        "width": "100%",
                                                        "backgroundColor": "#232323",
                                                        "color": "#eee",
                                                        "border": "1.5px solid #444",
                                                        "borderRadius": APPLE_RADIUS,
                                                        "padding": "12px",
                                                        "fontSize": "16px"
                                                    }
                                                )
                                            ),
                                            dbc.ModalFooter(
                                                dbc.Button("Save", id="chat-rename-save-btn", color="primary", n_clicks=0, style={"width": "120px"})
                                            ),
                                        ],
                                        id="chat-rename-modal",
                                        is_open=False,
                                        centered=True,
                                        backdrop=True
                                    ),
                                ]
                            )
                        ]
                    )
                ]
            ),
            # --- Apple-style floating Copilot AI button (bottom right) ---
            html.Button(
                id="show-scanning-copilot-btn",
                n_clicks=0,
                title="Open Copilot",
                style={
                    "position": "fixed",
                    "bottom": "34px",
                    "right": "34px",
                    "width": "66px",
                    "height": "66px",
                    "borderRadius": "50%",
                    "backdropFilter": "blur(16px)",
                    "background": "rgba(30,30,30,0.7)",
                    "border": "1.5px solid #2c2c2c",
                    "boxShadow": "0 8px 32px rgba(0,122,255,0.28), 0 2px 8px #222a",
                    "display": "flex",
                    "alignItems": "center",
                    "justifyContent": "center",
                    "zIndex": 2002,
                    "cursor": "pointer",
                    "transition": "all 0.18s cubic-bezier(.4,0,.2,1)",
                    "outline": "none",
                    "backgroundImage": "url(\"data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='38' height='38' viewBox='0 0 44 44' fill='none'><defs><radialGradient id='copilot-glow' cx='50%' cy='50%' r='60%' fx='50%' fy='50%'><stop offset='0%' stop-color='%237fdcff' stop-opacity='0.92'/><stop offset='100%' stop-color='%23007aff' stop-opacity='0.82'/></radialGradient></defs><circle cx='22' cy='22' r='19' fill='url(%23copilot-glow)' stroke='%23fff' stroke-width='2'/><rect x='14' y='20' width='16' height='3' rx='1.2' fill='%23fff'/><rect x='14' y='26' width='9' height='3' rx='1.2' fill='%23fff'/><circle cx='22' cy='15' r='2.1' fill='%23fff'/></svg>\")",
                    "backgroundRepeat": "no-repeat",
                    "backgroundPosition": "center",
                    "backgroundSize": "38px 38px"
                },
                children=[]
            )
        ]
    )


# LOGIN and REGISTER LAYOUTS
def dash_login_layout():
    return html.Div(
        style={
            "height": "100vh", "display": "flex", "flexDirection": "column",
            "justifyContent": "center", "alignItems": "center", "backgroundColor": "#000"
        },
        children=[
            html.Div([
                html.H2("Login", style={"color": "#fff"}),
                dcc.Input(id='dash-login-email', type='email', placeholder='Email', style={"margin": "8px", "padding": "8px"}),
                dcc.Input(id='dash-login-password', type='password', placeholder='Password', style={"margin": "8px", "padding": "8px"}),
                html.Button('Login', id='dash-login-submit', style={"margin": "8px"}),
                html.Div(id='dash-login-error', style={'color': 'red', "margin": "8px"}),
                html.Br(),
                dcc.Link('Register here', href='/register', style={"color": "#0af"})
            ], style={
                "backgroundColor": "#111", "padding": "36px", "borderRadius": "18px",
                "boxShadow": "0 2px 24px #2229"
            }),
        ]
    )

def dash_register_layout():
    return html.Div(
        style={
            "height": "100vh", "display": "flex", "flexDirection": "column",
            "justifyContent": "center", "alignItems": "center", "backgroundColor": "#000"
        },
        children=[
            html.Div([
                html.H2("Register", style={"color": "#fff"}),
                dcc.Input(id='dash-register-email', type='email', placeholder='Email', style={"margin": "8px", "padding": "8px"}),
                dcc.Input(id='dash-register-password', type='password', placeholder='Password', style={"margin": "8px", "padding": "8px"}),
                html.Button('Register', id='dash-register-submit', style={"margin": "8px"}),
                html.Div(id='dash-register-error', style={'color': 'red', "margin": "8px"}),
                html.Br(),
                dcc.Link('Already have an account? Login', href='/login', style={"color": "#0af"})
            ], style={
                "backgroundColor": "#111", "padding": "36px", "borderRadius": "18px",
                "boxShadow": "0 2px 24px #2229"
            }),
        ]
    ) 



###################################################
# --- Info Panel Drawer/Backdrop Toggle Callback ---
@app.callback(
    [
        Output("info-panel-open", "data"),
        Output("info-panel", "style"),
        Output("info-backdrop", "style"),
    ],
    [
        Input("show-info-panel-btn", "n_clicks"),
        Input("close-info-panel-btn", "n_clicks"),
        Input("info-backdrop", "n_clicks"),
    ],
    State("info-panel-open", "data"),
    prevent_initial_call=False
)
def toggle_info_panel(show_btn, close_btn, backdrop_btn, panel_open):
    # Robust toggling: open on show, close on X or backdrop click
    ctx = dash.callback_context
    # Default closed
    open_panel = False if panel_open is None else bool(panel_open)
    # Find which triggered
    if ctx.triggered:
        trig = ctx.triggered[0]["prop_id"].split(".")[0]
        if trig == "show-info-panel-btn":
            open_panel = True
        elif trig in ("close-info-panel-btn", "info-backdrop"):
            open_panel = False
    # Panel style
    base_panel = {
        "position": "fixed",
        "top": "0px",
        "right": "-600px",
        "width": "600px",
        "height": "100vh",
        "backgroundColor": APPLE_PANEL,
        "boxShadow": "0 0 32px 0 #000b",
        "borderRadius": "28px 0 0 28px",
        "padding": "0",
        "zIndex": 1061,
        "overflowY": "auto",
        "transition": "right 0.33s cubic-bezier(.67,.09,.34,.97), opacity 0.33s"
    }
    panel_style = dict(base_panel)
    backdrop_style = {
        "position": "fixed",
        "top": 0,
        "left": 0,
        "width": "100vw",
        "height": "100vh",
        "background": "rgba(0,0,0,0.44)",
        "zIndex": 1060,
        "transition": "opacity 0.25s"
    }
    if open_panel:
        panel_style["right"] = "0px"
        panel_style["display"] = "block"
        panel_style["opacity"] = "1"
        backdrop_style["display"] = "block"
        backdrop_style["opacity"] = "1"
        backdrop_style["pointerEvents"] = "auto"
    else:
        panel_style["right"] = "-600px"
        panel_style["display"] = "none"
        panel_style["opacity"] = "0"
        backdrop_style["display"] = "none"
        backdrop_style["opacity"] = "0"
        backdrop_style["pointerEvents"] = "none"
    return open_panel, panel_style, backdrop_style



@app.callback(
    [
        Output("tsne-plot", "figure"),
        Output("explorer-table", "data"),
        Output("explorer-table", "tooltip_data"),
    ],
    [
        Input("search-term", "n_submit"),
        Input("apply-filters-button", "n_clicks"),
        Input("reset-filters-button", "n_clicks"),
        Input("cluster-highlight", "value"),
        Input("driving-force-filter", "value"),
        Input("show-edges", "value"),
        Input("show-signals-toggle", "value"),
        Input("show-node-titles", "value"),
        Input("network-dimension", "value"),
    ],
    [
        State("search-term", "value"),
    ],
    prevent_initial_call=False
)
def update_tsne_plot(
    search_submit,
    apply_filters_clicks,
    reset_filters_clicks,
    cluster_highlight,
    driving_force_filter,
    show_edges,
    show_signals_toggle,
    show_titles_value,
    network_dimension,
    search_query,
):
    ctx = dash.callback_context
    triggered = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
    filtered_data = data.copy()

    # --- Reset logic ---
    if triggered == "reset-filters-button":
        filtered_data = data.copy()
        search_query = ""
        cluster_highlight = -1
        driving_force_filter = ["(All)"]

    # Only apply filter if search_query is not None and not empty after stripping
    if search_query is not None and str(search_query).strip() != "":
        query = search_query.strip()
        mask = data["PreprocessedText"].apply(lambda txt: match_advanced_query(txt, query))
        if mask.sum() == 0:
            filtered_data = data.copy()  # No results, show all instead of nothing
        else:
            filtered_data = filtered_data[mask]
    else:
        filtered_data = data.copy()

    # --- Additional filtering ---
    # [CLUSTER FILTER]
    if cluster_highlight is not None and cluster_highlight != -1:
        filtered_data = filtered_data[filtered_data["Cluster"] == cluster_highlight]

    # [DRIVING-FORCE FILTER]
    if driving_force_filter and isinstance(driving_force_filter, list):
        if "(All)" not in driving_force_filter:
            filtered_data = filtered_data[filtered_data["Driving Force"].isin(driving_force_filter)]
    elif driving_force_filter and driving_force_filter != "(All)":
        filtered_data = filtered_data[filtered_data["Driving Force"] == driving_force_filter]

    # [SIGNALS VISIBILITY]
    # Hide “signals” unless the toggle checklist contains "show"
    if not show_signals_toggle or "show" not in show_signals_toggle:
        filtered_data = filtered_data[filtered_data["Driving Force"].str.lower() != "signals"]

    # --- Final prep ---
    filtered_data = filtered_data.reset_index(drop=True)
    show_titles = "titles" in (show_titles_value or [])

    fig = generate_tsne_plot(
        filtered_data,
        show_edges=("edges" in (show_edges or [])),
        dimension=network_dimension,
        show_titles=show_titles
    )

    # --- Explorer-table data & tooltips ---
    def short_description(desc):
        if not isinstance(desc, str):
            return ""
        words = desc.split()
        return desc if len(words) <= 12 else " ".join(words[:12]) + "…"

    explorer_data = [
        {
            "ID": row["ID"],
            "Title": row["Title"],
            "ShortDescription": short_description(row["Description"]),
            "Driving Force": row["Driving Force"],
            "Cluster": row["Cluster"],
        }
        for _, row in filtered_data.iterrows()
    ]
    explorer_tooltips = [
        {
            "ID": {"value": str(row["ID"]), "type": "markdown"},
            "Title": {"value": row["Title"], "type": "markdown"},
            "ShortDescription": {"value": row["Description"], "type": "markdown"},
            "Driving Force": {"value": row["Driving Force"], "type": "markdown"},
            "Cluster": {"value": str(row["Cluster"]), "type": "markdown"},
        }
        for _, row in filtered_data.iterrows()
    ]

    # --- 2D / 3D handling ---
    if network_dimension == "3d":
        initial_camera = dict(
            eye=dict(x=1.7, y=1.7, z=1.7),
            up=dict(x=0, y=0, z=1)
        )
        fig.update_layout(scene_camera=initial_camera)
        return fig, explorer_data, explorer_tooltips
    else:
        return fig, explorer_data, explorer_tooltips


# --- Reset-camera callback (unchanged) ---
@app.callback(
    Output("tsne-plot", "relayoutData"),
    Input("reset-camera-btn", "n_clicks"),
    State("scene-camera-store", "data"),
    prevent_initial_call=True
)
def reset_camera(n_clicks, stored_camera):
    if n_clicks and stored_camera:
        return {"scene.camera": stored_camera}
    return dash.no_update


#############################################
# PAGE NAVIGATION AND APP LAYOUT
#############################################

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])


@app.callback(Output('page-content', 'children'), [Input('url', 'pathname')])
def display_page(pathname):
    from flask_login import current_user
    if pathname == '/login':
        return dash_login_layout()
    if pathname == '/register':
        return dash_register_layout()
    # All other pages require authentication
    if not current_user.is_authenticated:
        return dcc.Location(href='/login', id='redirect-login')
    if pathname == '/scanning':
        return scanning_layout()
    if pathname == '/copilot':
        return copilot_layout()
    # default/home
    return home_layout()

# Login/register Dash callbacks
@app.callback(
    Output('dash-login-error', 'children'),
    Output('url', 'pathname', allow_duplicate=True),
    Input('dash-login-submit', 'n_clicks'),
    State('dash-login-email', 'value'),
    State('dash-login-password', 'value'),
    prevent_initial_call=True
)
def dash_login(n_clicks, email, password):
    if not email or not password:
        return "Please enter both fields.", dash.no_update
    user = User.query.filter_by(email=email).first()
    if user and user.check_password(password):
        login_user(user)
        return "", "/"
    return "Invalid credentials.", dash.no_update

@app.callback(
    Output('dash-register-error', 'children'),
    Output('url', 'pathname', allow_duplicate=True),
    Input('dash-register-submit', 'n_clicks'),
    State('dash-register-email', 'value'),
    State('dash-register-password', 'value'),
    prevent_initial_call=True
)
def dash_register(n_clicks, email, password):
    if not email or not password:
        return "Please enter both fields.", dash.no_update
    if User.query.filter_by(email=email).first():
        return "Email already registered.", dash.no_update
    new_user = User(email=email)
    new_user.set_password(password)
    db.session.add(new_user)
    db.session.commit()
    login_user(new_user)
    return "", "/"

#############################################
# CHAT MANAGEMENT CALLBACKS (PROJECTS PAGE)
#############################################
@app.callback(
    [
        Output('chat-threads', 'data'),
        Output('copilot-conversation', 'data'),
        Output('copilot-chat-container', 'children'),
        Output('user-input-copilot', 'value')
    ],
    Input('new-chat-button', 'n_clicks'),
    State('chat-threads', 'data'),
    State('copilot-conversation', 'data'),
    prevent_initial_call=True
)
def add_new_chat(n_clicks, threads, conv):
    import datetime
    if not isinstance(threads, list):
        threads = []
    if not isinstance(conv, dict):
        conv = {'thread_id': None, 'messages': [], 'active': False}
    archived_threads = list(threads)
    title = None
    messages = conv.get("messages", []) if isinstance(conv, dict) else []
    # Title selection logic
    for msg in messages:
        if msg.get("role") == "user" and msg.get("text"):
            title = msg["text"][:40] + ("…" if len(msg["text"]) > 40 else "")
            break
    if not title:
        for msg in messages:
            if msg.get("file"):
                file_info = msg.get("file")
                if file_info and isinstance(file_info, dict):
                    fname = file_info.get("name", "")
                    if fname and fname.strip():
                        title = fname.strip()
                        break
    # Fallback always applies
    if not title or not str(title).strip():
        title = "Untitled Chat"
    # Avoid duplicate consecutive
    is_duplicate = False
    if archived_threads:
        last_conv = archived_threads[-1].get("conv", {})
        last_msgs = last_conv.get("messages", []) if isinstance(last_conv, dict) else []
        if messages == last_msgs:
            is_duplicate = True
    if not is_duplicate:
        archived_threads.append({
            "id": str(uuid.uuid4()),
            "label": title,
            "conv": conv,
            "created_at": datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
        })
    fresh_conv = {'thread_id': None, 'messages': [], 'active': False}
    return archived_threads, fresh_conv, [], ""

@app.callback(
    Output('chat-list', 'children'),
    Input('chat-threads', 'data')
)
def update_chat_list(threads):
    if not threads:
        return []
    items = []
    for t in threads:
        # Ensure label fallback
        label = t.get("label", "") or "Untitled Chat"
        items.append(
            html.Li(
                style={"display": "flex", "alignItems": "center", "gap": "8px", "marginBottom": "4px"},
                children=[
                    html.Button(
                        label,
                        id={"type": "chat-select", "index": t["id"]},
                        n_clicks=0,
                        style={
                            "all": "unset",
                            "display": "block",
                            "width": "65%",
                            "padding": "4px 0",
                            "textAlign": "left",
                            "color": "#fff",
                            "cursor": "pointer",
                            "fontWeight": "bold"
                        },
                        title=label
                    ),
                    html.Button(
                        "✏️",
                        id={"type": "chat-rename", "index": t["id"]},
                        n_clicks=0,
                        style={
                            "all": "unset",
                            "color": "#aaa",
                            "fontSize": "14px",
                            "cursor": "pointer"
                        },
                        title="Rename chat"
                    ),
                    html.Button(
                        "🗑",
                        id={"type": "chat-delete", "index": t["id"]},
                        n_clicks=0,
                        style={
                            "all": "unset",
                            "color": "#f66",
                            "cursor": "pointer",
                            "fontSize": "14px"
                        },
                        title="Delete chat"
                    ),
                    html.Span(
                        t.get("created_at", ""),
                        style={"color": "#777", "fontSize": "11px", "marginLeft": "6px"}
                    )
                ]
            )
        )
    return items


# --- FILE UPLOAD HANDLING FOR CHAT ---
@app.callback(
    Output('copilot-conversation', 'data', allow_duplicate=True),
    Output('copilot-chat-container', 'children', allow_duplicate=True),
    Input('copilot-file-upload', 'contents'),
    State('copilot-file-upload', 'filename'),
    State('copilot-file-upload', 'last_modified'),
    State('copilot-conversation', 'data'),
    prevent_initial_call=True
)
def handle_file_upload(contents, filename, last_modified, conv_data):
    import mimetypes
    if not contents:
        raise dash.exceptions.PreventUpdate
    if not isinstance(conv_data, dict):
        conv_data = {'thread_id': None, 'messages': [], 'active': False}
    # Parse file info
    # contents: "data:<mimetype>;base64,<data>"
    try:
        header, b64data = contents.split(",", 1)
        mimetype = header.split(";")[0].replace("data:", "") or mimetypes.guess_type(filename)[0] or "application/octet-stream"
    except Exception:
        mimetype = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        b64data = ""
    size = 0
    try:
        import base64
        decoded = base64.b64decode(b64data)
        size = len(decoded)
    except Exception:
        size = 0
    file_info = {
        "name": filename,
        "type": mimetype,
        "size": size,
        "content": b64data
    }
    # Insert file message into conversation
    conv_data.setdefault("messages", []).append({
        "role": "user",
        "file": file_info
    })
    return conv_data, _render_chat(conv_data["messages"])


@app.callback(
    Output('chat-threads', 'data', allow_duplicate=True),
    Input({'type': 'chat-delete', 'index': ALL}, 'n_clicks'),
    State('chat-threads', 'data'),
    prevent_initial_call=True
)
def delete_chat(n_clicks_list, threads):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    id_dict = json.loads(triggered_id)
    chat_index = id_dict['index']
    threads = [t for t in (threads or []) if t['id'] != chat_index]
    return threads

# --- Chat Rename Modal: open modal callback
@app.callback(
    Output("chat-rename-modal-store", "data", allow_duplicate=True),
    Output("chat-rename-modal", "is_open", allow_duplicate=True),
    Output("chat-rename-input", "value", allow_duplicate=True),
    Input({'type': 'chat-rename', 'index': ALL}, 'n_clicks'),
    State('chat-threads', 'data'),
    prevent_initial_call=True
)
def open_rename_modal(n_clicks_list, threads):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    import json
    id_dict = json.loads(triggered_id)
    chat_index = id_dict['index']
    title = ""
    for t in (threads or []):
        if t['id'] == chat_index:
            title = t['label']
            break
    return {"show": True, "chat_id": chat_index}, True, title

# --- Chat Rename Modal: save callback
@app.callback(
    Output('chat-threads', 'data', allow_duplicate=True),
    Output("chat-rename-modal-store", "data", allow_duplicate=True),
    Output("chat-rename-modal", "is_open", allow_duplicate=True),
    Input("chat-rename-save-btn", "n_clicks"),
    State("chat-rename-input", "value"),
    State("chat-rename-modal-store", "data"),
    State('chat-threads', 'data'),
    prevent_initial_call=True
)
def rename_chat(n_clicks, new_label, modal_state, threads):
    if not modal_state or not modal_state.get("show"):
        raise dash.exceptions.PreventUpdate
    chat_id = modal_state.get("chat_id")
    updated_threads = []
    for t in (threads or []):
        if t['id'] == chat_id:
            t = {**t, "label": new_label.strip() or "Untitled Chat"}
        updated_threads.append(t)
    return updated_threads, {"show": False, "chat_id": None}, False

@app.callback(
    [
        Output('copilot-chat-container', 'children', allow_duplicate=True),
        Output('copilot-conversation', 'data', allow_duplicate=True),
        Output('user-input-copilot', 'value', allow_duplicate=True)
    ],
    Input('submit-question-copilot', 'n_clicks'),
    State('user-input-copilot', 'value'),
    State('copilot-conversation', 'data'),
    prevent_initial_call=True
)
def handle_copilot_query(n_clicks, user_input, conv_data):
    if not user_input or not user_input.strip():
        raise dash.exceptions.PreventUpdate

    if not isinstance(conv_data, dict):
        conv_data = {'thread_id': None, 'messages': [], 'active': False}
    # Find if last message was a file message
    file_message = None
    if conv_data.get("messages"):
        # If last message is a user file, attach file to this prompt
        last_msg = conv_data["messages"][-1]
        if last_msg.get("role") == "user" and last_msg.get("file"):
            file_message = last_msg.get("file")
            # Remove file message so file is referenced as part of this prompt
            conv_data["messages"] = conv_data["messages"][:-1]
    # Append user message, referencing file if present
    user_msg = {"role": "user", "text": user_input}
    if file_message:
        user_msg["file"] = file_message
    conv_data['messages'].append(user_msg)

    thread_id = conv_data.get("thread_id")
    if not thread_id:
        thread = client.beta.threads.create()
        thread_id = thread.id
        conv_data["thread_id"] = thread_id

    # Compose OpenAI message content
    # If file present, add a note or instruction
    msg_content = user_input
    if file_message:
        # If OpenAI Assistants API supports file context, attach file here (not implemented)
        # For now, just mention in prompt
        msg_content += f"\n\n[Attached file: {file_message.get('name','file')}]"
    client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=msg_content
    )

    run = client.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id=assistant_id
    )

    import time
    while True:
        run = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run.id)
        if run.status == "completed":
            break
        elif run.status in ("failed", "cancelled", "expired"):
            conv_data["messages"].append({
                "role": "assistant",
                "text": f"[Error: Assistant run {run.status}]"
            })
            return _render_chat(conv_data["messages"]), conv_data, ""
        time.sleep(0.5)

    messages = client.beta.threads.messages.list(thread_id=thread_id)
    for m in messages.data:
        if m.role == "assistant":
            # Support assistant returning image
            img_url = None
            text_content = None
            for content in m.content:
                if content.type == "text":
                    text_content = content.text.value
                elif content.type == "image_file":
                    # If OpenAI returns image as file (not supported yet), handle here
                    # For now, just skip
                    pass
                elif content.type == "image_url":
                    img_url = content.image_url
            if img_url:
                conv_data["messages"].append({
                    "role": "assistant",
                    "image_url": img_url
                })
            if img_url:
                bubbles.append(
                    html.Img(
                        src=img_url,
                        style={
                            "maxWidth": "70%",
                            "alignSelf": "flex-start",
                            "margin": "4px 0",
                            "borderRadius": "12px",
                            "boxShadow": APPLE_SHADOW
                        },
                    )
                )
            if text_content:
                conv_data["messages"].append({
                    "role": "assistant",
                    "text": text_content
                })
            break

    return _render_chat(conv_data["messages"]), conv_data, ""

@app.callback(
    Output('submit-question-copilot', 'disabled'),
    Input('copilot-conversation', 'data'),
)
def toggle_submit_disabled(conv):
    return False

#############################################
# TABLE SELECT/DESELECT + EXPORT TO PDF
#############################################

def wrap_text(text, width=90):
    """Simple word wrap for a line, returns a list of lines."""
    import textwrap
    return textwrap.wrap(text, width=width, break_long_words=False, replace_whitespace=False)

@app.callback(
    Output("download-explorer-pdf", "data"),
    Input("export-explorer-pdf-button", "n_clicks"),
    State("explorer-table", "data"),
    State("explorer-table", "selected_rows"),
    prevent_initial_call=True
)
def export_explorer_pdf(n_clicks, table_data, selected_rows):
    if not selected_rows:
        raise dash.exceptions.PreventUpdate

    selected_records = [table_data[i] for i in selected_rows][:100]  # Max 100

    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    left_margin = 50
    top_margin = 750
    line_height = 17
    bottom_margin = 50  # Minimum y before page break
    current_y = top_margin

    def wrap_text(text, max_width, pdf_canvas, fontname="Helvetica", fontsize=12):
        pdf_canvas.setFont(fontname, fontsize)
        words = text.split()
        lines = []
        cur_line = ""
        for word in words:
            test_line = f"{cur_line} {word}".strip()
            if pdf_canvas.stringWidth(test_line, fontname, fontsize) <= max_width:
                cur_line = test_line
            else:
                lines.append(cur_line)
                cur_line = word
        if cur_line:
            lines.append(cur_line)
        return lines

    for record in selected_records:
        # Truncate description to first 20 words + "..."
        description = record.get('Description') or record.get('ShortDescription') or ""
        words = description.split()
        short_desc = " ".join(words[:50]) + ("..." if len(words) > 50 else "")

        fields = [
            ("ID", str(record.get('ID', ''))),
            ("Title", str(record.get('Title', ''))),
            ("Description", short_desc),
            ("Driving Force", str(record.get('Driving Force', ''))),
            ("Cluster", str(record.get('Cluster', '')) if 'Cluster' in record else ""),
            ("Source", str(record.get('Source', '')) if 'Source' in record else ""),
        ]

        for label, value in fields:
            lines = wrap_text(f"{label}: {value}", width - 2*left_margin, p)
            for line in lines:
                if current_y < bottom_margin:
                    p.showPage()
                    p.setFont("Helvetica", 12)
                    current_y = top_margin
                p.drawString(left_margin, current_y, line)
                current_y -= line_height

        # Add space after each record
        current_y -= line_height
        if current_y < bottom_margin:
            p.showPage()
            p.setFont("Helvetica", 12)
            current_y = top_margin

    p.save()
    buffer.seek(0)

    return send_bytes(buffer.getvalue(), "Selected_Forces.pdf")

#############################################
# DASH SERVER LAUNCH
#############################################

# --- Apple-style Dash DataTable pagination CSS ---
app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>ORION Platform</title>
        {%favicon%}
        {%css%}
        <style>
        .dash-spreadsheet-container .dash-spreadsheet-pagination {
            background: #111 !important;
            border-radius: 16px;
            margin-top: 18px !important;
            margin-bottom: 12px !important;
            padding: 8px 18px 8px 10px;
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 8px;
            box-shadow: 0 2px 12px rgba(0,0,0,0.13);
        }
        .dash-spreadsheet-container .dash-spreadsheet-pagination button,
        .dash-spreadsheet-container .dash-spreadsheet-pagination input[type="number"] {
            background: #181818 !important;
            color: #fff !important;
            border-radius: 10px !important;
            border: none !important;
            margin: 0 4px;
            font-family: 'SF Pro Display', 'Helvetica Neue', Arial, sans-serif !important;
            font-weight: 600;
            font-size: 16px;
            padding: 7px 17px;
            transition: background 0.18s;
            box-shadow: 0 2px 8px rgba(0,0,0,0.18);
        }
        .dash-spreadsheet-container .dash-spreadsheet-pagination button:hover {
            background: #007aff !important;
            color: #fff !important;
            box-shadow: 0 2px 12px #007aff44;
        }
        .dash-spreadsheet-container .dash-spreadsheet-pagination button:focus {
            outline: 2px solid #007aff !important;
        }
        .dash-spreadsheet-container .dash-spreadsheet-pagination input[type="number"] {
            width: 52px !important;
            text-align: center;
            background: #232323 !important;
            color: #f3f3f3 !important;
            box-shadow: 0 1px 3px rgba(0,0,0,0.10);
        }
        /* --- Dropdown custom styles --- */
        .Select-menu-outer, .VirtualizedSelectOption {
            background-color: #181818 !important;
            color: #f3f3f3 !important;
            font-family: 'SF Pro Display', 'Helvetica Neue', Arial, sans-serif !important;
            border-radius: 10px !important;
        }
        .VirtualizedSelectOption {
            border-bottom: 1px solid #333 !important;
        }
        .VirtualizedSelectOption.is-focused {
            background-color: #222 !important;
            color: #fff !important;
        }
        .VirtualizedSelectOption.is-selected {
            background-color: #007aff !important;
            color: #fff !important;
        }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
"""

#############################################
# SINGLE MERGED TABLE SELECTION CALLBACK (no duplicates)
#############################################
# Only one callback sets Output("explorer-table", "selected_rows") and Output("explorer-selected-rows", "data", allow_duplicate=True)
@app.callback(
    [
        Output("explorer-table", "selected_rows"),
        Output("explorer-selected-rows", "data", allow_duplicate=True),
    ],
    [
        Input("tsne-plot", "clickData"),
        Input("explorer-table", "selected_rows"),
        Input("select-all-button", "n_clicks"),
        Input("deselect-all-button", "n_clicks"),
    ],
    [
        State("explorer-table", "data"),
    ],
    prevent_initial_call="initial_duplicate"
)
def unified_table_selection(
    tsne_click, selected_rows, select_all_clicks, deselect_all_clicks,
    table_data
):
    ctx = dash.callback_context
    if not ctx.triggered or not table_data:
        return dash.no_update, dash.no_update, dash.no_update

    triggered = ctx.triggered[0]['prop_id'].split('.')[0]

    # Handle Select All
    if triggered == "select-all-button":
        rows = list(range(len(table_data)))
        return rows, rows

    # Handle Deselect All
    if triggered == "deselect-all-button":
        return [], None, []

    # --- PATCH START: Multi-select logic for Plot clicks ---
    if triggered == "tsne-plot" and tsne_click:
        pt = tsne_click.get("points", [{}])[0]
        node_id = pt.get("customdata", [None])[0]
        row_indices = [i for i, row in enumerate(table_data) if str(row.get("ID")) == str(node_id)]
        if not row_indices:
            return dash.no_update, dash.no_update, dash.no_update
        row_idx = row_indices[0]
        selected_rows = selected_rows or []
        # Toggle node: add if not selected, remove if already selected
        if row_idx in selected_rows:
            selected_rows = [i for i in selected_rows if i != row_idx]
        else:
            selected_rows = selected_rows + [row_idx]
        # Keep focus on the most recently toggled node
        return selected_rows, selected_rows
    # --- PATCH END ---

    # Handle table row selection (focus follows first selected row, preserves multi)
    if triggered == "explorer-table":
        if selected_rows and len(selected_rows) > 0:
            idx = selected_rows[0]
            if idx < len(table_data):
                node_id = table_data[idx].get("ID")
                return selected_rows, node_id, selected_rows
        return [], None, []

    return dash.no_update, dash.no_update, dash.no_update

from dash.exceptions import PreventUpdate

@app.callback(
    [Output('new-project-name', 'style'),
     Output('rename-project-name', 'style'),
     Output('confirm-new-project', 'style'),
     Output('confirm-rename-project', 'style')],
    [Input('create-project', 'n_clicks'),
     Input('rename-project', 'n_clicks')],
    prevent_initial_call=True
)
def show_input_fields(create_clicks, rename_clicks):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate
    trig = ctx.triggered[0]['prop_id'].split('.')[0]

    confirm_style = {
        "background": "none",
        "border": "none",
        "color": "#00d26a",
        "fontSize": "22px",
        "marginLeft": "8px",
        "minWidth": "44px",
        "minHeight": "44px",
    }

    if trig == 'create-project':
        return (
            {'display': 'block', 'width': '95%'},
            {'display': 'none'},
            {**confirm_style, 'display': 'inline-block'},
            {**confirm_style, 'display': 'none'}
        )
    elif trig == 'rename-project':
        return (
            {'display': 'none'},
            {'display': 'block', 'width': '95%'},
            {**confirm_style, 'display': 'none'},
            {**confirm_style, 'display': 'inline-block'}
        )
    return (
        {'display': 'none'},
        {'display': 'none'},
        {**confirm_style, 'display': 'none'},
        {**confirm_style, 'display': 'none'}
    )

# === SAVE the current filters / search into the selected project =============
@app.callback(
    Output("save-status", "children", allow_duplicate=True),
    Input("save-project", "n_clicks"),
    State("project-selector", "value"),
    State("search-term", "value"),
    State("search-chips", "data"),
    State("cluster-highlight", "value"),
    State("driving-force-filter", "value"),
    State("explorer-selected-rows", "data"),
    prevent_initial_call=True,
)
def _save_project_cb(_, project_name, search, chips,
                     cluster, driving, explorer_rows):
    state = _gather_current_state(search, chips, cluster,
                                  driving, explorer_rows)
    projects[project_name] = {**default_project_state(), **state}
    save_projects(projects)
    return "Saved ✓"

# === LOAD project state whenever the selector changes ========================
@app.callback(
    Output("search-term", "value", allow_duplicate=True),
    Output("search-chips", "data", allow_duplicate=True),
    Output("cluster-highlight", "value", allow_duplicate=True),
    Output("driving-force-filter", "value", allow_duplicate=True),
    Output("explorer-selected-rows", "data", allow_duplicate=True),
    Input("project-selector", "value"),
    prevent_initial_call=True,
)
def _load_project_cb(project_name):
    st = projects.get(project_name)
    if not st:
        raise PreventUpdate
    return (st["search"], st["chips"], st["cluster"],
            st["driving_forces"], st["explorer_selected_rows"])

@app.callback(
    Output('confirm-delete', 'displayed'),
    Input('delete-project', 'n_clicks'),
    prevent_initial_call=True
)
def show_confirm_dialog(n_clicks):
    if n_clicks:
        return True
    return False

@app.callback(
    Output('save-status', 'children'),
    [Input('search-term', 'value'),
     Input('cluster-highlight', 'value'),
     Input('driving-force-filter', 'value'),
     Input('search-chips', 'data'),
     Input('explorer-selected-rows', 'data'),
     Input('save-project', 'n_clicks')],
    State('project-selector', 'value'),
    prevent_initial_call=True
)
def persist_state(search, cluster, driving_forces, chips, explorer_rows, n_save, project_name):
    state = projects.get(project_name, default_project_state())
    state['search'] = search or ""
    state['cluster'] = cluster if cluster is not None else -1
    state['driving_forces'] = driving_forces if driving_forces else ["(All)"]
    state['chips'] = chips or []
    state['explorer_selected_rows'] = explorer_rows or []
    projects[project_name] = state
    save_projects(projects)
    return f"Saved: {datetime.now().strftime('%H:%M:%S')}"

# --- Project Management Callbacks ---
# (Check for presence at file bottom)

from dash.exceptions import PreventUpdate

    
    
# Display chips as styled elements
@app.callback(
    Output('chips-area', 'children'),
    Input('search-chips', 'data')
)
def display_chips(chips):
    if not chips:
        return html.Span("No keywords added.", style={'color': '#888'})
    return [
        html.Span([
            chip,
            html.Button(
                '✕',
                id={'type': 'remove-chip', 'index': i},
                n_clicks=0,
                style={
                    'border': 'none', 'background': 'none', 'color': '#b00',
                    'marginLeft': '6px', 'cursor': 'pointer', 'fontWeight': 'bold'
                }
            )
        ],
        style={
            'display': 'inline-block', 'background': '#262626', 'color': '#fff',
            'borderRadius': '16px', 'padding': '6px 12px', 'margin': '3px',
            'fontSize': '15px'
        }) for i, chip in enumerate(chips)
    ]


# --- Unified Scanning Copilot Chatbox Logic ---
from dash.dependencies import ALL
from dash.exceptions import PreventUpdate

# Dynamic suggestions for Copilot, max 2 at a time
def get_scanning_copilot_prompts(selected_rows, search_term):
    n_selected = len(selected_rows or [])
    current_query = search_term or ""
    prompts = []
    if n_selected > 0:
        prompts.append("Summarize my current selection")
    elif current_query:
        prompts.append(f"Summarize search results for: {current_query}")
    # Always max 2 prompts
    return prompts[:2]

# Sync Copilot panel visibility
@app.callback(
    Output("scanning-copilot-open", "data"),
    Input("show-scanning-copilot-btn", "n_clicks"),
    State("scanning-copilot-open", "data"),
    prevent_initial_call=True
)
def toggle_scanning_copilot(n_clicks, is_open):
    if n_clicks:
        return not is_open if is_open is not None else True
    raise PreventUpdate

# Copilot panel box style (always block if open)
@app.callback(
    Output("scanning-copilot-panel", "style", allow_duplicate=True),
    Input("scanning-copilot-open", "data"),
    prevent_initial_call=True,
)

def show_hide_scanning_copilot(is_open):
    style = {
        "display": "block" if is_open else "none",
        "position": "fixed",
        "bottom": "110px",
        "right": "40px",
        "width": "420px",
        "maxHeight": "70vh",
        "backgroundColor": "#20242b",
        "borderRadius": "18px",
        "boxShadow": "0 2px 24px #2229",
        "padding": "0",
        "zIndex": 1101,
        "transition": "all 0.26s cubic-bezier(.4,1.5,.8,1)",
        "display": "flex",
        "flexDirection": "column",
        "overflowY": "auto",
    }
    return style

# Copilot panel container style synced with open store
@app.callback(
    Output("scanning-copilot-panel", "style"),
    Input("scanning-copilot-open", "data"),
    State("scanning-copilot-panel", "style"),
)
def update_scanning_copilot_panel(is_open, current_style):
    style = dict(current_style) if current_style else {}
    style["display"] = "block" if is_open else "none"
    return style

# Suggestions (update dynamically with context)
@app.callback(
    Output("scanning-copilot-suggestions", "children"),
    [Input("explorer-selected-rows", "data"),
     Input("search-term", "value")]
)
def update_scanning_copilot_suggestions(selected_rows, search_term):
    prompts = get_scanning_copilot_prompts(selected_rows, search_term)
    count = len(selected_rows or [])
    info = f"You currently have {count} driving forces selected."
    buttons = [
        html.Button(
            text,
            n_clicks=0,
            id={"type": "copilot-suggestion", "index": i},
            style={
                "backgroundColor": "#181818",
                "color": "#0af",
                "border": "1.2px solid #333",
                "borderRadius": "14px",
                "padding": "7px 13px",
                "margin": "2px",
                "fontSize": "15px",
                "cursor": "pointer"
            }
        )
        for i, text in enumerate(prompts)
    ]
    return [html.Div(info, style={"color": "#aaa", "fontSize": "13px", "marginRight": "10px"})] + buttons

# Fill input box from suggestion click (single callback handling all suggestion buttons)
@app.callback(
    Output("scanning-copilot-input", "value", allow_duplicate=True),
    Input({"type": "copilot-suggestion", "index": ALL}, "n_clicks"),
    State({"type": "copilot-suggestion", "index": ALL}, "children"),
    prevent_initial_call=True,
)
def handle_copilot_suggestion_click(n_clicks_list, suggestions):
    if not n_clicks_list or sum(n_clicks_list) == 0:
        raise PreventUpdate
    for i, n in enumerate(n_clicks_list):
        if n and i < len(suggestions):
            return suggestions[i]
    raise PreventUpdate

# Chat send callback: handles OpenAI reply, manages history and thread
@app.callback(
    Output("scanning-copilot-history", "data"),
    Output("scanning-copilot-input", "value"),
    Input("scanning-copilot-send-btn", "n_clicks"),
    Input({"type": "copilot-suggestion", "index": ALL}, "n_clicks"),
    State("scanning-copilot-input", "value"),
    State({"type": "copilot-suggestion", "index": ALL}, "children"),
    State("tsne-plot", "selectedData"),
    State("explorer-table", "data"),          # NEW
    State("explorer-selected-rows", "data"),  # NEW
    State("scanning-copilot-thread", "data"),
    State("scanning-copilot-history", "data"),
    prevent_initial_call=True,
)
def _copilot_send(_, __, typed, sugg_texts,
                  selected, table_data, selected_rows,
                  thread_store, history):
    ctx = dash.callback_context
    trig = ctx.triggered_id
    if trig is None:
        raise PreventUpdate

    # prompt text comes either from a clicked suggestion chip or the input box
    if isinstance(trig, dict):
        user_prompt = sugg_texts[trig["index"]]
    else:
        user_prompt = (typed or "").strip()
    if not user_prompt:
        raise PreventUpdate

    # build context from up to 50 selected nodes or table rows
    context = ""
    titles = []

    # 1) titles from nodes clicked in the graph
    if selected and selected.get("points"):
        titles = [p["customdata"][1] for p in selected["points"][:50]]

    # 2) if nothing there, pull titles from table-row selections
    if not titles and table_data and selected_rows:
        for idx in selected_rows[:50]:
            if 0 <= idx < len(table_data):
                titles.append(table_data[idx].get("Title", ""))

    if titles:
        context = "\n\nContext – selected nodes:\n" + "\n".join(f"- {t}" for t in titles)

    # lazily create / reuse an OpenAI thread
    if not thread_store:
        thread = client.beta.threads.create()
        thread_store = {"id": thread.id}
    thread_id = thread_store["id"]

    # send the user prompt (plus selected-nodes context) to the thread
    client.beta.threads.messages.create(
        thread_id, role="user", content=user_prompt + context
    )
    # run the assistant and wait
    client.beta.threads.runs.create_and_poll(
        thread_id=thread_id,
        assistant_id=scanning_assistant_id
    )
    # grab the latest assistant reply
    assistant_msg = client.beta.threads.messages.list(
        thread_id, limit=1
    ).data[0].content[0].text.value

    # store Q-A pair in history (keys match update_chatbox)
    history = (history or []) + [
        {"question": user_prompt, "answer": assistant_msg}
    ]
    return history, ""          # 2nd output clears the input box

# Update chatbox contents from history
@app.callback(
    Output("scanning-copilot-chatbox", "children"),
    Input("scanning-copilot-history", "data")
)
def update_chatbox(history):
    if not history:
        return html.Div(
            "No messages yet. Try a suggestion or ask a question!",
            style={"color": "#888", "padding": "12px"}
        )

    bubbles = []
    for entry in history:
        q = entry.get("question", "")
        a = entry.get("answer", "")

        # assistant bubble (dark, left-aligned)
        bubbles.append(
            html.Div(
                a,
                style={
                    "alignSelf": "flex-start",
                    "maxWidth": "92%",
                    "background": "#2c2c2c",
                    "color": "#eee",
                    "borderRadius": "16px 16px 16px 4px",
                    "padding": "10px 14px",
                    "margin": "2px 0 6px 0",
                    "fontSize": "15px",
                    "whiteSpace": "pre-wrap",   # NEW: keep line-breaks & wrap
                    "wordBreak": "break-word",  # NEW: wrap very long words/URLs
                },
            )
        )
        # user bubble (blue, right-aligned) – add the same two lines
        bubbles.append(
            html.Div(
                q,
                style={
                    "alignSelf": "flex-end",
                    "maxWidth": "92%",
                    "background": "#007aff",
                    "color": "#fff",
                    "borderRadius": "16px 16px 4px 16px",
                    "padding": "10px 14px",
                    "margin": "6px 0 2px 0",
                    "fontSize": "15px",
                    "whiteSpace": "pre-wrap",   # NEW
                    "wordBreak": "break-word",  # NEW
                },
            )
        )

        # wrap bubbles in a flex column that can shrink / scroll
    return html.Div(
        bubbles,
        style={
            "flex": "1 1 0%",          # takes all remaining height
            "minHeight": 0,            # lets flexbox shrink it
            "overflowY": "auto",       # scrolls when content is long
            "display": "flex",
            "flexDirection": "column",
            "gap": "6px",
            "padding": "8px 14px"         
        },
    )
    
    # --- PROJECT MANAGEMENT CALLBACK: Only single, unified callback for all project management UI logic remains. ---
# (All other @app.callback functions with outputs to project management UI components have been removed.)
# --- Place the project creation callback at the end of the file with other callbacks ---
# --- PROJECT CREATION CALLBACK (Unified, no duplicates, proper placement) ---
# Handles Enter/Submit/button, updates dropdown/options/value, resets input, prevents duplicates/empty
@app.callback(
    [
        Output('project-selector', 'options'),
        Output('project-selector', 'value'),
        Output('new-project-name', 'value'),
        Output('new-project-error', 'children')
    ],
    [
        Input('delete-project', 'n_clicks'),
        Input('confirm-new-project', 'n_clicks'),
        Input('new-project-name', 'n_submit')
    ],
    [
        State('project-selector', 'value'),
        State('project-selector', 'options'),
        State('new-project-name', 'value')
    ],
    prevent_initial_call=True
)
def manage_projects(delete_clicks, confirm_clicks, n_submit, current_project, options, new_name):
    import dash
    from dash.exceptions import PreventUpdate
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate
    triggered = ctx.triggered[0]['prop_id'].split('.')[0]
    global projects

    # Delete project branch
    if triggered == 'delete-project':
        if current_project:
            projects.pop(current_project, None)
            if not projects:
                projects["Default Project"] = default_project_state()
            save_projects(projects)
            opts = [
                {'label': html.Span([
                    html.I(className="fa fa-plus", style={"marginRight": "7px", "color": "#0af"}), "+ New Project…"
                ]), 'value': "__new_project__"},
                {'label': html.Span([
                    html.Span("⭐", className="orion-project-star"), "Default Project"
                ], className="orion-project-default"), 'value': "Default Project"}
            ] + [{'label': n, 'value': n} for n in projects if n != "Default Project"]
            new_current = list(projects.keys())[0]
            return opts, new_current, "", ""
        raise PreventUpdate

    # Create project branch (confirm button or Enter)
    if triggered in ('confirm-new-project', 'new-project-name'):
        if not new_name or not new_name.strip():
            return dash.no_update, dash.no_update, "", "Project name cannot be empty."

        name = new_name.strip()
        if name in projects:
            return dash.no_update, dash.no_update, "", "Project already exists."

        projects[name] = default_project_state()
        save_projects(projects)
        opts = [
            {'label': html.Span([
                html.I(className="fa fa-plus", style={"marginRight": "7px", "color": "#0af"}), "+ New Project…"
            ]), 'value': "__new_project__"},
            {'label': html.Span([
                html.Span("⭐", className="orion-project-star"), "Default Project"
            ], className="orion-project-default"), 'value': "Default Project"}
        ] + [{'label': n, 'value': n} for n in projects if n != "Default Project"]

        return opts, name, "", ""

    raise PreventUpdate


# Callback to toggle visibility of rename project error
@app.callback(
    Output('rename-project-error', 'style'),
    [Input('confirm-rename-project', 'n_clicks'),
     Input('rename-project-name', 'n_submit')],
    State('rename-project-error', 'children'),
    prevent_initial_call=True
)
def show_rename_error_style(n_clicks, n_submit, error_text):
    """Show or hide the rename project error based on returned text."""
    if error_text:
        return {'color': '#ff4d4f', 'fontSize': '14px', 'marginTop': '4px'}
    return {'display': 'none'}

##############################################################
# --- CHIP/LOGIC TO QUERY CALLBACK (for search-term) ---
# (This should be the only callback that sets 'search-term', 'value')
@app.callback(
    Output('search-term', 'value'),
    [
        Input('search-chips', 'data'),
        Input('apply-filters-button', 'n_clicks'),
    ],
    State('search-term', 'value')
)
def build_query(chips, n_clicks,current_value):
    # Só reconstrói se houver chips!
    if not chips:
        return ""
    return ' AND '.join([str(c) for c in chips])


# --- UNIFIED CHIP/LOGIC/FILTER MANAGEMENT CALLBACK ---
# Define default_logic at the top-level if not already defined
default_logic = "AND"

@app.callback(
    [
        Output('search-chips', 'data'),
        Output('cluster-highlight', 'value'),
        Output('driving-force-filter', 'value'),
    ],
    [
        Input('reset-filters-button', 'n_clicks'),
        Input('project-selector', 'value'),
        Input('add-chip-btn', 'n_clicks'),
        Input('clear-all-chips', 'n_clicks'),
        Input({'type': 'remove-chip', 'index': ALL}, 'n_clicks'),
    ],
    [
        State('search-chips', 'data'),
        State('logic-dropdown', 'value'),
        State('cluster-highlight', 'value'),
        State('driving-force-filter', 'value'),
        State('project-selector', 'value'),
    ],
    prevent_initial_call=True
)
def unified_chip_logic_filter_callback(
    reset_clicks, project_value, add_chip, clear_all, remove_chip_clicks,
    chips, logic, cluster, driving_force, selected_project
):
    from dash import callback_context as ctx
    triggered = ctx.triggered_id
    chips = chips or []
    default_cluster = -1
    default_driving_force = ["(All)"]
    default_rows = []

    # 1. Reset button
    if triggered == 'reset-filters-button':
        return [], "", default_cluster, default_driving_force

    # 2. Project selector (load project state)
    if triggered == 'project-selector' and selected_project:
        projs = load_projects()
        state = projs.get(selected_project, default_project_state())
        chips = state.get("chips", [])
        cluster = state.get("cluster", default_cluster)
        driving_forces = state.get("driving_forces", default_driving_force)
        return chips, "", cluster, driving_forces

    # 3. Remove chip
    if isinstance(triggered, dict) and triggered.get("type") == "remove-chip":
        idx = triggered.get("index")
        if 0 <= idx < len(chips):
            chips = chips[:idx] + chips[idx+1:]
        return chips, "", cluster, driving_force,

    # 4. Clear all chips
    if triggered == "clear-all-chips":
        return [], "", cluster, driving_force

    # 5. Add chip (button)
    if triggered == "add-chip-btn":
        # No input value passed in, so cannot add new chip directly here
        return chips, "", cluster, driving_force

    # Default (shouldn't happen)
    raise PreventUpdate



# Ensure server start block is at the end of the file, outside any function or conditional
if __name__ == "__main__":
    with server.app_context():
        db.create_all()
    port = int(os.environ.get("PORT", 8050))
    # port = int(os.environ.get("PORT", 8050))
    app.run(host="0.0.0.0", port=port, debug=False)

