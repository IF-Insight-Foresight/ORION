import os
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from preprocessteste import preprocess_text


def load_data(file_path='ORION_Scanning_DB_Updated.xlsx', features_path='precomputed_features.pkl'):
    data = pd.read_excel(file_path)
    data.columns = data.columns.str.strip()

    data['PreprocessedText'] = (
        data['Title'].fillna('') + ' ' +
        data['Description'].fillna('') + ' ' +
        data['Tags'].fillna('')
    ).apply(preprocess_text)
    data = data[data['PreprocessedText'].str.strip() != '']

    with open(features_path, 'rb') as f:
        feats = pickle.load(f)
    data['tsne_x'] = feats['tsne_x']
    data['tsne_y'] = feats['tsne_y']
    data['Cluster'] = feats['cluster_labels']
    if 'tsne_z' in feats:
        data['tsne_z'] = feats['tsne_z']
    else:
        scaler = StandardScaler()
        data['tsne_z'] = scaler.fit_transform(data[['tsne_x']])[:,0] * 2.2

    return data


vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1, 2))
