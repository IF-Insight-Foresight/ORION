import numpy as np
import plotly.graph_objects as go


def generate_tsne_plot(data, dimension='2d'):
    if dimension == '3d':
        fig = go.Figure(go.Scatter3d(
            x=data['tsne_x'], y=data['tsne_y'], z=data['tsne_z'],
            mode='markers', marker=dict(size=6)
        ))
    else:
        fig = go.Figure(go.Scatter(
            x=data['tsne_x'], y=data['tsne_y'],
            mode='markers', marker=dict(size=6)
        ))
    return fig
