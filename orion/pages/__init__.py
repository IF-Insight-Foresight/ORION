"""Page layout and routing callbacks for the ORION dashboard."""

from dash import html, dcc, callback, Output, Input

from .. import app
from ..layouts import login, register, main


def layout():
    """Return the top-level application layout."""
    return html.Div([
        dcc.Location(id="url", refresh=False),
        html.Div(id="page-content")
    ])


@callback(Output("page-content", "children"), Input("url", "pathname"))
def render_page(pathname: str):
    """Render the appropriate page for ``pathname``."""
    if pathname == "/login":
        return login.layout()
    if pathname == "/register":
        return register.layout()
    # Default route
    return main.layout()
