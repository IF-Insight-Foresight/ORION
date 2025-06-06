from dash import Output, Input, callback
from .. import app


@callback(Output('info-placeholder', 'children'), Input('info-btn', 'n_clicks'))
def show_info(n):
    if not n:
        return ''
    return f'Info clicked {n}'
