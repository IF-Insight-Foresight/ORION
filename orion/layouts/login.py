from dash import html, dcc


def layout():
    return html.Div([
        dcc.Input(id='login-email', type='email'),
        dcc.Input(id='login-password', type='password'),
        html.Button('Login', id='login-submit'),
        html.Div(id='login-error'),
    ])
