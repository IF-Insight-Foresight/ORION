from dash import html, dcc


def layout():
    return html.Div([
        dcc.Input(id='register-email', type='email'),
        dcc.Input(id='register-password', type='password'),
        html.Button('Register', id='register-submit'),
        html.Div(id='register-error'),
    ])
