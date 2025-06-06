from .layouts import login as login_page, register as register_page
from dash import html, dcc, callback, Output, Input, State, no_update
from flask_login import login_user
from . import app
from .models import User, db




@callback(
    Output('login-error', 'children'),
    Output('url', 'pathname', allow_duplicate=True),
    Input('login-submit', 'n_clicks'),
    State('login-email', 'value'),
    State('login-password', 'value'),
    prevent_initial_call=True,
)
def perform_login(n_clicks, email, password):
    if not email or not password:
        return 'Missing credentials', no_update
    user = User.query.filter_by(email=email).first()
    if user and user.check_password(password):
        login_user(user)
        return '', '/'  # redirect home
    return 'Invalid credentials', no_update


@callback(
    Output('register-error', 'children'),
    Output('url', 'pathname', allow_duplicate=True),
    Input('register-submit', 'n_clicks'),
    State('register-email', 'value'),
    State('register-password', 'value'),
    prevent_initial_call=True,
)
def perform_register(n_clicks, email, password):
    if not email or not password:
        return 'Missing credentials', no_update
    if User.query.filter_by(email=email).first():
        return 'Email exists', no_update
    user = User(email=email)
    user.set_password(password)
    db.session.add(user)
    db.session.commit()
    login_user(user)
    return '', '/'
