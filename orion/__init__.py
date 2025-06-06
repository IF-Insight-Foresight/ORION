import dash_bootstrap_components as dbc
import dash
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager

server = Flask(__name__)
server.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///orion_users.db'
server.config['SECRET_KEY'] = 'orion_secret_key_123'

db = SQLAlchemy(server)

login_manager = LoginManager()
login_manager.init_app(server)

app = dash.Dash(__name__, server=server, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.config.suppress_callback_exceptions = True

__all__ = ['app', 'server', 'db', 'login_manager']
