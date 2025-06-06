from orion import app, server, db
import orion.callbacks  # noqa: F401 ensures callbacks are registered

if __name__ == '__main__':
    with server.app_context():
        db.create_all()
    app.run(host='0.0.0.0', port=8050, debug=False)
