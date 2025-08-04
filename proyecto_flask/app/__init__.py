# app/__init__.py

import os
from flask import Flask
from .config import Config
from .main.routes import main_bp

def create_app():
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_object(Config)
    app.config.from_pyfile('config.py', silent=True)
    app.register_blueprint(main_bp)

    @app.context_processor
    def inject_series_list():
        """Inyecta `series_list` en todas las plantillas."""
        # Busca todos los .pkl básicos (sin sufijo _exog_advanced)
        model_dir = app.config.get('MODEL_PATH', 'modelos')
        try:
            files = [f for f in os.listdir(model_dir)
                     if f.endswith('.pkl') and not f.endswith('_exog_advanced.pkl')]
            series_list = [os.path.splitext(f)[0] for f in files]
        except Exception:
            series_list = []
        return dict(series_list=series_list)

    return app

