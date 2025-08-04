# app/main/__init__.py
from flask import Blueprint

main_bp = Blueprint('main', __name__)

# Importa routes para que las rutas se registren al cargar el paquete
from . import routes

