# proyecto_flask/app/config.py
import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'cambiame-por-instancia')
    DATA_PATH  = os.path.join(os.getcwd(), 'datos')
    MODEL_PATH = os.path.join(os.getcwd(), 'modelos')
    DEBUG      = True

