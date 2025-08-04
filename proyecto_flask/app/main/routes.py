import os
import joblib
from flask import Blueprint, render_template, request, current_app

from ..ml.preprocesamiento import limpiar_excel
from ..ml.division_temporal import dividir_ultimos_n_meses
from ..ml.estacionariedad import analizar_estacionariedad_train
from ..ml.autocorrelacion import calcular_autocorrelaciones_train
from ..ml.entrenamiento_basico import entrenar_modelo_basico
from ..ml.entrenamiento_exogenas_avanzado import entrenar_exogenas_avanzado
from ..ml.future_forecast import predecir_futuro


main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def home():
    modelos_dir = current_app.config.get('MODEL_PATH', 'modelos')
    model_list = []
    if os.path.isdir(modelos_dir):
        for fn in os.listdir(modelos_dir):
            if fn.startswith('forecaster_exog_adv_') and fn.endswith('.pkl'):
                serie = fn.replace('forecaster_exog_adv_','').replace('.pkl','')
                model_list.append(serie)
    return render_template('index.html', model_list=model_list)

@main_bp.route('/limpiar', methods=['POST'])
def limpiar():
    ruta = limpiar_excel()
    mensaje = f"✅ Datos limpiados y guardados en: {ruta}"
    return render_template('index.html', mensaje=mensaje)

@main_bp.route('/estacionariedad', methods=['POST'])
def estacionariedad():
    resultados, inicio, fin = analizar_estacionariedad_train()
    rango = f"{inicio.date()} → {fin.date()}"
    return render_template('index.html',
                           estacionariedad=resultados,
                           rango_estacionariedad=rango)

@main_bp.route('/dividir', methods=['POST'])
def dividir():
    meses = int(request.form['meses'])
    train, val, test, info = dividir_ultimos_n_meses(
        pd.read_csv('datos/output.csv', parse_dates=['fecha']).set_index('fecha'),
        col='Handled', meses=meses
    )
    return render_template('index.html', info_division=info)

@main_bp.route('/autocorrelacion', methods=['POST'])
def autocorrelacion():
    orden = int(request.form['orden_diff'])
    lags  = int(request.form['n_lags'])
    resultados = calcular_autocorrelaciones_train(orden_diff=orden, n_lags=lags)
    return render_template('index.html', autocorrelaciones=resultados)

@main_bp.route('/entrenar_basico', methods=['POST'])
def entrenar_basico():
    window = int(request.form['window_size'])
    resultados = entrenar_modelo_basico(window_size=window)
    return render_template('index.html', resultados_basico=resultados)

@main_bp.route('/entrenar_exogenas_avanzado', methods=['POST'])
def exog_adv():
    series = request.form.getlist('series_to_train')
    lags   = [int(x) for x in request.form['lags_grid'].split(',') if x]
    trials = int(request.form['n_trials'])
    pais   = request.form['pais_feriados']
    resultados = entrenar_exogenas_avanzado(
        series_to_train=series,
        lags_grid=lags,
        n_trials=trials,
        pais_feriados=pais,
        data_path='datos',
        model_dir=current_app.config.get('MODEL_PATH','modelos')
    )
    return render_template('index.html',
                           resultados_exogenas_avanzado=resultados)

@main_bp.route('/pronostico_probabilistico', methods=['POST'])
def pronostico_probabilistico():
    serie = request.form['serie']
    ruta  = os.path.join(current_app.config.get('MODEL_PATH','modelos'),
                         f"forecaster_exog_adv_{serie}.pkl")
    forecaster = joblib.load(ruta)
    # acá llamarías tu función de pronóstico probabilístico, 
    # pasándole `forecaster` y devolviendo `resultados_pronostico`
    resultados_pr = ...  
    return render_template('index.html', resultados_pronostico=resultados_pr)

@main_bp.route('/interpretabilidad', methods=['POST'])
def interpretabilidad():
    serie = request.form['serie']
    ruta  = os.path.join(current_app.config.get('MODEL_PATH','modelos'),
                         f"forecaster_exog_adv_{serie}.pkl")
    forecaster = joblib.load(ruta)
    # tu función de interpretabilidad:
    resultados_int = ...
    return render_template('index.html', resultados_interpretabilidad=resultados_int)

@main_bp.route('/predecir_futuro', methods=['POST'])
def futuro_route():
    serie = request.form['serie']
    dias  = int(request.form['dias_predecir'])
    res = predecir_futuro(
        data_path=current_app.config.get('DATA_PATH','datos'),
        model_path=current_app.config.get('MODEL_PATH','modelos'),
        serie=serie,
        feriados=None,            # o pasa tu lista de feriados
        exog_semanticas=None,     # no es necesario: el forecaster ya las lleva
        lags_select=None,
        window_size=None,
        dias_a_predecir=dias
    )
    return render_template('index.html',
                           serie_futuro=res['serie_predicha'],
                           volumen_futuro=res['volumen_diario'],
                           descarga_prediccion=res['xlsx_prediccion'],
                           descarga_volumen=res['xlsx_volumen'])
