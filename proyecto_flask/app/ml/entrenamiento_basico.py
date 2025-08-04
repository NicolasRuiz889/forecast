# app/ml/entrenamiento_basico.py

import os
import joblib
import pandas as pd
import plotly.graph_objs as go
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tools.sm_exceptions import InterpolationWarning
import warnings

from skforecast.recursive        import ForecasterRecursive
from skforecast.preprocessing    import RollingFeatures
from skforecast.model_selection import TimeSeriesFold, backtesting_forecaster
from lightgbm import LGBMRegressor

# Suprimir warnings de KPSS y utilitarios de Skforecast
warnings.simplefilter("ignore", category=InterpolationWarning)

def entrenar_modelo_basico(data_path: str,
                          model_path: str,
                          alpha: float = 0.05):
    """
    1) Filtra cada serie de `train.csv` con ADF (p<alpha) y KPSS (p>=alpha).
    2) Para las estacionarias:
       - Concatena train+val, infiere o calcula freq y hace asfreq().
       - Rellena los huecos con ffill().bfill().
       - Entrena ForecasterRecursive (lags=48, ventana semanal).
       - Backtesting con TimeSeriesFold.
       - Alinea predicciones con test.index.
       - Grafica real vs predicción.
       - Guarda modelo en model_path.
    Devuelve lista de resultados.
    """
    # 1) Carga particiones
    train = pd.read_csv(
        os.path.join(data_path, 'train.csv'),
        parse_dates=['fecha']
    ).set_index('fecha')
    val   = pd.read_csv(
        os.path.join(data_path, 'val.csv'),
        parse_dates=['fecha']
    ).set_index('fecha')
    test  = pd.read_csv(
        os.path.join(data_path, 'test.csv'),
        parse_dates=['fecha']
    ).set_index('fecha')

    os.makedirs(model_path, exist_ok=True)
    resultados = []

    for col in train.columns:
        y_train = train[col].dropna()

        # Saltar vacías o constantes
        if y_train.empty or y_train.max() == y_train.min():
            resultados.append({
                "serie": col,
                "estado": "⚠️ Vacía/constante",
                "mae": None,
                "grafico_html": None,
                "modelo_guardado": None
            })
            continue

        # ADF & KPSS
        p_adf  = adfuller(y_train)[1] if len(y_train) > 0 else 1.0
        p_kpss = kpss(y_train, nlags="auto")[1] if len(y_train) > 0 else 0.0

        # Filtrar no estacionarias
        if not (p_adf < alpha and p_kpss >= alpha):
            resultados.append({
                "serie": col,
                "estado": "❌ No estacionaria",
                "mae": None,
                "grafico_html": None,
                "modelo_guardado": None
            })
            continue

        # 2) Concatena train + val + test
        y_val       = val[col].dropna()
        y_test      = test[col].dropna()
        y_train_val = pd.concat([y_train, y_val]).sort_index()
        y_total     = pd.concat([y_train_val, y_test]).sort_index()

        # 3) Inferir o calcular frecuencia
        freq = pd.infer_freq(y_train_val.index)
        if freq is None and len(y_train_val.index) > 1:
            delta = y_train_val.index[1] - y_train_val.index[0]
            freq = pd.tseries.frequencies.to_offset(delta)
        else:
            freq = pd.tseries.frequencies.to_offset(freq)

        # 4) Forzar índice con freq y rellenar huecos
        y_train_val = y_train_val.asfreq(freq).ffill().bfill()
        y_total     = y_total.asfreq(freq).ffill().bfill()

        # 5) Crear y entrenar ForecasterRecursive
        window_feats = RollingFeatures(stats=["mean"], window_sizes=[48 * 7])
        forecaster   = ForecasterRecursive(
            regressor       = LGBMRegressor(random_state=15926, verbose=-1),
            lags            = 48,
            window_features = window_feats
        )
        forecaster.fit(y=y_train_val)

        # 6) Backtesting
        cv = TimeSeriesFold(
            steps              = len(y_test),
            initial_train_size = len(y_train_val),
            refit              = False
        )
        metric, predictions = backtesting_forecaster(
            forecaster = forecaster,
            y          = y_total,
            cv         = cv,
            metric     = 'mean_absolute_error',
            verbose    = False
        )

        # 7) Alinear predicciones con test.index
        n_pred = len(predictions)
        aligned_index = y_test.index[:n_pred]
        predictions.index = aligned_index

        # 8) Graficar real vs predicción
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x    = y_test.index,
            y    = y_test,
            mode = 'lines',
            name = 'Real'
        ))
        fig.add_trace(go.Scatter(
            x    = predictions.index,
            y    = predictions['pred'],
            mode = 'lines',
            name = 'Predicción'
        ))
        fig.update_layout(
            title       = f'Pred vs Real — {col}',
            xaxis_title = 'Fecha',
            yaxis_title = col,
            width       = 900,
            height      = 450,
            legend      = dict(orientation="h", y=1.02, x=0),
            margin      = dict(l=40, r=40, t=40, b=40)
        )

        # 9) Guardar el modelo
        modelo_file = os.path.join(model_path, f"{col.replace(' ', '_')}.pkl")
        joblib.dump(forecaster, modelo_file)

        # 10) Registrar resultado
        resultados.append({
            "serie": col,
            "estado": "✅ Entrenado",
            "mae": round(metric, 4),
            "grafico_html": fig.to_html(full_html=False),
            "modelo_guardado": modelo_file
        })

    return resultados



