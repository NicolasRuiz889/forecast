import os
import joblib
import warnings
import pandas as pd
import plotly.graph_objs as go
from typing import List, Optional
from skforecast.model_selection import (
    TimeSeriesFold,
    backtesting_forecaster
)
from skforecast.utils.utils import IndexWarning
from skforecast.metrics import calculate_coverage

from .exogenas_features import crear_variables_exogenas_callcenter
from .interacciones_exogenas import crear_interacciones_exogenas

# Suprimir warnings molestos
warnings.simplefilter("ignore", category=IndexWarning)
warnings.filterwarnings("ignore", message="X does not have valid feature names.*")

def pronostico_probabilistico(
    data_path: str,
    model_path: str,
    series_list: List[str],
    exog_semanticas: List[str],
    feriados: Optional[List[pd.Timestamp]] = None,
    interval: List[int]       = [5,95],
    steps: int                = 48
) -> List[dict]:
    """
    Para cada `serie` en series_list:
      1. Carga modelo avanzado (exog_advanced.pkl).
      2. Genera exógenas con feriados y semánticas.
      3. Fit con store_in_sample_residuals=True.
      4. Predict_interval sobre validación+test.
      5. Backtest en train+val para obtener residuales.
      6. set_out_sample_residuals(...).
      7. Backtest final con intervalos.
      8. Calcula cobertura y area.
    Devuelve lista de dicts con métricas y plot HTML.
    """
    resultados = []
    for serie in series_list:
        modelo_file = os.path.join(model_path, f"{serie}_exog_advanced.pkl")
        if not os.path.exists(modelo_file):
            continue

        # 1) Carga datos
        train = pd.read_csv(os.path.join(data_path,'train.csv'),
                            parse_dates=['fecha']).set_index('fecha')
        val   = pd.read_csv(os.path.join(data_path,'val.csv'),
                            parse_dates=['fecha']).set_index('fecha')
        test  = pd.read_csv(os.path.join(data_path,'test.csv'),
                            parse_dates=['fecha']).set_index('fecha')

        # 2) Carga modelo
        fore = joblib.load(modelo_file)

        # 3) Construir df_full y exógenas
        df_full = pd.concat([train[serie], val[serie], test[serie]]).to_frame(serie)
        exog = crear_variables_exogenas_callcenter(df_full, feriados=feriados)
        exog = crear_interacciones_exogenas(exog, exog_semanticas)

        # Particiones
        y_train = train[serie].dropna()
        X_train = exog.reindex(y_train.index)
        y_val_test = pd.concat([val[serie], test[serie]]).dropna()
        X_val_test = exog.reindex(y_val_test.index)

        # Inferir freq + rellenar
        def asfreq_ffill(obj):
            obj = obj.copy()
            obj.index = pd.to_datetime(obj.index)
            freq = pd.infer_freq(obj.index) or (obj.index[1]-obj.index[0])
            obj.index = pd.DatetimeIndex(obj.index).to_period(freq).to_timestamp()
            return obj.asfreq(freq).ffill().bfill()
        y_train, X_train = asfreq_ffill(y_train), asfreq_ffill(X_train)
        y_val_test, X_val_test = asfreq_ffill(y_val_test), asfreq_ffill(X_val_test)
        df_full = asfreq_ffill(df_full)
        exog    = asfreq_ffill(exog)

        # 4) Fit con residuales in-sample
        fore.fit(
            y=y_train,
            exog=X_train,
            store_in_sample_residuals=True
        )

        # 5) Predict intervals en validación+test
        pred_int = fore.predict_interval(
            exog     = X_val_test,
            steps    = steps,
            interval = interval,
            method   = 'conformal'
        )

        # 6) Backtest en train+val para out-sample residuals
        tv = pd.concat([y_train, val[serie].dropna()])
        X_tv = exog.reindex(tv.index)
        cv_tv = TimeSeriesFold(steps=len(val), initial_train_size=len(y_train), refit=False)
        _, pred_tv = backtesting_forecaster(
            forecaster              = fore,
            y                        = tv,
            exog                     = X_tv,
            cv                       = cv_tv,
            metric                   = 'mean_absolute_error',
            return_predictors        = False
        )
        resid_os = tv.loc[pred_tv.index] - pred_tv['pred']

        # 7) Almacenar residuales out-sample
        fore.set_out_sample_residuals(
            y_true = tv.loc[pred_tv.index],
            y_pred = pred_tv['pred']
        )

        # 8) Backtest final con intervalos en test completo
        all_y = pd.concat([y_train, val[serie].dropna(), test[serie].dropna()])
        all_exog = exog.reindex(all_y.index)
        cv_full = TimeSeriesFold(
            steps=len(test), initial_train_size=len(tv), refit=False
        )
        m_final, pred_final = backtesting_forecaster(
            forecaster               = fore,
            y                         = all_y,
            exog                      = all_exog,
            cv                        = cv_full,
            metric                    = 'mean_absolute_error',
            interval                  = interval,
            interval_method           = 'conformal',
            use_in_sample_residuals   = False,
            use_binned_residuals      = True
        )

        # 9) Calcular cobertura y área
        coverage = calculate_coverage(
            y_true      = test[serie].dropna(),
            lower_bound = pred_final['lower_bound'],
            upper_bound = pred_final['upper_bound']
        )
        area = (pred_final['upper_bound'] - pred_final['lower_bound']).sum()

        # 10) Gráfico de intervalos
        fig = go.Figure([
            go.Scatter(name='Pred', x=pred_final.index, y=pred_final['pred'], mode='lines'),
            go.Scatter(name='Real', x=test[serie].dropna().index,
                       y=test[serie].dropna(), mode='lines'),
            go.Scatter(name='Upper', x=pred_final.index,
                       y=pred_final['upper_bound'], mode='lines',
                       line=dict(width=0), showlegend=False),
            go.Scatter(name='Lower', x=pred_final.index,
                       y=pred_final['lower_bound'], mode='lines',
                       fill='tonexty', fillcolor='rgba(0,0,0,0.1)',
                       line=dict(width=0), showlegend=False),
        ])
        fig.update_layout(title=f'Intervalos — {serie}')

        resultados.append({
            "serie":             serie,
            "mae_final":         round(m_final,4),
            "coverage_pct":      round(100*coverage,2),
            "interval_area":     round(area,2),
            "plot_interval":     fig.to_html(full_html=False)
        })

    return resultados
