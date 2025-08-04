import os
import joblib
import pandas as pd

from skforecast.recursive       import ForecasterRecursive
from skforecast.preprocessing   import RollingFeatures
from skforecast.model_selection import TimeSeriesFold, backtesting_forecaster, bayesian_search_forecaster

from .exogenas_features      import crear_variables_exogenas_callcenter
from .interacciones_exogenas import crear_interacciones_exogenas

def entrenar_exogenas_avanzado(
    series_to_train: list[str],
    lags_grid: list[int],
    n_trials: int,
    pais_feriados: str,
    data_path: str = 'datos',
    model_dir: str = 'modelos'
) -> list[dict]:
    resultados = []
    df_train = pd.read_csv(f"{data_path}/train.csv", parse_dates=['fecha']).set_index('fecha')
    df_val   = pd.read_csv(f"{data_path}/val.csv",   parse_dates=['fecha']).set_index('fecha')
    df_test  = pd.read_csv(f"{data_path}/test.csv",  parse_dates=['fecha']).set_index('fecha')

    for col in series_to_train:
        try:
            y_train = df_train[col]
            y_val   = df_val[col]
            y_test  = df_test[col]
            y_tr_val = pd.concat([y_train, y_val]).sort_index()

            # Exógenas e interacciones
            df_all = pd.concat([y_train, y_val, y_test]).to_frame(col)
            exog   = crear_variables_exogenas_callcenter(df_all, target_column=col)
            exog   = crear_interacciones_exogenas(exog, exog.columns.tolist())

            exog_tr_val = exog.loc[y_tr_val.index]

            # RollingFeatures fijo
            window_size = 96
            window_features = RollingFeatures(stats=["mean"], window_sizes=window_size)

            # Forecaster inicial con lag placeholder
            forecaster = ForecasterRecursive(
                regressor       = None,
                lags            = 48,
                window_features = window_features
            )

            # Backtesting básico
            metric_basic, _ = backtesting_forecaster(
                forecaster=forecaster,
                y=pd.concat([y_tr_val, y_test]),
                exog=exog.loc[pd.concat([y_tr_val, y_test]).index],
                cv=TimeSeriesFold(steps=48, initial_train_size=len(y_tr_val)),
                metric='mean_absolute_error'
            )

            # Búsqueda bayesiana
            def espacio(trial):
                return {
                    'n_estimators' : trial.suggest_int('n_estimators', 300, 1000, step=100),
                    'max_depth'    : trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.5),
                    'reg_alpha'    : trial.suggest_float('reg_alpha', 0, 1),
                    'reg_lambda'   : trial.suggest_float('reg_lambda', 0, 1),
                    'lags'         : trial.suggest_categorical('lags', lags_grid),
                    'num_leaves'   : trial.suggest_int('num_leaves', 15, 60)
                }

            forecaster = ForecasterRecursive(
                regressor       = None,
                lags            = 48,
                window_features = window_features
            )
            cv_search = TimeSeriesFold(steps=48, initial_train_size=len(y_tr_val))
            results_search, _ = bayesian_search_forecaster(
                forecaster   = forecaster,
                y            = y_tr_val,
                exog         = exog_tr_val,
                cv           = cv_search,
                metric       = 'mean_absolute_error',
                search_space = espacio,
                n_trials     = n_trials,
                return_best  = True
            )
            best = results_search.at[0,'params']
            best_lags = best.pop('lags')
            best |= {'random_state':15926, 'verbose':-1}

            # Backtesting con params óptimos
            forecaster = ForecasterRecursive(
                regressor       = type(forecaster.regressor)(**best),
                lags            = best_lags,
                window_features = window_features
            )
            metric_tuned, _ = backtesting_forecaster(
                forecaster=forecaster,
                y=pd.concat([y_tr_val, y_test]),
                exog=exog.loc[pd.concat([y_tr_val, y_test]).index],
                cv=TimeSeriesFold(steps=48, initial_train_size=len(y_tr_val)),
                metric='mean_absolute_error'
            )

            # Guardar modelo entrenado
            os.makedirs(model_dir, exist_ok=True)
            ruta = f"{model_dir}/forecaster_exog_adv_{col}.pkl"
            joblib.dump(forecaster, ruta)

            resultados.append({
                "serie": col,
                "backtest_basic": round(metric_basic,4),
                "backtest_tuned": round(metric_tuned,4),
                "ruta_modelo": ruta
            })

        except Exception as e:
            resultados.append({"serie": col, "error": str(e)})

    return resultados
