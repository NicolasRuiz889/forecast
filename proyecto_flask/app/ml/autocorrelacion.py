import pandas as pd
from statsmodels.tsa.stattools import acf, pacf

def calcular_autocorrelaciones_train(
        train_csv_path,
        n_lags=120,
        orden_diff=1,
        sort_by="partial_autocorrelation_abs"
    ):
    df = pd.read_csv(train_csv_path, parse_dates=['fecha'])
    df.set_index('fecha', inplace=True)
    resultados = {}

    for col in df.columns:
        serie = df[col].dropna()
        if serie.empty or serie.max()==serie.min():
            continue
        s = serie.copy()
        if orden_diff>0:
            s = s.diff(periods=orden_diff).dropna()

        acf_vals  = acf(s, nlags=n_lags)
        pacf_vals = pacf(s, nlags=n_lags, method='ywm')
        df_lags = pd.DataFrame({
            "lag": range(1, len(acf_vals)),
            "autocorrelation": acf_vals[1:],
            "partial_autocorrelation": pacf_vals[1:]
        })
        df_lags["autocorrelation_abs"]       = df_lags["autocorrelation"].abs()
        df_lags["partial_autocorrelation_abs"]= df_lags["partial_autocorrelation"].abs()
        if sort_by in df_lags:
            df_lags = df_lags.sort_values(by=sort_by, ascending=False)
        resultados[col] = df_lags.reset_index(drop=True)

    return resultados
