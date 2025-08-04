import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss

def analizar_estacionariedad_train(train_csv_path, alpha=0.05):
    df = pd.read_csv(train_csv_path, parse_dates=['fecha'])
    df.set_index('fecha', inplace=True)

    fecha_inicio = df.index.min()
    fecha_fin    = df.index.max()

    resultados = []
    for col in df.columns:
        serie = df[col].dropna()
        if serie.empty or serie.max()==serie.min():
            resultados.append((col,'⚠️ Vacía/constante','','',''))
            continue

        def evalu(ser):
            # ADF
            try:
                p_adf = adfuller(ser)[1]
                r_adf = '✅ Estac.' if p_adf<alpha else '❌ No estac.'
            except:
                r_adf = '⚠️ Error ADF'
            # KPSS
            try:
                p_kpss = kpss(ser, nlags="auto")[1]
                r_kpss = '❌ No estac.' if p_kpss<alpha else '✅ Estac.'
            except:
                r_kpss = '⚠️ Error KPSS'
            return r_adf, r_kpss

        adf0,kpss0 = evalu(serie)
        adf1,kpss1 = evalu(serie.diff().dropna())
        resultados.append((col, adf0, kpss0, adf1, kpss1))

    return resultados, fecha_inicio, fecha_fin
