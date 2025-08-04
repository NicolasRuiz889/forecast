import os
import joblib
import pandas as pd
import numpy as np
from collections import OrderedDict
from datetime import datetime, timedelta
from typing import List, Optional

from skforecast.recursive     import ForecasterRecursive
from skforecast.preprocessing import RollingFeatures

from .exogenas_features      import crear_variables_exogenas_callcenter
from .interacciones_exogenas import crear_interacciones_exogenas

def predecir_futuro(
    data_path: str,
    model_path: str,
    serie: str,
    feriados: Optional[List[pd.Timestamp]],
    exog_semanticas: List[str],
    lags_select: List[int],
    window_size: int,
    dias_a_predecir: int
) -> dict:
    """
    Genera predicciones intradía para los próximos `dias_a_predecir` días
    de la serie `serie`, usando el modelo avanzado ya guardado.
    Devuelve:
      - serie_predicha (pd.Series)
      - volumen_diario (pd.Series)
      - rutas de archivos Excel
    """
    # 1) Carga modelo
    modelo_file = os.path.join(model_path, f"{serie}_exog_advanced.pkl")
    forecaster = joblib.load(modelo_file)

    # 2) Rango futuro de fechas (30 min) para `dias_a_predecir`
    ultima_fecha = pd.read_csv(os.path.join(data_path,'test.csv'),
                               parse_dates=['fecha']).set_index('fecha')[serie].dropna().index.max()
    inicio = ultima_fecha + pd.Timedelta(minutes=30)
    fin    = inicio + pd.Timedelta(days=dias_a_predecir) - pd.Timedelta(minutes=30)
    fechas_nuevas = pd.date_range(start=inicio, end=fin, freq='30min')

    df_nuevo = pd.DataFrame(index=fechas_nuevas)

    # 3) Crear variables exógenas
    exog_base = crear_variables_exogenas_callcenter(
        df_nuevo, feriados=feriados
    )
    # 4) Crear interacciones polinómicas
    exog_full = crear_interacciones_exogenas(
        exog_base,
        columnas_interaccion=exog_semanticas
    )

    # 5) Filtrar solo las features que ya usó el modelo
    #    (las guardamos previamente en el pipeline; aquí asumimos que tenemos esa lista)
    exog_features = forecaster.exog_features  # atributo que guardaste al entrenar
    exog_final = exog_full.reindex(columns=exog_features).ffill().bfill()

    # 6) Predecir
    steps = len(exog_final)
    serie_predicha = forecaster.predict(steps=steps, exog=exog_final)
    serie_predicha = pd.Series(serie_predicha, index=fechas_nuevas, name=serie)

    # 7) Exportar a Excel
    downloads = os.path.expanduser("~/Downloads")
    fecha_str = datetime.now().strftime("%Y%m%d_%H%M")
    fn1 = f"{serie}_prediccion_{fecha_str}.xlsx"
    path1 = os.path.join(downloads, fn1)
    df_pred = serie_predicha.reset_index()
    df_pred.columns = ['Fecha', f'Prediccion_{serie}']
    df_pred.to_excel(path1, index=False)

    # 8) Volumen diario
    volumen_diario = serie_predicha.resample('D').sum()
    fn2 = f"{serie}_volumen_diario_{fecha_str}.xlsx"
    path2 = os.path.join(downloads, fn2)
    df_vol = volumen_diario.reset_index()
    df_vol.columns = ['Fecha', f'Volumen_Diario_{serie}']
    df_vol.to_excel(path2, index=False)

    return {
        "serie_predicha":   serie_predicha,
        "volumen_diario":   volumen_diario,
        "xlsx_prediccion":  fn1,
        "xlsx_volumen":     fn2
    }
