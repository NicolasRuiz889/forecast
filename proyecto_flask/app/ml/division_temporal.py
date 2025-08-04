# app/ml/division_temporal.py

import pandas as pd
from dateutil.relativedelta import relativedelta

def dividir_ultimos_n_meses(df, col: str, meses: int):
    """
    Toma el DataFrame df y la columna col, y extrae una ventana de los últimos
    `meses` meses completos (empezando el primer día de mes N meses atrás),
    luego la divide en 80% train, 10% val y 10% test basándose en días.
    Devuelve (train, val, test, info).
    """
    # 1) Asegurar índice datetime
    serie = df[col].dropna()
    if not pd.api.types.is_datetime64_any_dtype(serie.index):
        serie.index = pd.to_datetime(serie.index)

    # 2) Último timestamp disponible
    t_end = serie.index.max()

    # 3) Primer día del mes actual y restar meses
    primer_dia_mes_actual = pd.Timestamp(t_end.year, t_end.month, 1)
    periodo_inicio       = primer_dia_mes_actual - relativedelta(months=meses)

    # 4) Encontrar el primer índice >= periodo_inicio
    candidatos_inicio = serie.index[serie.index >= periodo_inicio]
    if candidatos_inicio.empty:
        raise ValueError(f"No hay datos desde {periodo_inicio.date()}")
    t_start = candidatos_inicio[0]

    # 5) Ventana completa
    ventana = df.loc[t_start:t_end].copy()

    # 6) Cálculo de días y tamaños de partición
    total_dias = (ventana.index.max() - ventana.index.min()).days
    dias_train = int(total_dias * 0.8)
    dias_val   = int(total_dias * 0.1)
    # El resto al test
    dias_test  = total_dias - dias_train - dias_val

    # Helpers para alinear a 23:30 y 00:00
    def ultimo_2330(idx, limite):
        cand = idx[(idx <= limite) & (idx.strftime('%H:%M') == '23:30')]
        return cand[-1] if not cand.empty else limite

    def primero_0000(idx, inicio):
        cand = idx[(idx >= inicio) & (idx.strftime('%H:%M') == '00:00')]
        return cand[0] if not cand.empty else inicio

    # 7) Definir cortes
    t_train_start = primero_0000(ventana.index, t_start)
    t_train_end   = ultimo_2330 (ventana.index, t_start + pd.Timedelta(days=dias_train))
    t_val_start   = primero_0000(ventana.index, t_train_end + pd.Timedelta(minutes=30))
    t_val_end     = ultimo_2330 (ventana.index, t_start + pd.Timedelta(days=dias_train + dias_val))
    t_test_start  = primero_0000(ventana.index, t_val_end + pd.Timedelta(minutes=30))
    t_test_end    = t_end

    # 8) Extraer particiones
    train = df.loc[t_train_start:t_train_end].copy()
    val   = df.loc[t_val_start:  t_val_end  ].copy()
    test  = df.loc[t_test_start: t_test_end ].copy()

    # 9) Construir info
    rango = df.loc[t_train_start:t_test_end].copy()
    # Sólo inferimos la frecuencia para reportarla; no la asignamos al índice
    freq_inferida = pd.infer_freq(rango.index)
    frecuencia = freq_inferida if freq_inferida is not None else "30min"

    info = {
        'train':       (t_train_start, t_train_end),
        'val':         (t_val_start,   t_val_end),
        'test':        (t_test_start,  t_test_end),
        'frecuencia':  frecuencia,
        'index_type':  type(rango.index),
        'index_inicio':rango.index.min(),
        'index_fin':   rango.index.max(),
        'tipos':       rango.dtypes.astype(str).to_dict(),
        'tipo_df':     str(type(rango))
    }

    return train, val, test, info
