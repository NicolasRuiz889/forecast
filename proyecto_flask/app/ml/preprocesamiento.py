# app/ml/preprocesamiento.py
import os
import pandas as pd

def limpiar_excel(input_path: str, output_dir: str) -> str:
    """
    Lee el Excel en input_path, aplica el preprocesamiento y guarda
    datos/output.csv en output_dir. Devuelve la ruta del CSV generado.
    """
    # 1) Validar existencia
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"No se encontró el archivo: {input_path}")

    # 2) Asegurar carpeta de salida
    os.makedirs(output_dir, exist_ok=True)
    ruta_salida = os.path.join(output_dir, 'output.csv')

    # 3) Leer y procesar
    df = pd.read_excel(input_path)

    # Cortar antes de 'Total'
    mask_total = df['DAY - Year'] == 'Total'
    idx_total  = mask_total[mask_total].index[0]
    df_trim    = df.iloc[:idx_total].copy()

    # Eliminar filas con 'Total' en otras columnas
    df_clean = df_trim[
        (df_trim['DAY - Month'] != 'Total') &
        (df_trim['DAY - Day']   != 'Total') &
        (df_trim['Hour']        != 'Total')
    ].copy()

    # Construir datetime
    df_clean['Hour_str'] = df_clean['Hour'].astype(str).str[:5]
    fecha_str = (
        df_clean['DAY - Year'].astype(int).astype(str) + '-' +
        df_clean['DAY - Month'] + '-' +
        df_clean['DAY - Day'].astype(int).astype(str) + ' ' +
        df_clean['Hour_str']
    )
    df_clean['fecha'] = pd.to_datetime(
        fecha_str, format='%Y-%B-%d %H:%M', errors='coerce'
    )
    df_clean.drop(columns=['Hour_str'], inplace=True)

    # Cálculos finales
    df_clean['Abandoned'] = df_clean['Offered'] - df_clean['Handled']
    df_clean.rename(columns={
        'ABA %': 'Real ABA%',
        'TSF'  : 'Real TSF',
        'ASA'  : 'Real ASA'
    }, inplace=True)

    columnas_finales = [
        'fecha','Offered','Handled','Abandoned',
        'Real ABA%','Real TSF','Real ASA','Calls SL'
    ]
    df_final = df_clean[columnas_finales].copy()
    df_final.fillna(0, inplace=True)
    for c in df_final.columns:
        if c != 'fecha':
            df_final[c] = df_final[c].astype(float)
    df_final[['Real ABA%','Real TSF']] = (
        df_final[['Real ABA%','Real TSF']] * 100
    ).round(1)

    # 4) Guardar CSV
    df_final.to_csv(ruta_salida, index=False)
    return ruta_salida

