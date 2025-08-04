import pandas as pd
import numpy as np

def crear_variables_exogenas_callcenter(
    df: pd.DataFrame,
    target_column: str = None
) -> pd.DataFrame:
    """
    Genera variables exógenas para un DataFrame de call center.
    """
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    exog = pd.DataFrame(index=df.index)

    # Básicas
    exog['day_of_week'] = df.index.dayofweek
    exog['month']       = df.index.month
    exog['hour']        = df.index.hour
    exog['day_of_year'] = df.index.dayof_year
    exog['week_of_year']= df.index.isocalendar().week.astype(int)
    exog['quarter']     = df.index.quarter
    exog['is_quarter_start'] = df.index.is_quarter_start.astype(int)
    exog['is_quarter_end']   = df.index.is_quarter_end.astype(int)

    # Cíclicas
    exog['sin_hour']         = np.sin(2 * np.pi * exog['hour'] / 24)
    exog['cos_hour']         = np.cos(2 * np.pi * exog['hour'] / 24)
    exog['sin_day_of_week']  = np.sin(2 * np.pi * exog['day_of_week'] / 7)
    exog['cos_day_of_week']  = np.cos(2 * np.pi * exog['day_of_week'] / 7)
    exog['sin_month']        = np.sin(2 * np.pi * exog['month'] / 12)
    exog['cos_month']        = np.cos(2 * np.pi * exog['month'] / 12)

    # Feriados (ejemplo simplificado, cargar según país en la ruta)
    # Aquí podrías cargar un dict de feriados y luego:
    # fechas = df.index.normalize()
    # exog['is_holiday'] = fechas.isin(feriados).astype(int)
    # ...

    # Semánticas de negocio
    exog['is_weekend']      = exog['day_of_week'].isin([5,6]).astype(int)
    exog['is_working_hour']= ((exog['hour']>=6)&(exog['hour']<20)).astype(int)
    exog['is_lunch_time']  = ((exog['hour']>=12)&(exog['hour']<14)).astype(int)
    exog['is_night_shift'] = ((exog['hour']>=22)|(exog['hour']<6)).astype(int)

    # Slot de 30min
    exog['time_slot']     = exog['hour']*2 + (df.index.minute//30)
    exog['sin_time_slot']= np.sin(2*np.pi*exog['time_slot']/48)
    exog['cos_time_slot']= np.cos(2*np.pi*exog['time_slot']/48)

    # Última semana de mes
    exog['is_last_week_of_month'] = (
        (df.index.is_month_end) &
        (df.index.day >= df.index.days_in_month - 7)
    ).astype(int)

    return exog.fillna(0)
