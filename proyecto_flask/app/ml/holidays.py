# app/ml/holidays.py

import holidays
from datetime import date
from typing import List

# Mapa nombre de país → código ISO para la librería `holidays`
COUNTRY_CODES = {
    'Chile':                'CL',
    'Panama':               'PA',
    'Costa Rica':           'CR',
    'Barbados':             'BB',
    'Jamaica':              'JM',
    'Canada':               'CA',
    'República Dominicana': 'DO',
    'Perú':                 'PE',
    'Uruguay':              'UY',
    'Trinidad y Tobago':    'TT',
    'Bahamas':              'BS'
}

def get_holidays_for_country(
    country_name: str,
    start_year: int = 2021,
    end_year:   int = 2032
) -> List[date]:
    """
    Devuelve una lista ordenada de fechas (`datetime.date`)
    con todos los feriados oficiales de `country_name`
    entre `start_year` y `end_year` (inclusive).
    """
    code = COUNTRY_CODES.get(country_name)
    if not code:
        raise ValueError(f"País desconocido: {country_name}")
    # Instancia el objeto de feriados
    hs = holidays.CountryHoliday(code, years=range(start_year, end_year+1))
    return sorted(hs.keys())
