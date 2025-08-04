import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

def crear_interacciones_exogenas(
    exog: pd.DataFrame,
    columnas_interaccion: list[str],
    degree: int = 2
) -> pd.DataFrame:
    """
    Genera interacciones polinómicas (sin bias) de las columnas indicadas.
    """
    # Filtrar sólo columnas existentes
    cols = [c for c in columnas_interaccion if c in exog.columns]
    if not cols:
        return exog

    transformer = PolynomialFeatures(
        degree=degree,
        interaction_only=True,
        include_bias=False
    )
    transformer.set_output(transform="pandas")

    df_poly = transformer.fit_transform(exog[cols])
    # Eliminar columnas originales si aparecieron duplicadas
    df_poly = df_poly.drop(columns=cols, errors='ignore')
    df_poly.columns = [f"poly__{c.replace(' ','__')}" for c in df_poly.columns]
    df_poly.index = exog.index

    return pd.concat([exog, df_poly], axis=1)
