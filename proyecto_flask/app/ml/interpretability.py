import os
import joblib
import warnings
import pandas as pd
import shap
import matplotlib.pyplot as plt
from typing import List, Optional
from skforecast.utils.utils import IndexWarning

from .exogenas_features      import crear_variables_exogenas_callcenter
from .interacciones_exogenas import crear_interacciones_exogenas

# Silenciar warnings de índice
warnings.simplefilter("ignore", category=IndexWarning)

def interpretar_modelos(
    data_path: str,
    model_path: str,
    series_list: List[str],
    exog_semanticas: List[str],
    feriados: Optional[List[pd.Timestamp]] = None,
    sample_frac: float = 0.5
) -> List[dict]:
    """
    Para cada serie:
      1. Carga modelo avanzado (_exog_advanced.pkl).
      2. Reconstruye X_train, y_train con create_train_X_y.
      3. Obtiene feature_importances.
      4. Calcula SHAP summary (50% sample) y retorna HTML.
      5. Permite waterfall/forceplot por fecha (dejamos placeholder).
    """
    resultados = []
    for serie in series_list:
        model_file = os.path.join(model_path, f"{serie}_exog_advanced.pkl")
        if not os.path.exists(model_file):
            continue

        # 1) Carga modelo y datos
        fore = joblib.load(model_file)
        train = pd.read_csv(os.path.join(data_path,'train.csv'),
                            parse_dates=['fecha']).set_index('fecha')
        df_full = train[serie].to_frame(serie)
        exog = crear_variables_exogenas_callcenter(df_full, feriados=feriados)
        exog = crear_interacciones_exogenas(exog, exog_semanticas)

        # 2) Genera X_train, y_train
        X_train, y_train = fore.create_train_X_y(
            y    = train[serie].dropna(),
            exog = exog.reindex(train.index).dropna()
        )

        # 3) Feature importances
        fi = fore.get_feature_importances()
        fi_tabla = fi.head(10).to_frame('importance').to_html()

        # 4) SHAP summary
        shap.initjs()
        explainer = shap.TreeExplainer(fore.regressor)
        idx_sample = (X_train.sample(frac=sample_frac, random_state=0)).index
        X_samp = X_train.loc[idx_sample]
        shap_vals = explainer.shap_values(X_samp)
        plt.clf()
        shap.summary_plot(shap_vals, X_samp, max_display=10, show=False)
        fig = plt.gcf()
        fig.set_size_inches(6,4)
        summary_html = mpl_to_html(fig)  # función auxiliar para HTML

        resultados.append({
            "serie":       serie,
            "fi_html":     fi_tabla,
            "shap_summary": summary_html,
            # Para waterfall/forceplot habrá endpoint separado
        })

    return resultados

def mpl_to_html(fig):
    """
    Convierte plt.gcf() a HTML <img> en base64.
    """
    import io, base64
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    return f'<img src="data:image/png;base64,{img_b64}" />'
