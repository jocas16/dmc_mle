# pycaret_automl_mlflow.py
import mlflow
from pycaret.classification import *
import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore")

# Ruta absoluta al CSV
data = pd.read_csv("hbr_caso_cliente_responde_oferta.csv")

# Configuración de MLflow
mlflow.set_tracking_uri("file:C:\PP\dmc_mle\taller03\mlruns")
mlflow.set_experiment("Answer_Prediction_T1")

with mlflow.start_run(run_name="AutoML_pycaret_T1"):
    # Setup y entrenamiento
    s = setup(data, target='respondio_oferta', session_id=131)
    
    best = compare_models()
    evaluate_model(best)
    
    # Registrar modelo
    mlflow.sklearn.log_model(best, "mejor_modelo_T3V2")
    
    # Extra logs si deseas
    mlflow.log_param("modelo_principal_T3V2", str(best))
    
    # Registrar matriz de confusión como imagen
    import matplotlib.pyplot as plt
    from pycaret.utils import check_metric
    from pycaret.classification import plot_model
    
    plot_model(best, plot='confusion_matrix', save=True)
    mlflow.log_artifact("Confusion Matrix.png")