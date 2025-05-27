# pycaret_automl_mlflow.py
import mlflow
from pycaret.classification import *
import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore")

# Ruta absoluta al CSV
data_path = "/home/brangovich_dmc/data/credit_data.csv"
data = pd.read_csv(data_path)

# Configuración de MLflow
mlflow.set_tracking_uri("http://20.106.188.85:5000")
mlflow.set_experiment("Clase_ml_S7_jocas")

with mlflow.start_run(run_name="AutoML_pycaret"):
    # Setup y entrenamiento
    s = setup(data, target='default', session_id=635,  experiment_name='Clase_ml_S7_jocas') #log_experiment=True,
    
    best = compare_models()
    evaluate_model(best)
    
    # Registrar modelo
    mlflow.sklearn.log_model(best, "mejor_modelo_jocas")
    
    # Extra logs si deseas
    mlflow.log_param("modelo_principal_jocas", str(best))
    
    # Registrar matriz de confusión como imagen
    import matplotlib.pyplot as plt
    #from pycaret.utils import check_metric
    from pycaret.classification import plot_model
    
    plot_model(best, plot='confusion_matrix', save=True)
    mlflow.log_artifact("Confusion Matrix.png")
