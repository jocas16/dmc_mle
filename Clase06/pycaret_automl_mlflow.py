# pycaret_automl_mlflow.py
import mlflow
from pycaret.classification import *
import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore")

# Ruta absoluta al CSV
data = pd.read_csv("/home/ubuntu/data/credit_data.csv")

# Configuración de MLflow
mlflow.set_tracking_uri("http://20.106.188.85:5000")
mlflow.set_experiment("Clase_PyCaret_AutoML_2025")

with mlflow.start_run(run_name="AutoML_pycaret"):
    # Setup y entrenamiento
    s = setup(data, target='default', session_id=123, log_experiment=True, experiment_name='Clase_PyCaret_AutoML_2025', silent=True)
    
    best = compare_models()
    evaluate_model(best)
    
    # Registrar modelo
    mlflow.sklearn.log_model(best, "mejor_modelo")
    
    # Extra logs si deseas
    mlflow.log_param("modelo_principal", str(best))
    
    # Registrar matriz de confusión como imagen
    import matplotlib.pyplot as plt
    from pycaret.utils import check_metric
    from pycaret.classification import plot_model
    
    plot_model(best, plot='confusion_matrix', save=True)
    mlflow.log_artifact("Confusion Matrix.png")
