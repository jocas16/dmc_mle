from pycaret.classification import *
import pandas as pd
import mlflow

# activar el mlflow
mlflow.set_experiment('banking-autoML')

df = pd.read_csv('churn_bank_automl.csv')

# Inicializamos el pipeline de pycaret

s= setup(data = df,
         target='cerrara_cuenta',
         session_id=123,
         log_experiment=True,
         experiment_name='banking-autoML',
         log_plots=True,
         silent = True
         )

# comparar modelos y el registro en el mlflow
best_model = compare_models()
final_model = tune_model(best_model)
save_model(final_model, 'modelo_bank_mlflow')