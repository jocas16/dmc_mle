from pycaret.classification import * 
#import mlflow
import pandas as pd
#activar el mlflow
#mlflow.set_experiment("banking-autoML")
df = pd.read_csv("hbr_caso_cliente_responde_oferta.csv")
#Iniciarmos el pipeline de pycaret
s = setup(data=df,
          target='respondio_oferta',
          session_id=131,
          log_experiment=True,
          ignore_features=['cliente_id'],
          experiment_name="respuesta-autoML",
          log_plots=True,
          )
#Comparar modelos y el registro en el mlflow
best_model = compare_models()
final_model = tune_model(best_model)
save_model(final_model,"modelo_responde_oferta_mlflow")