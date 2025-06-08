from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from pycaret.classification import load_model,predict_model

#Voy a crear una instancia para llamar al servicio
app = FastAPI()
model = load_model("modelo_responde_oferta_mlflow")
#Voy a crear una clase cliente, DONDE  ESTE LA ESTRUCTURA DE ESTE.
class Cliente(BaseModel):
    edad: int
    genero:str
    ingreso_mensual:float
    nivel_educacion:str
    usa_app:int
    usa_web:int
    satisfaccion:int
    num_productos:int
    reclamos_ult_6m:int
    tasa_credito:float
    region:str

#Para crear el API--http://1.1.1.1/predict/
@app.post("/predict")
def predict(cliente:Cliente):
    data = pd.DataFrame([cliente.dict()])
    pred = predict_model(model,data=data)
    #Esto es la respuesta del api
    return {
        "score":float(pred['prediction_score'][0]),
        "prediccion":int(pred['prediction_label'][0])
    }