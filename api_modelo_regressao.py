from pydantic import BaseModel
from fastapi import FastAPI
import uvicorn
import joblib

# Criar uma instacia do FastAPI
app = FastAPI()

# Carregar o modelo treinado

modelo_pontuacao = joblib.load('./modelo_regressao_linear.pkl')

# Criar uma classe para receber os dados de entrada

class Item(BaseModel):
    horas_estudo: float

@app.post('/predict')
def predict(data: Item):
    # Preparar os dados para predição
    input_feature = [[data.horas_estudo]]
    # Realizar a predição
    y_pred = modelo_pontuacao.predict(input_feature)[0].astype(int)
    return {'Pontuacao': y_pred.tolist()}