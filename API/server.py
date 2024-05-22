import pickle
import pandas as pd
import numpy as np
from fastapi import FastAPI, Request


def load_pickle(filename):
    with open(filename, 'rb') as file:
        contents = pickle.load(file)
    return contents


model_path = 'pricing_model.pkl'
model = load_pickle(model_path)

def make_prediction(data, model):
    return model.predict(data)


app_name = 'House Price Prediction'
app_version = '1.0'

app = FastAPI(
    title=app_name,
    description='Model to predict house prices',
    version=app_version)



@app.get('/')
def index():
    return {app_name:app_version}

@app.post('/predict')
async def single_predict(request: Request):
    body = await request.json()
    df = pd.DataFrame.from_dict(body)
    price = np.round(model.predict(df)[0].item(),2)

    return {'predicted_price': price}

# @app.post('/predict')
# async def predict(tamanho: float, ano: int,garagem:int):
#     data = {'tamanho': [tamanho], 'ano': [ano],'garagem':[garagem]}
#     df = pd.DataFrame.from_dict(data)
#     price = str(make_prediction(df, model)[0])
#     return {'predicted_price': price}




