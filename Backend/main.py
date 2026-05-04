from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, computed_field, field_validator
from typing import Annotated, Literal
import joblib 
import pandas as pd
import os


class Customer(BaseModel):
    Age:Annotated[int, Field(..., alias='Age',gt=0, lt=150, description='Age of the customer')]
    Gender:Annotated[Literal['Male', 'Female'], Field(...,alias='Gender', description='Gender of the customer')]
    Tenure: Annotated[int, Field(..., alias='Tenure',description='Tenure of the customer', gt=0)]
    Usage_frequency:Annotated[int, Field(..., alias='Usage Frequency')]
    Support_calls: Annotated[int, Field(...,alias='Support Calls', description='Number of support calls')]
    Payment_delay: Annotated[int, Field(..., alias='Payment Delay')]
    Subscription_type:Annotated[Literal['Basic', 'Standard', 'Premium'], Field(..., alias='Subscription Type')]
    Contract_length: Annotated[Literal['Monthly', 'Quarterly', 'Annual'], Field(..., alias='Contract Length')]
    Total_spend: Annotated[int, Field(..., alias='Total Spend')]
    Last_interaction: Annotated[int, Field(..., alias='Last Interaction')]

    @field_validator('Subscription_type', 'Contract_length')
    @classmethod
    def capitalize_col(cls, v:str) -> str:
        v = v.strip().title()
        return v



 

total_prediction = 0 
try:
    preprocessor = joblib.load('./Models/preprocessor.pkl')
    model = joblib.load("./Models/model.pkl")
except Exception as e:
    raise RuntimeError(f"Model loading failed: {e}")
app = FastAPI()

MODEL_VERSION = '1.0.0'


@app.post('/predict')
def predict(data:Customer):
    data = data.model_dump(by_alias=True)
    df = pd.DataFrame([data])
    data_enc = preprocessor.transform(df)
    prediction = model.predict(data_enc)[0]
    global total_prediction
    total_prediction+=1 

    return JSONResponse(status_code=200, content={"prediction": int(prediction),
        "label": "Yes" if prediction == 1 else "No"
    })
      


@app.get('/metrics')
def metrics():
    global total_prediction
    return JSONResponse(status_code=200, content={"total_predictions": total_prediction})

@app.get("/health")
def health():
    return {"status": "ok", 
            "version":MODEL_VERSION,
            "model_loaded": model is not None}

@app.get('/')
def home():
    return {'message': 'Customer churn prediction'}


    



