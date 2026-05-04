from fastapi import FastAPI
from fastapi.responses import JSONResponse
import joblib 
from schema.input_data import Customer
import pandas as pd



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


    



