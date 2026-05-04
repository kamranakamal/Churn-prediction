from fastapi import FastAPI
from fastapi.responses import JSONResponse
import joblib 
from schema.input_data import Customer
from predict import model, MODEL_VERSION, predict_out


app = FastAPI()


total_prediction = 0 


@app.post('/predict')
def predict(data:Customer):
    prediction = predict_out(data)
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


    



