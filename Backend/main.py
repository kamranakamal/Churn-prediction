from fastapi import FastAPI
from fastapi.responses import JSONResponse
import joblib 
from schema.input_data import Customer
from predict import model, MODEL_VERSION, predict_out


app = FastAPI()


total_prediction = 0 


@app.post('/predict')
def predict(data:Customer):
    try:
        prediction, pred_probs, conf_score = predict_out(data)
        pred_probs = {i:pred_probs[i].item() for i in range(2)}
    except Exception as e:
        return JSONResponse(status_code=500, content=str(e))
    global total_prediction
    total_prediction+=1 

    return JSONResponse(status_code=200, content={"prediction": int(prediction),
        "label": "Yes" if prediction == 1 else "No",
        "confidence score":conf_score,
        "probabilities":pred_probs
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


    



