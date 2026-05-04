import joblib
import pandas as pd
try:
    preprocessor = joblib.load('./Models/preprocessor.pkl')
    model = joblib.load("./Models/model.pkl")
except Exception as e:
    raise RuntimeError(f"Model loading failed: {e}")


MODEL_VERSION = '1.0.0'

def predict_out(data):
    data = data.model_dump(by_alias=True)
    df = pd.DataFrame([data])
    data_enc = preprocessor.transform(df)
    prediction_class = model.predict(data_enc)[0]
    pred_probs = model.predict_proba(data_enc)[0]

    return prediction_class, pred_probs