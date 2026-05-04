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
    conf_score = pred_probs[prediction_class].item()
    pred_probs = {i:pred_probs[i].item() for i in range(2)}
    label = "Yes" if prediction_class == 1 else "No"
    prediction_dict = {"prediction": int(prediction_class),
        "label": label,
        "confidence score":conf_score,
        "probabilities":pred_probs
    }

    return prediction_dict