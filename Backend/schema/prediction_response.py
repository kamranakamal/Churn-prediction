from pydantic import BaseModel, Field
from typing import Dict

class PredictionResponsse(BaseModel):
    prediction : int = Field(..., description="Model's Prediction Output", alias='prediction', examples=[0,1])

    label: str = Field(..., description="Model's prediction label class", alias='label', examples=['No', 'Yes'])

    conf_score:float = Field(..., description="Confidence score of model prediction", alias='confidence score', examples=[0.98999, 0.34533, 0.64546446, 0.733434])

    probabilities: Dict[int, float] = Field(..., description='Prediction probabilities of all class labels', alias='probabilities', examples=[{0:0.00003434, 1:0.99883342}])

