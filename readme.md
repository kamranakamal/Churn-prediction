# Customer Churn Analysis

## Overview
This project trains a churn prediction model, serves it via a FastAPI endpoint, and includes a Streamlit UI for interactive predictions.

## Features
- Churn prediction model training
- FastAPI inference endpoint
- Streamlit web app for interactive input

## Installation
Clone the repository and install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Train the model
```bash
python scripts/model_train.py
```

### Run the API
```bash
uvicorn main:app --reload
```

### Run the Streamlit app
```bash
streamlit run frontend.py
```
Then open http://localhost:8501 in your browser.

## Data
The dataset contains customer information including:
- Customer demographics
- Account tenure
- Service usage
- Churn status

## Results
Predictions are returned from the `/predict` endpoint and the Streamlit UI.

## Screenshots
Add Streamlit UI screenshots here.

## License
MIT License
