# Customer Churn Analysis

## Overview
This project trains a churn prediction model and serves it via a FastAPI endpoint.

## Features
- Churn prediction model training
- FastAPI inference endpoint

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

## Data
The dataset contains customer information including:
- Customer demographics
- Account tenure
- Service usage
- Churn status

## Results
Predictions are returned from the `/predict` endpoint.

## License
MIT License
