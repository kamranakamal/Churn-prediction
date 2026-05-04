# Customer Churn Analysis

## Overview
This project trains a churn prediction model, serves it via a FastAPI endpoint, and includes a Streamlit UI for interactive predictions.

## Project Structure
```
Backend/
	main.py
	requirements.txt
	Models/
data/
	customer_churn_dataset-testing-master.csv
notebooks/
	main.ipynb
scripts/
	model_train.py
	utils.py
frontend.py
requirements.txt
readme.md
```

## Features
- Churn prediction model training
- FastAPI inference endpoint
- Streamlit web app for interactive input

## Installation
Install the base dependencies:
```bash
pip install -r requirements.txt
```

If you only want the API dependencies:
```bash
pip install -r Backend/requirements.txt
```

## Usage

### Train the model
```bash
python scripts/model_train.py
```
This creates the artifacts in `Backend/Models/` used by the API.

### Run the API
```bash
uvicorn Backend.main:app --reload
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
Predictions are returned from the `/predict` endpoint and the Streamlit UI. The `/metrics` endpoint returns total API prediction count.

## Screenshots
Add Streamlit UI screenshots here.

## License
MIT License
