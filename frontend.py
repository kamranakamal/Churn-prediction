import streamlit as st 
import requests


cols = ['Age',
 'Gender',
 'Tenure',
 'Usage Frequency',
 'Support Calls',
 'Payment Delay',
 'Subscription Type',
 'Contract Length',
 'Total Spend',
 'Last Interaction']

API_URL = "http://localhost:8000/predict"

st.title("Customer Churn Prediction")

st.markdown("Enter Your details below")

age = st.number_input("Age", min_value=18, max_value=120, value=30)

gender = st.selectbox("Select your gender", ["Male", "Female"])

tenure = st.number_input('Enter Customer tenure', min_value=0, value=6)

usage = st.number_input('Enter customer usage', min_value=0, value=5)

support_call = st.number_input('Number of support calls', min_value=0, value=2)

payment_delay = st.number_input('Enter customer payment delay', min_value=0, value=4)

subscription_type = st.selectbox('Enter subscription type', ['Basic', 'Standard', 'Premium'])

contract_length = st.selectbox('Contract Length',['Monthly', 'Quarterly', 'Annual'])

total_spend = st.number_input('Enter Total Spend', min_value=0, value=700)

last_interaction = st.number_input('Last Interaction ', min_value=0, value=7)

if st.button("Make prediction"):
    data = {}
    input_data = [age, gender, tenure, usage, support_call, payment_delay, subscription_type, contract_length, total_spend, last_interaction]
    for i in range(len(cols)):
        data[cols[i]] = input_data[i]
    
    try:
        response = requests.post(API_URL, json=data)
        if response.status_code==200:
            result = response.json()
            st.success(f"Churn Prediction results: {result}")

        else:
            st.error(f"API Error: {response.status_code} - { response.text}")
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to the FastAPI server. Mkae sure its running on port 8000")









