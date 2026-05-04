import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier
import os 


def save_pickle_obj(obj, filename):

    if filename.split(".")[-1]=="pkl":
        joblib.dump(obj, filename='./Models/'+filename)
    else:
        joblib.dump(obj, filename=f"./Models/{filename}.pkl")




def data_process(path):
    cols = ['Age',
        'Gender',
        'Tenure',
        'Usage Frequency',
        'Support Calls',
        'Payment Delay',
        'Subscription Type',
        'Contract Length',
        'Total Spend',
        'Last Interaction',
        'Churn']
    df = pd.read_csv(path)
    df = df[cols]
    


    X = df.drop("Churn", axis=1)
    y = df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    cat_col = ['Subscription Type', 'Contract Length', 'Gender']
    num_col = [col for col in X_train.columns.tolist() if col not in cat_col]

    cat_pipeline = Pipeline([
        ('ohe', OneHotEncoder(handle_unknown='ignore'))
    ])

    num_pipline = Pipeline([
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer([
        ('ohe', cat_pipeline, cat_col),
        ('scaler', num_pipline, num_col)
    ])

    X_train_encoded = preprocessor.fit_transform(X_train)
    X_test_encoded = preprocessor.transform(X_test)
    return X_train_encoded, X_test_encoded, y_train, y_test

def eval_fn(y_true, y_pred):
    """Returns Classification Metrics 
    to evaluatae the model performance.

    Args:
        y_true (array): True Values of Target
        y_pred (array): Predicted Values of Model

    Returns:
        Accuracy: Model Accuracy
        recall: Recall value 
        precesion: Precision value of model 
        f1: F1 Score, It's the harmonic mean of the Recall and Precision 
        roc_auc_curve : Roc-Auc Curve
        confus_mat: Confusion Matrix of the model 

    """

    acc = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    confus_mat = confusion_matrix(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc_curve = roc_auc_score(y_true, y_pred)
    return [acc, recall, precision, f1, roc_auc_curve, confus_mat]


def train_model(X_train, y_train, X_test=None, y_test=None):
    model = XGBClassifier()
    model.fit(X_train,y_train)
    save_pickle_obj(model, 'model.pkl')

    
    if X_test is not None and y_test is not None:
        y_pred = model.predict(X_test)
        results = eval_fn(y_test, y_pred)
        return results
    elif X_test is not None:
        return model.predict(X_test)


