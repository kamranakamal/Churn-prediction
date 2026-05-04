import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier
from utils import train_model, save_pickle_obj, data_process, eval_fn

X_train, X_test, y_train, y_test = data_process("./data/customer_churn_dataset-testing-master.csv")

results = train_model(X_train, y_train, X_test, y_test)
print(results[:5])








