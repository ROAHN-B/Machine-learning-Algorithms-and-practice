from fastapi import FastAPI
from pydantic import BaseModel
import joblib 
import numpy as np

app=FastAPI(Titlt="Telco Churn prediction API")

knn=joblib.load("models/knn_model.pkl")
log_reg=joblib.load("models/logistic_regression_model.pkl")
scaler=joblib.load("models/scaler.pkl")

