from fastapi import FastAPI
from pydantic import BaseModel
import joblib 
import numpy as np

app=FastAPI(Titlt="Telco Churn prediction API")

knn=joblib.load("models/knn_model.pkl")
log_reg=joblib.load("models/logistic_regression_model.pkl")
scaler=joblib.load("models/scaler.pkl")

app=FastAPI()

class CustomerData(BaseModel):
        SeniorCitizen:int
        tenure:int
        MonthlyCharges: float
        TotalChargers:float
        Partner:int
        Dependents:int
        PhoneService:int
        PaperlessBilling:int
        gender:int
        OnlineSecurity:int
        OnlineBackup:int
        DeviceProtection:int
        TechSupport:int
        StreamingTV:int
        StreamingMovies:int


@app.post("/predict/KNN")
def predict_knn(data:CustomerData):
        x=np.array([[
                data.SeniorCitizen,
                data.tenure,
                data.MonthlyCharges,
                data.TotalChargers,
                data.Partner,
                data.Dependents,
                data.PhoneService,
                data.PaperlessBilling,
                data.gender,
                data.OnlineSecurity,
                data.OnlineBackup,
                data.DeviceProtection,
                data.TechSupport,
                data.StreamingTV,
                data.StreamingMovies


        ]])
        x_scaled=scaler.fit_transform(x)
        pred=knn.predict(x_scaled)[0]

        return {
                "model":"KNN CLASSIFICATION",
                "prediction":"YES" if pred==1 else "NO"
        }


@app.post("/predict/LOGISTIC_REG")
def predict_logistic(data:CustomerData):
        x=np.array([[
               data.SeniorCitizen,
                data.tenure,
                data.MonthlyCharges,
                data.TotalChargers,
                data.Partner,
                data.Dependents,
                data.PhoneService,
                data.PaperlessBilling,
                data.gender,
                data.OnlineSecurity,
                data.OnlineBackup,
                data.DeviceProtection,
                data.TechSupport,
                data.StreamingTV,
                data.StreamingMovies

        ]])

        x_scaled=scaler.fit_transform(x)
        pred=log_reg.predict(x_scaled)[0]

        return {
                "Model":"LOGISTIC REGRESSION",
                "PREDICTION":"YES" if pred==1 else "NO"
        }
