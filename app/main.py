from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd
import joblib
import uvicorn

try:
    model = joblib.load('best_model.pkl')
    scaler_amount = joblib.load('scaler_amount.pkl')
    scaler_time = joblib.load('scaler_time.pkl')
except FileNotFoundError as e:
    raise RuntimeError(f"Не найден файл модели или scaler'а: {e}")

class Transaction(BaseModel):
    Time: float = Field(..., description="Время от первой транзакции в секундах")
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float = Field(..., description="Сумма транзакции")

class FraudPrediction(BaseModel):
    fraud_probability: float
    is_fraud: bool

app = FastAPI(title="Credit Card Fraud Detection API",
              description="API для предсказания мошеннических транзакций")

@app.on_event("startup")
async def startup_event():
    print("Сервис запущен, модель готова к предсказаниям.")

@app.get("/")
def root():
    return {"message": "Fraud Detection API is running"}

@app.post("/predict", response_model=FraudPrediction)
def predict(transaction: Transaction):
    try:
        input_data = pd.DataFrame([transaction.dict()])

        input_data['Amount'] = scaler_amount.transform(input_data[['Amount']])
        input_data['Time'] = scaler_time.transform(input_data[['Time']])

        features = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9',
                    'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18',
                    'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27',
                    'V28', 'Amount']
        X = input_data[features]

        proba = model.predict_proba(X)[0, 1]
        pred = int(proba > 0.5)

        return FraudPrediction(fraud_probability=float(proba), is_fraud=bool(pred))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)