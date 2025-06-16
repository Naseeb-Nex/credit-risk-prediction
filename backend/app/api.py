# backend/app/api.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .schemas import CreditRiskRequest, CreditRiskResponse
from .model import predict_risk

app = FastAPI(
    title="Credit Risk Prediction API",
    version="1.0"
)

# ----------------------------
# CORS Middleware Configuration
# ----------------------------
origins = [
    "http://localhost:3000",         # your React dev server
    "http://127.0.0.1:3000",         # sometimes fetch uses 127.0.0.1
    # add production domains here when you deploy, e.g. "https://mydomain.com"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,           # or ["*"] to allow any origin
    allow_credentials=True,
    allow_methods=["*"],             # GET, POST, PUT, etc.
    allow_headers=["*"],             # Authorization, Content-Type, etc.
)

@app.post(
    "/predict",
    response_model=CreditRiskResponse,
    summary="Predict credit risk"
)
def predict(req: CreditRiskRequest):
    try:
        label, prob = predict_risk(req.dict())
        return CreditRiskResponse(risk=label, probability=prob)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
