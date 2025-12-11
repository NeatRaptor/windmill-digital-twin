from fastapi import FastAPI
import pandas as pd
import json
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Wind Turbine Digital Twin API",
    description="Backend for Unity digital twin integration",
    version="1.0"
)

# -------------------- CORS ENABLED --------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Allow Unity requests
    allow_credentials=True,
    allow_methods=["*"],          # GET, POST, PUT, DELETE, etc.
    allow_headers=["*"],
)
# ------------------------------------------------------

# ---------- 1. Predictive Maintenance ----------
@app.get("/maintenance")
def get_maintenance_schedule():
    df = pd.read_csv("data/processed/maintenance_schedule.csv")
    return df.to_dict(orient="records")


# ---------- 2. Health Report ----------
@app.get("/health")
def get_health_report():
    df = pd.read_csv("data/processed/health_report.csv")
    return df.to_dict(orient="records")


# ---------- 3. Live RUL Predictions ----------
@app.get("/rul")
def get_rul_predictions():
    df = pd.read_csv("data/processed/rul_predictions.csv")
    return df.to_dict(orient="records")


# ---------- 4. RCA Output ----------
@app.get("/rca")
def get_root_cause_analysis():
    df = pd.read_csv("data/processed/anomaly_with_root_cause.csv")
    return df.to_dict(orient="records")


# ---------- 5. Subsystem Contribution Bar Chart ----------
@app.get("/subsystem_contribution")
def get_subsystem_impacts():
    df = pd.read_csv("data/processed/subsystem_weights.csv")
    return df.to_dict(orient="records")


# ---------- 6. Live Telemetry Hook (Optional) ----------
@app.get("/telemetry")
def get_live_telemetry():
    df = pd.read_csv("data/processed/latest_telemetry.csv")
    return df.iloc[-1].to_dict()


# ---------- 7. Unity Heartbeat ----------
@app.get("/ping")
def ping():
    return {"status": "online", "time": str(datetime.now())}

@app.get("/")
def root():
    return {"status": "Digital Twin API is running"}
# ---------- 8. Root Endpoint ----------
